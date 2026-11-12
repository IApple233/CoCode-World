"""Pure DMD/DFD losses and memory-bounded gradient replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .dfd import build_dfd_teacher_input


PredictClean = Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor]


@dataclass
class DmdStudentOutput:
    loss: torch.Tensor
    generated: torch.Tensor
    teacher_score_data: torch.Tensor
    fake_score_input: torch.Tensor
    teacher_score_input: torch.Tensor
    teacher_x0: torch.Tensor
    fake_x0: torch.Tensor
    weight: torch.Tensor
    gradient: torch.Tensor
    pseudo_target: torch.Tensor
    replay_reference: torch.Tensor | None = None


@dataclass
class FakeScoreOutput:
    loss: torch.Tensor
    generated: torch.Tensor
    perturbed: torch.Tensor
    predicted_x0: torch.Tensor
    target_x0: torch.Tensor


def h3_score_perturb(clean: torch.Tensor, noise: torch.Tensor, u: float | torch.Tensor) -> torch.Tensor:
    """Apply the H3 RF forward process where u=0 is clean and u=1 is noise."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean and noise shapes differ: {tuple(clean.shape)} != {tuple(noise.shape)}.")
    u_tensor = torch.as_tensor(u, device=clean.device, dtype=clean.dtype)
    while u_tensor.ndim < clean.ndim:
        u_tensor = u_tensor.unsqueeze(-1)
    try:
        torch.broadcast_shapes(u_tensor.shape, clean.shape)
    except RuntimeError as exc:
        raise ValueError(f"u shape {tuple(u_tensor.shape)} is not broadcastable to {tuple(clean.shape)}.") from exc
    if not bool(torch.isfinite(u_tensor).all()) or bool(((u_tensor < 0) | (u_tensor > 1)).any()):
        raise ValueError("u must contain finite values in [0, 1].")
    return (1.0 - u_tensor) * clean + u_tensor * noise


def dmd_vsd_gradient(
    generated: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_x0: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    additional_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the detached upstream VSD pseudo-gradient and per-sample normalization."""
    if generated.shape != teacher_x0.shape or generated.shape != fake_x0.shape:
        raise ValueError("generated, teacher_x0 and fake_x0 must have identical shapes.")
    if generated.ndim < 2:
        raise ValueError("DMD expects a batch dimension and at least one feature dimension.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    dims = tuple(range(1, generated.ndim))
    with torch.no_grad():
        difference = generated.float() - teacher_x0.float()
        weight_fp32 = difference.abs().mean(dim=dims, keepdim=True).add(epsilon).reciprocal()
        if additional_scale is not None:
            scale = torch.as_tensor(additional_scale, device=generated.device, dtype=torch.float32)
            while scale.ndim < weight_fp32.ndim:
                scale = scale.unsqueeze(-1)
            try:
                weight_fp32 = weight_fp32 * torch.broadcast_to(scale, weight_fp32.shape)
            except RuntimeError as exc:
                raise ValueError("additional_scale is not broadcastable to the per-sample weight.") from exc
        weight = weight_fp32.to(generated.dtype)
        gradient = (fake_x0 - teacher_x0) * weight
    return gradient.detach(), weight.detach()


def dmd_pseudo_loss(generated: torch.Tensor, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject a fixed DMD pseudo-gradient through a half-MSE objective."""
    if generated.shape != gradient.shape:
        raise ValueError("generated and gradient must have identical shapes.")
    pseudo_target = generated.detach() - gradient.detach()
    loss = 0.5 * F.mse_loss(generated, pseudo_target, reduction="mean")
    return loss, pseudo_target


def fake_score_dsm_loss(predicted_x0: torch.Tensor, generated_x0: torch.Tensor) -> torch.Tensor:
    """Train fake-score only on detached generated x0, never on forced real data."""
    if predicted_x0.shape != generated_x0.shape:
        raise ValueError("predicted_x0 and generated_x0 must have identical shapes.")
    return F.mse_loss(predicted_x0.float(), generated_x0.detach().float(), reduction="mean")


def _score_student_generation(
    generated: torch.Tensor,
    real: torch.Tensor,
    score_noise: torch.Tensor,
    score_u: float | torch.Tensor,
    *,
    teacher_predict: PredictClean,
    fake_score_predict: PredictClean,
    use_real_teacher_score: bool,
) -> tuple[torch.Tensor, ...]:
    teacher_score_data = build_dfd_teacher_input(generated, real, use_real_teacher_score)
    fake_score_input = h3_score_perturb(generated, score_noise, score_u)
    teacher_score_input = h3_score_perturb(teacher_score_data, score_noise, score_u)
    with torch.no_grad():
        fake_x0 = fake_score_predict(fake_score_input, score_u)
        teacher_x0 = teacher_predict(teacher_score_input, score_u)
        gradient, weight = dmd_vsd_gradient(generated, teacher_x0, fake_x0)
    return (
        teacher_score_data,
        fake_score_input,
        teacher_score_input,
        teacher_x0,
        fake_x0,
        gradient,
        weight,
    )


def dmd_student_loss_naive(
    student_input: torch.Tensor,
    real: torch.Tensor,
    score_noise: torch.Tensor,
    student_u: float | torch.Tensor,
    score_u: float | torch.Tensor,
    *,
    student_predict: PredictClean,
    teacher_predict: PredictClean,
    fake_score_predict: PredictClean,
    use_real_teacher_score: bool,
) -> DmdStudentOutput:
    """Single-graph reference used only for correctness tests."""
    generated = student_predict(student_input, student_u)
    scored = _score_student_generation(
        generated,
        real,
        score_noise,
        score_u,
        teacher_predict=teacher_predict,
        fake_score_predict=fake_score_predict,
        use_real_teacher_score=use_real_teacher_score,
    )
    loss, pseudo_target = dmd_pseudo_loss(generated, scored[5])
    return DmdStudentOutput(loss, generated, *scored[:5], scored[6], scored[5], pseudo_target)


def dmd_student_loss_replay(
    student_input: torch.Tensor,
    real: torch.Tensor,
    score_noise: torch.Tensor,
    student_u: float | torch.Tensor,
    score_u: float | torch.Tensor,
    *,
    student_predict: PredictClean,
    teacher_predict: PredictClean,
    fake_score_predict: PredictClean,
    use_real_teacher_score: bool,
    replay_atol: float = 1e-6,
    replay_rtol: float = 1e-5,
) -> DmdStudentOutput:
    """Recompute only the student graph after detached teacher/fake-score scoring."""
    with torch.no_grad():
        generated_reference = student_predict(student_input, student_u)
        scored = _score_student_generation(
            generated_reference,
            real,
            score_noise,
            score_u,
            teacher_predict=teacher_predict,
            fake_score_predict=fake_score_predict,
            use_real_teacher_score=use_real_teacher_score,
        )
    generated = student_predict(student_input, student_u)
    if not torch.allclose(generated.detach(), generated_reference, atol=replay_atol, rtol=replay_rtol):
        maximum = float((generated.detach().float() - generated_reference.float()).abs().max())
        raise RuntimeError(f"Student gradient replay changed its output; max_abs={maximum:.8g}.")
    loss, pseudo_target = dmd_pseudo_loss(generated, scored[5])
    return DmdStudentOutput(
        loss,
        generated,
        *scored[:5],
        scored[6],
        scored[5],
        pseudo_target,
        replay_reference=generated_reference,
    )


def fake_score_loss(
    student_input: torch.Tensor,
    score_noise: torch.Tensor,
    student_u: float | torch.Tensor,
    score_u: float | torch.Tensor,
    *,
    student_predict: PredictClean,
    fake_score_predict: PredictClean,
) -> FakeScoreOutput:
    """Compute the generated-only fake-score DSM update."""
    with torch.no_grad():
        generated = student_predict(student_input, student_u)
        perturbed = h3_score_perturb(generated, score_noise, score_u)
    predicted_x0 = fake_score_predict(perturbed, score_u)
    loss = fake_score_dsm_loss(predicted_x0, generated)
    return FakeScoreOutput(loss, generated, perturbed, predicted_x0, generated.detach())
