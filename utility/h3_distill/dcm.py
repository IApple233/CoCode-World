"""JVP-free discrete consistency training for the H3 video student."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .consistency import h3_consistency_mse, h3_euler_between, h3_interpolate
from .rcm_adapter import H3RcmAdapter
from .schedules import DcmInterval


@dataclass
class DcmStepOutput:
    loss: torch.Tensor
    predicted_clean: torch.Tensor
    target_clean: torch.Tensor
    teacher_endpoint: torch.Tensor
    noisy_sample: torch.Tensor
    interval: DcmInterval


def h3_dcm_loss(
    adapter: H3RcmAdapter,
    clean: torch.Tensor,
    noise: torch.Tensor,
    teacher_schedule: torch.Tensor,
    interval: DcmInterval,
) -> DcmStepOutput:
    """Build one rCM dCM target using H3-native time and velocity conventions."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean and noise shapes differ: {tuple(clean.shape)} != {tuple(noise.shape)}.")
    if interval.index + interval.skip_steps >= teacher_schedule.numel():
        raise ValueError("The dCM interval extends past the teacher schedule.")
    expected_noisy = teacher_schedule[interval.index]
    expected_clean = teacher_schedule[interval.index + interval.skip_steps]
    if not torch.equal(interval.noisy_u, expected_noisy) or not torch.equal(interval.clean_u, expected_clean):
        raise ValueError("The dCM interval does not belong to the supplied teacher schedule.")

    noisy_sample = h3_interpolate(clean, noise, 1.0 - interval.noisy_u)
    predicted_clean = adapter.predict_clean(noisy_sample, interval.noisy_u, "student")

    with torch.no_grad():
        teacher_sample = noisy_sample
        for offset in range(interval.skip_steps):
            current_u = teacher_schedule[interval.index + offset]
            next_u = teacher_schedule[interval.index + offset + 1]
            teacher_velocity = adapter.predict_h3_velocity(teacher_sample, current_u, "teacher")
            teacher_sample = h3_euler_between(
                teacher_sample,
                teacher_velocity,
                1.0 - current_u,
                1.0 - next_u,
            )
        target_clean = adapter.predict_clean(teacher_sample, interval.clean_u, "student_ema").detach()

    loss = h3_consistency_mse(predicted_clean, target_clean)
    return DcmStepOutput(
        loss=loss,
        predicted_clean=predicted_clean,
        target_clean=target_clean,
        teacher_endpoint=teacher_sample,
        noisy_sample=noisy_sample,
        interval=interval,
    )


@torch.no_grad()
def sum_context_parallel_gradients(
    parameters: list[torch.nn.Parameter], group: dist.ProcessGroup | None = None
) -> None:
    """Sum replicated parameter gradients whose sequence contributions are split across CP ranks."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)


def optimize_dcm_step(
    adapter: H3RcmAdapter,
    optimizer: torch.optim.Optimizer,
    clean: torch.Tensor,
    noise: torch.Tensor,
    teacher_schedule: torch.Tensor,
    interval: DcmInterval,
    *,
    max_grad_norm: float = 1.0,
    context_parallel_group: dist.ProcessGroup | None = None,
) -> tuple[DcmStepOutput, torch.Tensor]:
    """Run backward, CP gradient reduction, optimizer update and EMA update for one dCM microbatch."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The H3 student has no trainable parameters.")
    if adapter.ema is None:
        raise ValueError("dCM optimization requires a student EMA target.")

    optimizer.zero_grad(set_to_none=True)
    output = h3_dcm_loss(adapter, clean, noise, teacher_schedule, interval)
    if not bool(torch.isfinite(output.loss)):
        raise FloatingPointError(f"Non-finite dCM loss: {float(output.loss.detach())}.")
    output.loss.backward()
    sum_context_parallel_gradients(parameters, context_parallel_group)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError(f"Non-finite dCM gradient norm: {float(grad_norm)}.")
    optimizer.step()
    adapter.ema.update()
    return output, grad_norm
