"""Few-forward consistency sampler for the H3 rCM student."""

from __future__ import annotations

import torch

from .rcm_adapter import H3RcmAdapter, H3Role


def _validated_boundaries(boundaries: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    boundaries = torch.as_tensor(boundaries, dtype=torch.float32, device=reference.device)
    if boundaries.ndim != 1 or boundaries.numel() < 2:
        raise ValueError("boundaries must be a 1-D schedule with at least two points.")
    if boundaries[0].item() != 1.0 or boundaries[-1].item() != 0.0:
        raise ValueError("Consistency boundaries must start at u=1 and end at u=0.")
    if not bool((boundaries[1:] < boundaries[:-1]).all()):
        raise ValueError("Consistency boundaries must be strictly descending.")
    return boundaries


@torch.no_grad()
def consistency_sample(
    adapter: H3RcmAdapter,
    initial_noise: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    role: H3Role = "student_ema",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate with one clean prediction per interval and stochastic re-noising between boundaries."""
    boundaries = _validated_boundaries(boundaries, initial_noise)

    sample = initial_noise
    for index, u in enumerate(boundaries[:-1]):
        clean = adapter.predict_clean(sample, u, role)
        next_u = boundaries[index + 1]
        if next_u.item() == 0.0:
            return clean.float()
        noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
        sample = (1.0 - next_u) * clean + next_u * noise
    raise AssertionError("The terminal consistency boundary was not reached.")


@torch.no_grad()
def ode_sample(
    adapter: H3RcmAdapter,
    initial_noise: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    role: H3Role = "student_ema",
) -> torch.Tensor:
    """Generate with deterministic PF-ODE updates between clean predictions.

    H3's rectified-flow coordinate is ``x_u = (1-u) * x0 + u * noise``.
    Holding the predicted clean sample fixed over an interval therefore gives
    ``x_next = (u_next/u) * x_u + (1-u_next/u) * x0``.  The blend is computed
    in float32 for half-precision inputs, matching H3's Euler implementation.
    """
    boundaries = _validated_boundaries(boundaries, initial_noise)

    sample = initial_noise
    for index, u in enumerate(boundaries[:-1]):
        clean = adapter.predict_clean(sample, u, role)
        next_u = boundaries[index + 1]
        if next_u.item() == 0.0:
            return clean.float()
        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        ratio = (next_u / u).to(dtype=compute_dtype)
        sample_compute = sample.to(dtype=compute_dtype)
        clean_compute = clean.to(dtype=compute_dtype)
        sample = ratio * sample_compute + (1.0 - ratio) * clean_compute
    raise AssertionError("The terminal consistency boundary was not reached.")
