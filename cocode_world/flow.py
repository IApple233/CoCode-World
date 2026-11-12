"""Flow interpolation and Euler integration for MiniMax-H3."""

from __future__ import annotations

import torch

def _broadcast_timestep(timestep: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    timestep = torch.as_tensor(timestep, device=reference.device, dtype=reference.dtype)
    while timestep.ndim < reference.ndim:
        timestep = timestep.unsqueeze(-1)
    try:
        torch.broadcast_shapes(timestep.shape, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"timestep shape {tuple(timestep.shape)} is not broadcastable to sample shape {tuple(reference.shape)}"
        ) from exc
    if not bool(torch.isfinite(timestep).all()) or bool(((timestep < 0) | (timestep > 1)).any()):
        raise ValueError("MiniMax-H3 timesteps must be finite values in [0, 1].")
    return timestep


def h3_interpolate(clean: torch.Tensor, noise: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
    """Apply H3's forward process: ``x_t = t * x0 + (1 - t) * noise``."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean and noise shapes differ: {tuple(clean.shape)} != {tuple(noise.shape)}")
    timestep = _broadcast_timestep(timestep, clean)
    return timestep * clean + (1.0 - timestep) * noise


def h3_euler_between(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    timestep: float | torch.Tensor,
    next_timestep: float | torch.Tensor,
) -> torch.Tensor:
    """Take a deterministic H3 Euler/PF-ODE step from ``t`` toward cleaner ``next_t``.

    Both time arguments use H3's convention: zero is noise and one is clean.
    The function computes the blend in float32 for half-precision inputs, as the
    released H3 scheduler does.
    """
    if sample.shape != velocity.shape:
        raise ValueError(f"sample and velocity shapes differ: {tuple(sample.shape)} != {tuple(velocity.shape)}")
    timestep = _broadcast_timestep(timestep, sample)
    next_timestep = _broadcast_timestep(next_timestep, sample)
    if bool((next_timestep < timestep).any()):
        raise ValueError("next_timestep must move toward clean data (next_timestep >= timestep).")
    sigma = 1.0 - timestep
    if bool((sigma <= 0).any()):
        raise ValueError("An Euler step cannot start at the terminal clean timestep t=1.")

    compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    sample_compute = sample.to(compute_dtype)
    velocity_compute = velocity.to(compute_dtype)
    sigma_compute = sigma.to(compute_dtype)
    next_sigma_compute = (1.0 - next_timestep).to(compute_dtype)
    clean = sample_compute + sigma_compute * velocity_compute
    ratio = next_sigma_compute / sigma_compute
    return ratio * sample_compute + (1.0 - ratio) * clean
