"""MiniMax-H3 noise schedules."""

from __future__ import annotations

import torch

def shift_noise_ratio(noise_ratio: torch.Tensor, shift: float = 12.0) -> torch.Tensor:
    """Apply MiniMax-H3's exponential video schedule shift to an RF noise ratio."""
    if shift <= 0:
        raise ValueError("shift must be positive.")
    noise_ratio = torch.as_tensor(noise_ratio)
    if not bool(torch.isfinite(noise_ratio).all()) or bool(((noise_ratio < 0) | (noise_ratio > 1)).any()):
        raise ValueError("noise_ratio must contain finite values in [0, 1].")
    return shift * noise_ratio / (1.0 + (shift - 1.0) * noise_ratio)


def shifted_sigma_schedule(
    num_forwards: int,
    *,
    shift: float = 12.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``num_forwards + 1`` descending MiniMax-H3 noise boundaries, including 1 and 0."""
    if num_forwards < 1:
        raise ValueError("num_forwards must be positive.")
    base = torch.linspace(1.0, 0.0, num_forwards + 1, dtype=torch.float32)
    schedule = shift_noise_ratio(base, shift=shift)
    if not bool((schedule[1:] < schedule[:-1]).all()) or schedule[-1].item() != 0.0:
        raise AssertionError("The shifted H3 schedule must be strictly descending and terminate at zero.")
    return schedule.to(device)
