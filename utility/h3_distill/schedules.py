"""Noise schedules shared by H3 dCM training and few-forward sampling."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Return ``num_forwards + 1`` descending H3/rCM noise boundaries, including 1 and 0."""
    if num_forwards < 1:
        raise ValueError("num_forwards must be positive.")
    base = torch.linspace(1.0, 0.0, num_forwards + 1, dtype=torch.float32)
    schedule = shift_noise_ratio(base, shift=shift)
    if not bool((schedule[1:] < schedule[:-1]).all()) or schedule[-1].item() != 0.0:
        raise AssertionError("The shifted H3 schedule must be strictly descending and terminate at zero.")
    return schedule.to(device)


@dataclass(frozen=True)
class DcmInterval:
    """One teacher transition from a noisier RF boundary to a cleaner boundary."""

    noisy_u: torch.Tensor
    clean_u: torch.Tensor
    index: int
    skip_steps: int

    @property
    def noisy_h3_t(self) -> torch.Tensor:
        return 1.0 - self.noisy_u

    @property
    def clean_h3_t(self) -> torch.Tensor:
        return 1.0 - self.clean_u


def sample_dcm_interval(
    teacher_schedule: torch.Tensor,
    *,
    skip_steps: int = 1,
    generator: torch.Generator | None = None,
) -> DcmInterval:
    """Uniformly select one valid discrete teacher interval from a descending sigma schedule."""
    teacher_schedule = torch.as_tensor(teacher_schedule, dtype=torch.float32)
    if teacher_schedule.ndim != 1 or teacher_schedule.numel() < 2:
        raise ValueError("teacher_schedule must be a 1-D tensor with at least two boundaries.")
    if skip_steps < 1 or skip_steps >= teacher_schedule.numel():
        raise ValueError("skip_steps must select at least one transition and leave a valid endpoint.")
    if not bool((teacher_schedule[1:] < teacher_schedule[:-1]).all()):
        raise ValueError("teacher_schedule must be strictly descending from noise toward clean data.")
    high = teacher_schedule.numel() - skip_steps
    index = int(torch.randint(high, (), generator=generator, device=teacher_schedule.device).item())
    return DcmInterval(
        noisy_u=teacher_schedule[index],
        clean_u=teacher_schedule[index + skip_steps],
        index=index,
        skip_steps=skip_steps,
    )
