"""Small, explicit chunk-AR denoise loop built on H3 scheduler algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from utility.h3_depth_ar.ar_sampler import (
    H3ARSchedule,
    build_ar_schedule,
    sample_with_velocity,
)


@dataclass(frozen=True)
class ChunkARDenoiseConfig:
    """Frozen denoise settings that are bound into an AR checkpoint."""

    role: str = "A50-depth"
    depth_profile: str = "causal_live_depth"
    forward_count: int = 4
    shift: float = 12.0
    schedule_profile: str = "A50-varlen"

    def validate(self) -> None:
        if self.role not in {"A50-depth", "B50-depth", "C4-depth", "C2-depth"}:
            raise ValueError(f"unsupported chunk-AR role: {self.role!r}")
        if self.depth_profile not in {"offline_global_depth", "causal_live_depth"}:
            raise ValueError(f"unsupported depth profile: {self.depth_profile!r}")
        if self.forward_count <= 0:
            raise ValueError("forward_count must be positive")
        if not torch.isfinite(torch.tensor(float(self.shift))) or self.shift <= 0:
            raise ValueError("shift must be finite and positive")

    def schedule(self, *, device: torch.device | str | None = None) -> H3ARSchedule:
        self.validate()
        return build_ar_schedule(
            forward_count=self.forward_count,
            shift=self.shift,
            profile=self.schedule_profile,
            device=device,
        )


def denoise_chunk(
    sample: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    config: ChunkARDenoiseConfig,
    *,
    condition: Any = None,
) -> tuple[torch.Tensor, H3ARSchedule]:
    """Denoise one bootstrap/steady chunk with a fixed H3 schedule.

    The callback is responsible for constructing the causal row layout and
    selecting the depth profile.  This function only owns scheduler points and
    shape/finite checks, which keeps it usable by both training probes and the
    eventual streaming runtime.
    """

    if sample.ndim < 1 or not bool(torch.isfinite(sample).all()):
        raise ValueError("chunk sample must be finite")
    schedule = config.schedule(device=sample.device)
    result = sample_with_velocity(sample, velocity_fn, schedule, condition=condition)
    if tuple(result.shape) != tuple(sample.shape) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("chunk-AR denoise returned an invalid sample")
    return result, schedule


__all__ = ["ChunkARDenoiseConfig", "denoise_chunk"]
