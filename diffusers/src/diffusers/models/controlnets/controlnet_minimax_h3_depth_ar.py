# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

"""Explicit AR-role depth ControlNet wrapper.

The generic H3 depth branch remains the implementation of the residual math.
This role wrapper makes the AR contract visible: depth rows are target-only,
history is optional side metadata, and offline/causal profiles cannot be
silently mixed.
"""

from __future__ import annotations

from typing import Any

import torch

from .controlnet_minimax_h3 import (
    MiniMaxH3DepthControlNetModel,
    MiniMaxH3DepthControlOutput,
)


class MiniMaxH3DepthARControlNetModel(MiniMaxH3DepthControlNetModel):
    """Target-row depth adapter for the H3 chunk-AR role."""

    is_ar_role = True

    def forward(
        self,
        depth_rows: torch.Tensor,
        noisy_video_rows: torch.Tensor | None = None,
        timestep: torch.Tensor | float | None = None,
        depth_valid: torch.Tensor | None = None,
        target_position_ids: torch.Tensor | None = None,
        *,
        depth_profile: str | None = None,
        history_rows: torch.Tensor | None = None,
        history_valid: torch.Tensor | None = None,
        row_metadata: Any | None = None,
        **kwargs,
    ) -> MiniMaxH3DepthControlOutput | tuple[torch.Tensor, ...]:
        if depth_profile is not None and depth_profile not in {
            "offline_global_depth",
            "causal_live_depth",
        }:
            raise ValueError(f"unsupported AR depth profile: {depth_profile!r}")
        if history_rows is not None:
            if history_rows.ndim != 3 or history_rows.shape[0] != depth_rows.shape[0]:
                raise ValueError("history_rows must be [B,N,C] when supplied")
            if history_valid is not None and tuple(history_valid.shape[:2]) != tuple(history_rows.shape[:2]):
                raise ValueError("history_valid must align with history_rows")
        if row_metadata is not None:
            try:
                if len(row_metadata) != depth_rows.shape[1]:
                    raise ValueError("AR depth row metadata must cover target rows exactly")
            except TypeError as error:
                raise ValueError("row_metadata must be a sized sequence") from error
        return super().forward(
            depth_rows=depth_rows,
            noisy_video_rows=noisy_video_rows,
            timestep=timestep,
            depth_valid=depth_valid,
            target_position_ids=target_position_ids,
            **kwargs,
        )


__all__ = ["MiniMaxH3DepthARControlNetModel"]
