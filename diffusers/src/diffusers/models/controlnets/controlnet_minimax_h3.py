# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""A small, independent depth adapter for MiniMax-H3 Ref2VA.

The adapter follows the LongVie/ControlNet contract without widening H3's base
``proj_in``: it consumes already patchified depth and noisy target-video rows and
returns zero-initialized residuals at selected H3 blocks.  Reference/image rows
never enter this branch, so appearance remains the responsibility of Ref2VA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from .controlnet import zero_module
from ..modeling_utils import ModelMixin, get_parameter_dtype


@dataclass
class MiniMaxH3DepthControlOutput(BaseOutput):
    """Residual rows, one tensor per configured H3 injection block."""

    control_block_samples: tuple[torch.Tensor, ...]
    injection_layers: tuple[int, ...]


class MiniMaxH3DepthSideBlock(nn.Module):
    """Lightweight adapter block kept at ``adapter_dim`` rather than H3 width."""

    def __init__(self, adapter_dim: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(adapter_dim)
        self.in_proj = nn.Linear(adapter_dim, expansion * adapter_dim)
        self.out_proj = nn.Linear(expansion * adapter_dim, adapter_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = nn.functional.silu(self.in_proj(hidden_states))
        return residual + self.out_proj(hidden_states)


def _validate_layers(layers: Iterable[int], num_layers: int) -> tuple[int, ...]:
    result = tuple(int(layer) for layer in layers)
    if not result:
        raise ValueError("injection_layers must contain at least one block index")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"injection_layers must be sorted and unique, got {result}")
    if result[0] < 0 or result[-1] >= num_layers:
        raise ValueError(f"injection_layers {result} do not fit num_layers={num_layers}")
    return result


class MiniMaxH3DepthControlNetModel(ModelMixin, ConfigMixin):
    """Depth-only side branch for a MiniMax-H3 ``transformer_ref``.

    Args:
        depth_channels: Width of one patchified H3-VAE depth row. The released
            24-channel latent and ``1x2x2`` patch produce 96 values.
        video_patch_dim: Width of one noisy target-video row (96 for the released
            24-channel H3 latent and ``(1,2,2)`` patches).
        hidden_size: H3 residual width (5376 in the released checkpoint).
        adapter_dim: Internal side-branch width.  This does not change H3 memory
            or checkpoint shapes.
        injection_layers: H3 block indices receiving residuals.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        depth_channels: int = 96,
        video_patch_dim: int = 96,
        hidden_size: int = 5376,
        adapter_dim: int = 1024,
        num_layers: int = 50,
        injection_layers: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48),
    ):
        super().__init__()
        if min(depth_channels, video_patch_dim, hidden_size, adapter_dim, num_layers) <= 0:
            raise ValueError("Depth adapter dimensions must be positive")
        self.injection_layers = _validate_layers(injection_layers, num_layers)
        self.depth_proj = nn.Linear(depth_channels, adapter_dim)
        self.video_proj = nn.Linear(video_patch_dim, adapter_dim)
        self.timestep_proj = nn.Linear(1, adapter_dim)
        self.position_proj = nn.Linear(6, adapter_dim, bias=False)
        self.side_blocks = nn.ModuleList([MiniMaxH3DepthSideBlock(adapter_dim) for _ in self.injection_layers])
        # Zero init is the scale=0/base-parity contract and lets the side branch
        # learn its influence without perturbing the released Ref2VA model.
        self.controlnet_blocks = nn.ModuleList(
            [zero_module(nn.Linear(adapter_dim, hidden_size)) for _ in self.injection_layers]
        )
        self.gradient_checkpointing = False

    @property
    def injection_layer_indices(self) -> tuple[int, ...]:
        return self.injection_layers

    def _validate_rows(
        self,
        depth_rows: torch.Tensor,
        noisy_video_rows: torch.Tensor | None,
        depth_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if depth_rows.ndim != 3:
            raise ValueError(f"depth_rows must be [B,N,Cd], got {tuple(depth_rows.shape)}")
        if depth_rows.shape[-1] != self.config.depth_channels:
            raise ValueError(
                f"depth_rows width must be {self.config.depth_channels}, got {depth_rows.shape[-1]}"
            )
        if not depth_rows.is_floating_point() or not bool(torch.isfinite(depth_rows).all()):
            raise ValueError("depth_rows must be a finite floating-point tensor")
        if noisy_video_rows is None:
            noisy_video_rows = depth_rows.new_zeros(
                depth_rows.shape[0], depth_rows.shape[1], self.config.video_patch_dim
            )
        if noisy_video_rows.ndim != 3 or noisy_video_rows.shape[:2] != depth_rows.shape[:2]:
            raise ValueError(
                "noisy_video_rows must have shape [B,N,video_patch_dim] with the same B,N as depth_rows, "
                f"got {tuple(noisy_video_rows.shape)}"
            )
        if noisy_video_rows.shape[-1] != self.config.video_patch_dim:
            raise ValueError(
                f"noisy_video_rows width must be {self.config.video_patch_dim}, got {noisy_video_rows.shape[-1]}"
            )
        if not noisy_video_rows.is_floating_point() or not bool(torch.isfinite(noisy_video_rows).all()):
            raise ValueError("noisy_video_rows must be a finite floating-point tensor")
        if depth_valid is not None:
            if depth_valid.ndim == 2:
                depth_valid = depth_valid.unsqueeze(-1)
            if depth_valid.shape != (*depth_rows.shape[:2], 1):
                raise ValueError(f"depth_valid must be [B,N,1], got {tuple(depth_valid.shape)}")
            if not bool(torch.isfinite(depth_valid).all()) or bool((depth_valid < 0).any()) or bool((depth_valid > 1).any()):
                raise ValueError("depth_valid values must be finite and in [0, 1]")
            depth_valid = depth_valid.to(dtype=depth_rows.dtype)
        return depth_rows, noisy_video_rows, depth_valid

    def forward(
        self,
        depth_rows: torch.Tensor,
        noisy_video_rows: torch.Tensor | None = None,
        timestep: torch.Tensor | float | None = None,
        depth_valid: torch.Tensor | None = None,
        target_position_ids: torch.Tensor | None = None,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> MiniMaxH3DepthControlOutput | tuple[torch.Tensor, ...]:
        """Produce target-row residuals in configured injection-layer order.

        ``depth_rows`` and ``noisy_video_rows`` contain target rows only. The MVP
        computes them globally on every context-parallel rank; the H3 transformer's
        block hook scatters and shards each residual exactly like the main sequence.
        """
        depth_rows, noisy_video_rows, depth_valid = self._validate_rows(depth_rows, noisy_video_rows, depth_valid)
        scale = float(conditioning_scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"conditioning_scale must be finite and non-negative, got {conditioning_scale}")
        if scale == 0.0:
            zeros = tuple(
                torch.zeros(
                    (*depth_rows.shape[:2], self.config.hidden_size),
                    device=depth_rows.device,
                    dtype=get_parameter_dtype(self),
                )
                for _ in self.injection_layers
            )
            output = MiniMaxH3DepthControlOutput(control_block_samples=zeros, injection_layers=self.injection_layers)
            return output if return_dict else zeros

        compute_dtype = get_parameter_dtype(self)
        depth_rows = depth_rows.to(dtype=compute_dtype)
        noisy_video_rows = noisy_video_rows.to(dtype=compute_dtype)
        hidden_states = self.depth_proj(depth_rows) + self.video_proj(noisy_video_rows)
        if target_position_ids is not None:
            if not target_position_ids.is_floating_point() or not bool(torch.isfinite(target_position_ids).all()):
                raise ValueError("target_position_ids must be a finite floating-point tensor")
            if target_position_ids.ndim == 2:
                target_position_ids = target_position_ids.unsqueeze(0)
            valid_batch = target_position_ids.shape[0] in (1, hidden_states.shape[0])
            if (
                target_position_ids.ndim != 3
                or not valid_batch
                or target_position_ids.shape[1] != hidden_states.shape[1]
                or target_position_ids.shape[-1] != 3
            ):
                raise ValueError(
                    "target_position_ids must be [N,3] or [B,N,3] and align with depth rows, got "
                    f"{tuple(target_position_ids.shape)}"
                )
            if target_position_ids.shape[0] == 1 and hidden_states.shape[0] != 1:
                target_position_ids = target_position_ids.expand(hidden_states.shape[0], -1, -1)
            # A bounded deterministic basis exposes H3's target MM-RoPE geometry
            # without adding depth tokens to the main sequence.
            position_basis = torch.cat(
                (target_position_ids.sin(), target_position_ids.cos()), dim=-1
            ).to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states + self.position_proj(position_basis)
        if timestep is None:
            timestep = hidden_states.new_zeros((hidden_states.shape[0], 1))
        else:
            timestep = torch.as_tensor(timestep, device=hidden_states.device, dtype=hidden_states.dtype)
            if timestep.ndim == 0:
                timestep = timestep.expand(hidden_states.shape[0]).unsqueeze(-1)
            elif timestep.ndim == 1:
                if timestep.numel() == 1:
                    timestep = timestep.expand(hidden_states.shape[0]).unsqueeze(-1)
                elif timestep.numel() == hidden_states.shape[0]:
                    timestep = timestep.unsqueeze(-1)
                else:
                    raise ValueError("timestep must be scalar or one value per batch item")
            if timestep.shape != (hidden_states.shape[0], 1):
                raise ValueError(f"timestep must resolve to [B,1], got {tuple(timestep.shape)}")
        hidden_states = hidden_states + self.timestep_proj(timestep).unsqueeze(1)
        valid_mask = None if depth_valid is None else depth_valid.to(device=hidden_states.device, dtype=hidden_states.dtype)

        residuals = []
        for side_block, output_block in zip(self.side_blocks, self.controlnet_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(side_block, hidden_states)
            else:
                hidden_states = side_block(hidden_states)
            residual = output_block(hidden_states)
            if valid_mask is not None:
                residual = residual * valid_mask
            residuals.append(residual * scale)
        residuals = tuple(residuals)
        output = MiniMaxH3DepthControlOutput(control_block_samples=residuals, injection_layers=self.injection_layers)
        return output if return_dict else residuals


__all__ = ["MiniMaxH3DepthControlNetModel", "MiniMaxH3DepthControlOutput"]
