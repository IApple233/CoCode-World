# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

"""Geometry and schedule helpers shared by H3 Ref2VA depth training/inference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniMaxH3DepthControlPlan:
    """Resolved side-branch inputs for one packed Ref2VA request."""

    depth_rows: torch.Tensor
    target_video_indices: torch.Tensor
    injection_layers: tuple[int, ...]
    scale: float


def patchify_depth(depth: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """Pack ``[B,C,T,H,W]`` depth grids in exactly H3's target-video row order."""
    if depth.ndim != 5:
        raise ValueError(f"depth must have shape [B,C,T,H,W], got {tuple(depth.shape)}")
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = depth.shape
    if min(patch_t, patch_h, patch_w) <= 0 or frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"depth shape {tuple(depth.shape)} is not divisible by patch_size={patch_size}")
    rows = depth.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()


def target_video_indices(video_indices: torch.Tensor, num_condition_video_rows: int = 0) -> torch.Tensor:
    """Return only target rows, preserving packed order and CP-local mapping."""
    if video_indices.ndim != 1:
        raise ValueError("video_indices must be one-dimensional")
    if not 0 <= num_condition_video_rows <= video_indices.numel():
        raise ValueError("num_condition_video_rows is outside video_indices")
    return video_indices[num_condition_video_rows:]


def depth_control_schedule(
    timestep: torch.Tensor | float,
    *,
    scale: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
    kind: str = "constant",
) -> torch.Tensor:
    """Resolve the documented H3 timestep gate.

    H3 uses ``t=0`` for the noisiest state and ``t=1`` for the clean state.  The
    gate is active on the inclusive ``[start,end]`` interval; outside it returns
    zero.  ``linear`` and ``cosine`` ramp from zero at ``start`` to ``scale`` at
    ``end`` in this explicitly documented timestep direction.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    if not 0 <= start <= end <= 1:
        raise ValueError("start/end must satisfy 0 <= start <= end <= 1")
    if kind not in {"constant", "linear", "cosine"}:
        raise ValueError(f"Unknown depth control schedule {kind!r}")
    value = torch.as_tensor(timestep, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("timestep must be finite and in [0,1]")
    active = (value >= start) & (value <= end)
    if end == start:
        progress = torch.ones_like(value)
    else:
        progress = ((value - start) / (end - start)).clamp(0, 1)
    if kind == "constant":
        weight = torch.ones_like(value)
    elif kind == "linear":
        weight = progress
    else:
        weight = 0.5 - 0.5 * torch.cos(progress * math.pi)
    return torch.where(active, weight * scale, torch.zeros_like(weight))


def h3_depth_frame_indices(
    num_frames: int,
    *,
    frames_per_chunk: int = 17,
    latents_per_chunk: int = 5,
) -> torch.Tensor:
    """Return pixel-frame anchors of H3's chunked causal VAE latent frames.

    A 17-frame H3 encoder chunk produces causal anchors at ``0,4,8,12,16``.
    The final partial chunk is last-frame padded and the encoder drops the final
    three latent tokens globally, giving 37 anchors for the 124-frame profile.
    """
    if num_frames < 1 or frames_per_chunk < 1 or latents_per_chunk < 1:
        raise ValueError("frame counts must be positive")
    if frames_per_chunk != 4 * (latents_per_chunk - 1) + 1:
        raise ValueError("This helper requires H3's causal 4x temporal compression geometry")
    num_chunks = math.ceil(num_frames / frames_per_chunk)
    indices = [
        min(chunk * frames_per_chunk + latent * 4, num_frames - 1)
        for chunk in range(num_chunks)
        for latent in range(latents_per_chunk)
    ]
    num_latent_frames = (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"num_frames must satisfy frames % {frames_per_chunk} == {latents_per_chunk}, got {num_frames}"
        )
    return torch.tensor(indices[:num_latent_frames], dtype=torch.long)


def prepare_h3_depth_rows(
    depth: torch.Tensor,
    depth_valid: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    min_valid_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample normalized pixel depth and patchify it onto H3's target grid.

    Depth values are validity-weighted spatial means. Temporal anchors follow
    H3's causal VAE exactly; validity is conservative at the transformer-patch
    level, so an invalid depth pixel cannot leak a nonzero control residual.
    """
    if depth.ndim != 5 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,T,H,W], got {tuple(depth.shape)}")
    if not depth.is_floating_point() or not bool(torch.isfinite(depth).all()):
        raise ValueError("depth must be a finite floating-point tensor")
    if bool((depth < 0).any()) or bool((depth > 1).any()):
        raise ValueError("depth values must be normalized to [0,1]")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    if min(latent_height, latent_width) <= 0:
        raise ValueError("latent_height and latent_width must be positive")

    if depth_valid.ndim == 1 and depth_valid.numel() in (1, depth.shape[0]):
        depth_valid = depth_valid.reshape(-1, 1, 1, 1, 1)
        if depth_valid.shape[0] == 1 and depth.shape[0] != 1:
            depth_valid = depth_valid.expand(depth.shape[0], -1, -1, -1, -1)
        depth_valid = depth_valid.expand_as(depth)
    if depth_valid.shape != depth.shape:
        raise ValueError(f"depth_valid must be scalar-per-item or match depth, got {tuple(depth_valid.shape)}")
    valid = depth_valid.to(device=depth.device, dtype=depth.dtype)
    if bool((valid < 0).any()) or bool((valid > 1).any()):
        raise ValueError("depth_valid values must be in [0,1]")

    frame_indices = h3_depth_frame_indices(depth.shape[2]).to(depth.device)
    depth = depth.index_select(2, frame_indices)
    valid = valid.index_select(2, frame_indices)
    output_size = (frame_indices.numel(), latent_height, latent_width)
    valid_weight = F.adaptive_avg_pool3d(valid, output_size)
    weighted_depth = F.adaptive_avg_pool3d(depth * valid, output_size)
    depth_grid = torch.where(valid_weight > 0, weighted_depth / valid_weight.clamp_min(1e-6), 0)
    valid_grid = valid_weight >= min_valid_fraction

    depth_rows = patchify_depth(depth_grid, patch_size)
    valid_rows = patchify_depth(valid_grid.to(depth_grid.dtype), patch_size).amin(dim=-1, keepdim=True).to(torch.bool)
    return depth_rows, valid_rows


def prepare_h3_depth_valid_rows(
    depth_valid: torch.Tensor,
    *,
    depth_shape: tuple[int, int, int, int, int],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    min_valid_fraction: float = 0.5,
) -> torch.Tensor:
    """Project only the canonical validity mask onto H3 target rows.

    This helper does not encode depth values. It remains independent from the
    visual-VAE depth representation and only prevents invalid source pixels from
    contributing control residuals.
    """
    if len(depth_shape) != 5 or depth_shape[1] != 1 or min(depth_shape) <= 0:
        raise ValueError(f"depth_shape must be [B,1,T,H,W], got {depth_shape}")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    if min(latent_height, latent_width) <= 0:
        raise ValueError("latent_height and latent_width must be positive")

    if depth_valid.ndim == 1 and depth_valid.numel() in (1, depth_shape[0]):
        depth_valid = depth_valid.reshape(-1, 1, 1, 1, 1)
        if depth_valid.shape[0] == 1 and depth_shape[0] != 1:
            depth_valid = depth_valid.expand(depth_shape[0], -1, -1, -1, -1)
        depth_valid = depth_valid.expand(depth_shape)
    if tuple(depth_valid.shape) != depth_shape:
        raise ValueError(f"depth_valid must be scalar-per-item or match depth, got {tuple(depth_valid.shape)}")
    valid = depth_valid.to(dtype=torch.float32)
    if not bool(torch.isfinite(valid).all()) or bool((valid < 0).any()) or bool((valid > 1).any()):
        raise ValueError("depth_valid values must be finite and in [0,1]")

    frame_indices = h3_depth_frame_indices(depth_shape[2]).to(depth_valid.device)
    valid = valid.index_select(2, frame_indices)
    valid_weight = F.adaptive_avg_pool3d(valid, (frame_indices.numel(), latent_height, latent_width))
    valid_grid = valid_weight >= min_valid_fraction
    return patchify_depth(valid_grid.to(torch.float32), patch_size).amin(dim=-1, keepdim=True).to(torch.bool)


def resolve_depth_control_plan(
    depth: torch.Tensor,
    video_indices: torch.Tensor,
    *,
    num_condition_video_rows: int = 0,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    injection_layers: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48),
    scale: float = 1.0,
) -> MiniMaxH3DepthControlPlan:
    """Build a checked target-only plan for a denoise forward."""
    rows = patchify_depth(depth, patch_size)
    target = target_video_indices(video_indices, num_condition_video_rows)
    if rows.shape[1] != target.numel():
        raise ValueError(f"Depth rows ({rows.shape[1]}) do not match target video rows ({target.numel()})")
    return MiniMaxH3DepthControlPlan(rows, target, tuple(injection_layers), float(scale))


__all__ = [
    "MiniMaxH3DepthControlPlan",
    "depth_control_schedule",
    "h3_depth_frame_indices",
    "patchify_depth",
    "prepare_h3_depth_valid_rows",
    "prepare_h3_depth_rows",
    "resolve_depth_control_plan",
    "target_video_indices",
]
