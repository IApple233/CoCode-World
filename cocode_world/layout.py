"""Video-only packed layouts and patch transforms for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class H3VideoLayout:
    """Structural tensors shared by every item in a video-only H3 microbatch."""

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_latent_frames: int
    latent_height: int
    latent_width: int
    patch_size: tuple[int, int, int]
    video_channels: int
    audio_channels: int
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def num_video_rows(self) -> int:
        return int(self.video_indices.numel())

    @property
    def num_target_video_rows(self) -> int:
        return self.num_video_rows - self.num_condition_video_rows

    @property
    def condition_video_indices(self) -> torch.Tensor:
        return self.video_indices[: self.num_condition_video_rows]

    @property
    def target_video_indices(self) -> torch.Tensor:
        return self.video_indices[self.num_condition_video_rows :]

    @property
    def video_row_width(self) -> int:
        return self.video_channels * math.prod(self.patch_size)

    def validate(self, context_parallel_degree: int = 1) -> None:
        if context_parallel_degree < 1:
            raise ValueError("context_parallel_degree must be positive.")
        if self.position_ids.shape != (self.sequence_length, 3):
            raise ValueError("position_ids must have shape (sequence_length, 3).")
        if self.token_tags.shape != (self.sequence_length,):
            raise ValueError("token_tags must have shape (sequence_length,).")
        for name, value in (
            ("token_tags", self.token_tags),
            ("video_indices", self.video_indices),
            ("audio_indices", self.audio_indices),
            ("text_indices", self.text_indices),
        ):
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor.")
        if self.video_indices.numel() and not bool((self.video_indices[1:] > self.video_indices[:-1]).all()):
            raise ValueError("video_indices must be strictly increasing.")
        if self.text_indices.numel() and not bool((self.text_indices[1:] > self.text_indices[:-1]).all()):
            raise ValueError("text_indices must be strictly increasing.")
        indices = torch.cat((self.text_indices, self.video_indices, self.audio_indices)).sort().values
        expected = torch.arange(self.sequence_length, device=indices.device)
        if not torch.equal(indices, expected):
            raise ValueError("text/video/audio indices must partition the packed sequence exactly.")
        if self.audio_indices.numel() != 0:
            raise ValueError("The video-only layout must not contain audio rows.")
        if not 0 <= self.num_condition_video_rows < self.num_video_rows:
            raise ValueError("num_condition_video_rows must leave at least one generated video row.")
        if self.num_condition_audio_rows != 0:
            raise ValueError("The video-only layout cannot contain conditioning audio rows.")
        if self.sequence_length % context_parallel_degree:
            raise ValueError(
                f"Packed sequence length {self.sequence_length} is not divisible by CP={context_parallel_degree}. "
                "Choose a compatible text-token budget or latent geometry; unmasked padding would change H3 attention."
            )

    def to(self, device: torch.device | str) -> "H3VideoLayout":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return H3VideoLayout(**values)

    def structural_kwargs(
        self,
        h3_timestep: float | torch.Tensor,
        device: torch.device | str,
        *,
        condition_video_timestep: float | torch.Tensor | None = None,
        fixed_target_prefix_rows: int = 0,
        fixed_target_timestep: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        timestep = torch.as_tensor(h3_timestep, dtype=torch.float32, device=device).flatten()
        if timestep.numel() != 1 or not bool(torch.isfinite(timestep).all()) or not 0 <= float(timestep[0]) <= 1:
            raise ValueError("A video-only H3 microbatch must use one finite timestep in [0, 1].")
        row_timesteps = timestep.expand(self.sequence_length).clone()
        if self.num_condition_video_rows:
            if condition_video_timestep is None:
                raise ValueError("A Ref2VA layout requires condition_video_timestep.")
            condition_timestep = torch.as_tensor(
                condition_video_timestep, dtype=torch.float32, device=device
            ).flatten()
            if (
                condition_timestep.numel() != 1
                or not bool(torch.isfinite(condition_timestep).all())
                or not 0 <= float(condition_timestep[0]) <= 1
            ):
                raise ValueError("condition_video_timestep must be one finite value in [0, 1].")
            row_timesteps[self.condition_video_indices.to(device)] = condition_timestep[0]
        elif condition_video_timestep is not None:
            raise ValueError("condition_video_timestep was provided for a layout without reference rows.")
        if isinstance(fixed_target_prefix_rows, bool) or not isinstance(fixed_target_prefix_rows, int):
            raise ValueError("fixed_target_prefix_rows must be a non-negative integer")
        if not 0 <= fixed_target_prefix_rows <= self.num_target_video_rows:
            raise ValueError(
                f"fixed_target_prefix_rows must be in [0, {self.num_target_video_rows}], "
                f"got {fixed_target_prefix_rows}"
            )
        if fixed_target_prefix_rows:
            fixed_timestep = torch.as_tensor(
                fixed_target_timestep, dtype=torch.float32, device=device
            ).flatten()
            if (
                fixed_timestep.numel() != 1
                or not bool(torch.isfinite(fixed_timestep).all())
                or not 0 <= float(fixed_timestep[0]) <= 1
            ):
                raise ValueError("fixed_target_timestep must be one finite value in [0, 1]")
            fixed_indices = self.target_video_indices[:fixed_target_prefix_rows].to(device)
            row_timesteps[fixed_indices] = fixed_timestep[0]
        unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        return {
            "timestep": unique_timesteps,
            "timestep_indices": timestep_indices,
            "token_tags": self.token_tags.to(device),
            "position_ids": self.position_ids.to(device),
            "video_indices": self.video_indices.to(device),
            "audio_indices": self.audio_indices.to(device),
            "text_indices": self.text_indices.to(device),
        }


def build_video_only_layout(
    *,
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    video_channels: int = 24,
    audio_channels: int = 32,
    context_parallel_degree: int = 1,
) -> H3VideoLayout:
    """Build the native ``[text | target video]`` H3 layout with empty audio."""
    if min(num_text_tokens, num_latent_frames, latent_height, latent_width) <= 0:
        raise ValueError("Text length and latent geometry must be positive.")

    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    tensors = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(num_text_tokens, dtype=torch.long),
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=0,
        patch_size=patch_size,
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )
    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_video, num_condition_audio = tensors
    if num_condition_video or num_condition_audio:
        raise AssertionError("A text-to-video layout unexpectedly contains conditioning rows.")
    layout = H3VideoLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        patch_size=tuple(patch_size),
        video_channels=video_channels,
        audio_channels=audio_channels,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
    )
    layout.validate(context_parallel_degree)
    return layout


def patchify_video(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Convert ``[B,C,T,H,W]`` H3 latents to ``[B,V,C*pt*ph*pw]`` rows."""
    if latents.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W] latents, got {tuple(latents.shape)}.")
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}.")
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video(rows: torch.Tensor, layout: H3VideoLayout) -> torch.Tensor:
    """Invert target-video rows using the target geometry recorded by the layout."""
    if rows.ndim != 3 or rows.shape[1:] != (layout.num_target_video_rows, layout.video_row_width):
        raise ValueError(
            f"Expected target rows [B,{layout.num_target_video_rows},{layout.video_row_width}], "
            f"got {tuple(rows.shape)}."
        )
    patch_t, patch_h, patch_w = layout.patch_size
    rows = rows.reshape(
        rows.shape[0],
        layout.num_latent_frames // patch_t,
        layout.latent_height // patch_h,
        layout.latent_width // patch_w,
        layout.video_channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        rows.shape[0],
        layout.video_channels,
        layout.num_latent_frames,
        layout.latent_height,
        layout.latent_width,
    ).contiguous()
