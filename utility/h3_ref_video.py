"""Native MiniMax-H3 Ref2VA training with frozen image/video references."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.h3_distill.consistency import h3_interpolate
from utility.h3_distill.packed_video import H3VideoLayout, patchify_video


@dataclass
class H3RefVideoTrainingOutput:
    """One target-only rectified-flow result with frozen visual references."""

    loss: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    noisy_target_rows: torch.Tensor
    noised_reference_rows: torch.Tensor


def build_ref_video_layout(
    *,
    text_token_tags: torch.Tensor,
    num_image_reference_latent_frames: int | None = None,
    image_reference_latent_height: int | None = None,
    image_reference_latent_width: int | None = None,
    num_reference_latent_frames: int,
    reference_latent_height: int,
    reference_latent_width: int,
    num_target_latent_frames: int,
    target_latent_height: int,
    target_latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    video_channels: int = 24,
    audio_channels: int = 32,
    context_parallel_degree: int = 1,
    reference_latent_timestamps: list[float] | None = None,
    reference_duration_seconds: float | None = None,
) -> H3VideoLayout:
    """Build the official image + proxy-video Ref2VA packed layout.

    When image geometry is supplied, the packed order is ``[presentation |
    first-frame image | proxy video | target video]``.  The image and proxy are
    both frozen conditions; reference audio is disabled.
    """
    if text_token_tags.ndim != 1 or text_token_tags.dtype not in (torch.int32, torch.int64):
        raise ValueError("text_token_tags must be a one-dimensional integer tensor")
    if text_token_tags.numel() == 0 or not bool(((text_token_tags == 0) | (text_token_tags == 1)).all()):
        raise ValueError("Ref2VA text_token_tags must contain only picture/video tag 0 and text tag 1")
    if not bool((text_token_tags == 0).any()) or not bool((text_token_tags == 1).any()):
        raise ValueError("A Ref2VA presentation must contain both vision and text rows")
    dimensions = [
        num_reference_latent_frames,
        reference_latent_height,
        reference_latent_width,
        num_target_latent_frames,
        target_latent_height,
        target_latent_width,
        video_channels,
        audio_channels,
    ]
    image_dimensions = (
        num_image_reference_latent_frames,
        image_reference_latent_height,
        image_reference_latent_width,
    )
    has_image_reference = any(value is not None for value in image_dimensions)
    if has_image_reference and any(value is None for value in image_dimensions):
        raise ValueError("All image-reference latent dimensions must be supplied together")
    if has_image_reference:
        dimensions.extend(int(value) for value in image_dimensions if value is not None)
    if min(dimensions) <= 0:
        raise ValueError("Reference/target latent dimensions and channel counts must be positive")
    patch_t, patch_h, patch_w = patch_size
    for frames, height, width in (
        *((image_dimensions,) if has_image_reference else ()),
        (num_reference_latent_frames, reference_latent_height, reference_latent_width),
        (num_target_latent_frames, target_latent_height, target_latent_width),
    ):
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError("Reference and target latent geometry must be divisible by patch_size")

    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference

    # Only ``kind``/``has_audio`` are inspected by the layout builder. The real
    # pixels have already been encoded into ``reference_latent`` offline.
    references = []
    condition_latents = []
    if has_image_reference:
        references.append(MiniMaxH3ImageReference(image=torch.empty(3, 1, 1, device="meta")))
        condition_latents.append(
            torch.empty(
                1,
                video_channels,
                int(num_image_reference_latent_frames),
                int(image_reference_latent_height),
                int(image_reference_latent_width),
                device="meta",
            )
        )
    references.append(MiniMaxH3VideoReference(
        frames=torch.empty(1, 3, 1, 1, device="meta"),
        fps=24.0,
        audio=None,
    ))
    condition_latents.append(torch.empty(
        1,
        video_channels,
        num_reference_latent_frames,
        reference_latent_height,
        reference_latent_width,
        device="meta",
    ))
    values = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_token_tags=text_token_tags.cpu(),
        references=references,
        condition_latents=condition_latents,
        audio_condition_latents=[],
        num_latent_frames=num_target_latent_frames,
        latent_height=target_latent_height,
        latent_width=target_latent_width,
        num_audio_latents=0,
        patch_size=patch_size,
        audio_channels=audio_channels,
        audio_tag=2,
        video_tag=0,
    )
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = values
    expected_reference_rows = (
        (num_reference_latent_frames // patch_t)
        * (reference_latent_height // patch_h)
        * (reference_latent_width // patch_w)
    )
    if has_image_reference:
        expected_reference_rows += (
            (int(num_image_reference_latent_frames) // patch_t)
            * (int(image_reference_latent_height) // patch_h)
            * (int(image_reference_latent_width) // patch_w)
        )
    expected_target_rows = (
        (num_target_latent_frames // patch_t)
        * (target_latent_height // patch_h)
        * (target_latent_width // patch_w)
    )
    if num_condition_video_rows != expected_reference_rows or num_condition_audio_rows != 0:
        raise AssertionError("Official Ref2VA returned an unexpected visual-condition row count")
    if reference_latent_timestamps is not None or reference_duration_seconds is not None:
        if reference_latent_timestamps is None or patch_t != 1:
            raise ValueError("Sampled proxy source clocks require timestamps and temporal patch size 1")
        times = torch.as_tensor(reference_latent_timestamps, dtype=torch.float64)
        if (times.shape != (num_reference_latent_frames,) or not torch.isfinite(times).all()
                or times[0] != 0 or not bool((times[1:] >= times[:-1]).all())
                or reference_duration_seconds is None or not math.isfinite(reference_duration_seconds)
                or reference_duration_seconds <= float(times[-1])):
            raise ValueError("Invalid sampled proxy source clock")
        image_rows = expected_reference_rows - (
            num_reference_latent_frames * (reference_latent_height // patch_h) * (reference_latent_width // patch_w)
        )
        start = text_token_tags.numel() + image_rows
        stop = text_token_tags.numel() + expected_reference_rows
        origin = position_ids[start, 0].item()
        rows_per_frame = (reference_latent_height // patch_h) * (reference_latent_width // patch_w)
        # H3's native rotary time has 40 units per second. Sparse reference
        # latents retain their source times and the target's original start.
        position_ids[start:stop, 0] = (origin + times * 40.0).repeat_interleave(rows_per_frame)
        native_target_origin = position_ids[stop, 0].item()
        position_ids[stop:, 0] += origin + reference_duration_seconds * 40.0 - native_target_origin
    layout = H3VideoLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_latent_frames=num_target_latent_frames,
        latent_height=target_latent_height,
        latent_width=target_latent_width,
        patch_size=tuple(patch_size),
        video_channels=video_channels,
        audio_channels=audio_channels,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
    )
    layout.validate(context_parallel_degree)
    if layout.num_target_video_rows != expected_target_rows:
        raise AssertionError("Official Ref2VA returned an unexpected target-video row count")
    return layout


def patchify_ref_video_latents(
    target_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    layout: H3VideoLayout,
    reference_image_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify target and frozen references in official packed order."""
    target_rows = patchify_video(target_latent, layout.patch_size)
    reference_blocks = []
    if reference_image_latent is not None:
        reference_blocks.append(patchify_video(reference_image_latent, layout.patch_size))
    reference_blocks.append(patchify_video(reference_latent, layout.patch_size))
    reference_rows = torch.cat(reference_blocks, dim=1)
    if target_rows.shape[1:] != (layout.num_target_video_rows, layout.video_row_width):
        raise ValueError("Target latent does not match the Ref2VA layout")
    if reference_rows.shape[1:] != (layout.num_condition_video_rows, layout.video_row_width):
        raise ValueError("Proxy-video latent does not match the Ref2VA layout")
    return target_rows, reference_rows


class H3RefVideoTrainingModel(nn.Module):
    """Apply target-only flow matching with frozen Ref2VA visual conditions."""

    def __init__(self, transformer_ref: nn.Module, *, reference_noise_aug: float = 0.999) -> None:
        super().__init__()
        if not math.isfinite(reference_noise_aug) or not 0 <= reference_noise_aug <= 1:
            raise ValueError("reference_noise_aug must be finite and in [0,1]")
        self.transformer_ref = transformer_ref
        self.reference_noise_aug = float(reference_noise_aug)

    @staticmethod
    def _scalar_timestep(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
        timestep = torch.as_tensor(value, device=device, dtype=torch.float32).flatten()
        if timestep.numel() != 1 or not bool(torch.isfinite(timestep).all()):
            raise ValueError("Ref2VA training requires one finite timestep per microbatch")
        if not 0 <= float(timestep[0]) <= 1:
            raise ValueError("Ref2VA timestep must be in [0,1]")
        return timestep

    def predict_velocity(
        self,
        *,
        noisy_target_rows: torch.Tensor,
        noised_reference_rows: torch.Tensor,
        prompt_embeds: torch.Tensor,
        layout: H3VideoLayout,
        h3_timestep: float | torch.Tensor,
        fixed_target_prefix_rows: int = 0,
        fixed_target_timestep: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if noisy_target_rows.ndim != 3 or noised_reference_rows.ndim != 3:
            raise ValueError("Target and visual-condition rows must be [B,N,C]")
        batch_size = noisy_target_rows.shape[0]
        width = layout.video_row_width
        if noisy_target_rows.shape[1:] != (layout.num_target_video_rows, width):
            raise ValueError("noisy_target_rows do not match the target layout")
        if noised_reference_rows.shape != (batch_size, layout.num_condition_video_rows, width):
            raise ValueError("noised_reference_rows do not match the visual-condition layout")
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[:2] != (batch_size, layout.text_indices.numel()):
            raise ValueError("prompt_embeds must align one-to-one with layout text rows")
        for name, value in (
            ("noisy_target_rows", noisy_target_rows),
            ("noised_reference_rows", noised_reference_rows),
            ("prompt_embeds", prompt_embeds),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be a finite floating-point tensor")

        timestep = self._scalar_timestep(h3_timestep, noisy_target_rows.device)
        structural = layout.structural_kwargs(
            timestep,
            noisy_target_rows.device,
            condition_video_timestep=max(float(timestep[0]), self.reference_noise_aug),
            fixed_target_prefix_rows=fixed_target_prefix_rows,
            fixed_target_timestep=fixed_target_timestep,
        )
        video_rows = torch.cat((noised_reference_rows, noisy_target_rows), dim=1)
        empty_audio = video_rows.new_empty((batch_size, 0, layout.audio_channels))
        video_velocity, audio_velocity = self.transformer_ref(
            hidden_states=video_rows,
            audio_hidden_states=empty_audio,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            **structural,
        )
        if video_velocity.shape != video_rows.shape or audio_velocity.shape != empty_audio.shape:
            raise RuntimeError("MiniMax-H3 returned an unexpected video/audio velocity shape")
        return video_velocity[:, layout.num_condition_video_rows :]

    def flow_matching_loss(
        self,
        *,
        clean_target_rows: torch.Tensor,
        target_noise: torch.Tensor,
        clean_reference_rows: torch.Tensor,
        reference_noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        layout: H3VideoLayout,
        h3_timestep: float | torch.Tensor,
    ) -> H3RefVideoTrainingOutput:
        if clean_target_rows.shape != target_noise.shape:
            raise ValueError("clean_target_rows and target_noise must have identical shapes")
        if clean_reference_rows.shape != reference_noise.shape:
            raise ValueError("clean_reference_rows and reference_noise must have identical shapes")
        noisy_target = h3_interpolate(clean_target_rows, target_noise, h3_timestep)
        noised_reference = h3_interpolate(
            clean_reference_rows,
            reference_noise,
            self.reference_noise_aug,
        )
        prediction = self.predict_velocity(
            noisy_target_rows=noisy_target,
            noised_reference_rows=noised_reference,
            prompt_embeds=prompt_embeds,
            layout=layout,
            h3_timestep=h3_timestep,
        )
        target = clean_target_rows - target_noise
        loss = F.mse_loss(prediction.float(), target.detach().float(), reduction="mean")
        return H3RefVideoTrainingOutput(loss, prediction, target, noisy_target, noised_reference)


__all__ = [
    "H3RefVideoTrainingModel",
    "H3RefVideoTrainingOutput",
    "build_ref_video_layout",
    "patchify_ref_video_latents",
]
