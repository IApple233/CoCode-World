"""Causal attention primitives for MiniMax-H3's packed audio/video sequence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor


@dataclass(frozen=True)
class H3ChunkCausalMask:
    """Dense reference mask and row metadata for one H3 packed sequence.

    The dense matrix is intentionally a correctness oracle for T0-sized tests.
    A production 38K-row request must translate the same ``row_chunk_ids``
    contract to a block-sparse or varlen attention kernel.
    """

    attention_mask: torch.Tensor
    row_chunk_ids: torch.Tensor
    context_rows: torch.Tensor
    target_rows: torch.Tensor
    video_frame_times: torch.Tensor
    chunk_start_times: torch.Tensor


def _validate_indices(name: str, indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    indices = indices.detach().to(device="cpu", dtype=torch.long).flatten()
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= sequence_length):
        raise ValueError(f"{name} contains a row outside [0, {sequence_length}).")
    if indices.unique().numel() != indices.numel():
        raise ValueError(f"{name} contains duplicate rows.")
    return indices


def build_h3_chunk_causal_mask(
    *,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    num_condition_video_rows: int = 0,
    num_condition_audio_rows: int = 0,
    video_chunk_size: int = 5,
) -> H3ChunkCausalMask:
    """Build the H3 dense reference mask using physical rotary time.

    Text and explicit reference/keyframe rows are immutable, globally visible
    context. Context queries can only read context, which prevents future target
    rows from leaking through an updated text/reference representation in deeper
    layers. Target audio and video queries read all context and target rows from
    their current or earlier time chunk. Audio chunks are assigned by the next
    video chunk's physical start time, not by packed row order or a fixed ratio.

    A globally visible ``last`` keyframe is valid endpoint conditioning, but it
    is not an online-causal claim. The caller must record that policy separately.
    """

    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"position_ids must have shape (S, 3), got {tuple(position_ids.shape)}.")
    if video_chunk_size <= 0:
        raise ValueError(f"video_chunk_size must be positive, got {video_chunk_size}.")

    sequence_length = int(position_ids.shape[0])
    video_indices = _validate_indices("video_indices", video_indices, sequence_length)
    audio_indices = _validate_indices("audio_indices", audio_indices, sequence_length)
    text_indices = _validate_indices("text_indices", text_indices, sequence_length)
    if not 0 <= num_condition_video_rows <= video_indices.numel():
        raise ValueError("num_condition_video_rows is outside video_indices.")
    if not 0 <= num_condition_audio_rows <= audio_indices.numel():
        raise ValueError("num_condition_audio_rows is outside audio_indices.")

    all_indices = torch.cat((video_indices, audio_indices, text_indices))
    if all_indices.numel() != sequence_length or all_indices.unique().numel() != sequence_length:
        raise ValueError("video/audio/text indices must be disjoint and cover every packed row exactly once.")

    condition_video = video_indices[:num_condition_video_rows]
    condition_audio = audio_indices[:num_condition_audio_rows]
    target_video = video_indices[num_condition_video_rows:]
    target_audio = audio_indices[num_condition_audio_rows:]
    if target_video.numel() == 0:
        raise ValueError("At least one target video row is required to define the shared AV clock.")

    positions = position_ids.detach().to(device="cpu", dtype=torch.float64)
    video_frame_times = torch.unique(positions[target_video, 0], sorted=True)
    if video_frame_times.numel() == 0:
        raise ValueError("No target video frame times were found.")
    chunk_start_times = video_frame_times[::video_chunk_size]

    row_chunk_ids = torch.full((sequence_length,), -1, dtype=torch.long)
    video_time_to_chunk = {
        float(time): frame_index // video_chunk_size for frame_index, time in enumerate(video_frame_times.tolist())
    }
    for row in target_video.tolist():
        row_chunk_ids[row] = video_time_to_chunk[float(positions[row, 0])]

    if target_audio.numel():
        boundaries = chunk_start_times[1:]
        audio_times = positions[target_audio, 0]
        # right=True assigns an audio row exactly on a video chunk boundary to
        # the new chunk. Rows after the final start remain in the final chunk.
        audio_chunks = torch.bucketize(audio_times, boundaries, right=True)
        audio_chunks.clamp_(max=chunk_start_times.numel() - 1)
        row_chunk_ids[target_audio] = audio_chunks

    context_rows = torch.cat((text_indices, condition_video, condition_audio)).unique(sorted=True)
    target_rows = torch.cat((target_video, target_audio)).unique(sorted=True)
    if (row_chunk_ids[target_rows] < 0).any():
        raise RuntimeError("Every target row must receive a causal chunk id.")

    attention_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    attention_mask[context_rows[:, None], context_rows[None, :]] = True
    attention_mask[target_rows[:, None], context_rows[None, :]] = True
    target_chunks = row_chunk_ids[target_rows]
    target_causal = target_chunks[None, :] <= target_chunks[:, None]
    attention_mask[target_rows[:, None], target_rows[None, :]] = target_causal
    if not bool(attention_mask.diagonal().all()):
        raise RuntimeError("The causal mask must preserve self-attention for every row.")

    return H3ChunkCausalMask(
        attention_mask=attention_mask.to(position_ids.device),
        row_chunk_ids=row_chunk_ids.to(position_ids.device),
        context_rows=context_rows.to(position_ids.device),
        target_rows=target_rows.to(position_ids.device),
        video_frame_times=video_frame_times.to(position_ids.device),
        chunk_start_times=chunk_start_times.to(position_ids.device),
    )


class MiniMaxH3MaskedAttnProcessor(MiniMaxH3AttnProcessor):
    """Inject a precomputed mask through H3's existing attention processor API."""

    def __init__(self, attention_mask: torch.Tensor):
        if attention_mask.ndim not in (2, 3, 4) or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a 2D/3D/4D bool tensor.")
        self.attention_mask = attention_mask

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("MiniMaxH3MaskedAttnProcessor received two attention masks.")
        mask = self.attention_mask.to(device=hidden_states.device)
        return super().__call__(attn, hidden_states, rotary_emb=rotary_emb, attention_mask=mask)


@contextmanager
def use_h3_main_attention_mask(model, attention_mask: torch.Tensor) -> Iterator[None]:
    """Temporarily mask only H3's main block stack, leaving the text refiner full-attention."""

    if not hasattr(model, "transformer_blocks"):
        raise TypeError("Expected a MiniMax-H3 model with transformer_blocks.")
    original_processors = [block.attn.processor for block in model.transformer_blocks]
    try:
        for block in model.transformer_blocks:
            block.attn.set_processor(MiniMaxH3MaskedAttnProcessor(attention_mask))
        yield
    finally:
        for block, processor in zip(model.transformer_blocks, original_processors):
            block.attn.set_processor(processor)
