# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Explicit AR-role wrapper around the native MiniMax-H3 transformer.

The released ``MiniMaxH3Transformer3DModel`` intentionally keeps its default
full-attention behavior.  This class is a separate role with a strict dense
oracle limit and descriptor validation.  It must not silently turn a regular
H3 call into a causal call.  The explicit reference profiles remain
correctness paths; inference can select the native Ulysses SDPA executor and
its bounded exact cache.
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping

import torch

from ...configuration_utils import register_to_config
from .transformer_minimax_h3 import MiniMaxH3AttnProcessor, MiniMaxH3Transformer3DModel


CHUNK_AR_BACKEND_PROFILE = "dense_boolean_mask_t0_only"
CHUNK_AR_CP_REFERENCE_BACKEND_PROFILE = "packed_varlen_cp_reference_allgather_v1"
CHUNK_AR_CP_STREAMING_REFERENCE_BACKEND_PROFILE = "packed_varlen_cp_streaming_broadcast_reference_v1"
CHUNK_AR_CP_NATIVE_BACKEND_PROFILE = "packed_varlen_cp_ulysses_sdpa_native_v1"


def _validate_masked_text_padding(descriptor, text_indices: torch.Tensor, padding_rows: int) -> None:
    if padding_rows <= 0:
        return
    if text_indices.ndim != 1 or padding_rows > int(text_indices.numel()):
        raise ValueError("invalid text CP padding indices")
    padding = tuple(int(value) for value in text_indices[-padding_rows:].detach().cpu().tolist())
    padding_set = set(padding)
    if len(padding_set) != padding_rows:
        raise ValueError("text CP padding indices must be unique")
    for query_index in padding:
        if descriptor.key_ranges_for_query(query_index) != ((query_index, query_index + 1),):
            raise ValueError("each text CP padding query must attend only to itself")
    for segment in descriptor.segments:
        query_is_padding = all(
            query_index in padding_set
            for query_index in range(segment.query_start, segment.query_stop)
        )
        if query_is_padding:
            continue
        if any(
            key_index in padding_set
            for start, stop in segment.key_ranges
            for key_index in range(start, stop)
        ):
            raise ValueError("valid packed queries cannot attend to text CP padding keys")


class _VarlenAttentionProcessor(MiniMaxH3AttnProcessor):
    """Adapter that lets the role wrapper use a compact AR descriptor."""

    def __init__(
        self,
        descriptor,
        *,
        parallel_config=None,
        context_parallel_reference=False,
        context_parallel_streaming_reference=False,
        context_parallel_native=False,
        layer_index: int | None = None,
        layer_kv_cache=None,
        layer_kv_context: Mapping[str, Any] | None = None,
    ):
        self.descriptor = descriptor
        self._parallel_config = parallel_config
        self.context_parallel_reference = bool(context_parallel_reference)
        self.context_parallel_streaming_reference = bool(context_parallel_streaming_reference)
        self.context_parallel_native = bool(context_parallel_native)
        self.layer_index = layer_index
        self.layer_kv_cache = layer_kv_cache
        self.layer_kv_context = layer_kv_context
        if sum(
            (
                self.context_parallel_reference,
                self.context_parallel_streaming_reference,
                self.context_parallel_native,
            )
        ) > 1:
            raise ValueError("CP varlen backends are mutually exclusive")

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("varlen chunk-AR attention cannot receive a dense attention mask")
        if (
            self.layer_index == 0
            and self.layer_kv_cache is not None
            and self.layer_kv_context is not None
            and not attn.fused_projections
        ):
            query, key, value, minimum_token_rows = self._project_qkv_with_layer0_cache(
                attn,
                hidden_states,
                rotary_emb,
            )
        else:
            query, key, value, minimum_token_rows = self._project_qkv(attn, hidden_states, rotary_emb)
        cp_config = getattr(self._parallel_config, "context_parallel_config", None)
        cp_world_size = 1 if cp_config is None else int(cp_config._flattened_mesh.size())
        if cp_world_size > 1:
            if not (
                self.context_parallel_reference
                or self.context_parallel_streaming_reference
                or self.context_parallel_native
            ):
                raise ValueError("CP varlen attention requires an explicit reference or native backend")
            if not torch.distributed.is_initialized():
                raise RuntimeError("CP varlen attention requires an initialized process group")
            if self.descriptor.sequence_length % cp_world_size:
                raise ValueError("varlen descriptor sequence length is not divisible by CP degree")
            local_rows = self.descriptor.sequence_length // cp_world_size
            if query.shape[1] != local_rows:
                raise ValueError(
                    f"CP varlen query has {query.shape[1]} rows, expected {local_rows}"
                )
            group = cp_config._flattened_mesh.get_group()
            if self.context_parallel_native:
                if int(cp_config.ring_degree) != 1 or int(cp_config.ulysses_degree) != cp_world_size:
                    raise ValueError("native H3 AR attention requires a pure Ulysses CP mesh")
                from utility.h3_depth_ar.native_attention import attend_ulysses_sdpa

                output = attend_ulysses_sdpa(
                    query,
                    key,
                    value,
                    descriptor=self.descriptor,
                    group=cp_config._ulysses_mesh.get_group(),
                )
            elif self.context_parallel_streaming_reference:
                local_rank = int(cp_config._flattened_mesh.get_local_rank())
                query_start = local_rank * local_rows
                from utility.h3_depth_ar.streaming_attention import attend_query_range_streaming

                output = attend_query_range_streaming(
                    query,
                    key,
                    value,
                    descriptor=self.descriptor,
                    query_start=query_start,
                    query_stop=query_start + local_rows,
                    group=group,
                )
            else:
                local_rank = int(cp_config._flattened_mesh.get_local_rank())
                query_start = local_rank * local_rows
                from ...hooks.context_parallel import AllGatherFunction

                global_key = AllGatherFunction.apply(key, 1, group)
                global_value = AllGatherFunction.apply(value, 1, group)
                output = self.descriptor.attend_query_range(
                    query,
                    global_key,
                    global_value,
                    query_start=query_start,
                    query_stop=query_start + local_rows,
                )
        else:
            output = self.descriptor.attend(query, key, value)
        output = output.flatten(2, 3).type_as(query)
        from .transformer_minimax_h3 import _apply_tokenwise_module

        output = _apply_tokenwise_module(attn.to_out[0], output, minimum_token_rows)
        return attn.to_out[1](output)

    def _project_qkv(self, attn, hidden_states, rotary_emb):
        # Keep projection and RoPE arithmetic identical to the native H3
        # processor.  Importing the private helpers here avoids changing the
        # default transformer's public behavior.
        from .transformer_minimax_h3 import (
            _apply_rotary_emb,
            _apply_tokenwise_module,
            _short_cp_global_token_rows,
        )

        minimum_token_rows = _short_cp_global_token_rows(hidden_states.shape[1], self._parallel_config)
        if attn.fused_projections:
            query, key, value = _apply_tokenwise_module(
                attn.to_qkv, hidden_states, minimum_token_rows
            ).chunk(3, dim=-1)
        else:
            query = _apply_tokenwise_module(attn.to_q, hidden_states, minimum_token_rows)
            key = _apply_tokenwise_module(attn.to_k, hidden_states, minimum_token_rows)
            value = _apply_tokenwise_module(attn.to_v, hidden_states, minimum_token_rows)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = _apply_tokenwise_module(attn.norm_q, query, minimum_token_rows)
        key = _apply_tokenwise_module(attn.norm_k, key, minimum_token_rows)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        return query, key, value, minimum_token_rows

    def _project_qkv_with_layer0_cache(self, attn, hidden_states, rotary_emb):
        """Reuse only row-stable layer-0 K/V while always recomputing Q."""

        from .transformer_minimax_h3 import (
            _apply_rotary_emb,
            _apply_tokenwise_module,
            _short_cp_global_token_rows,
        )

        minimum_token_rows = _short_cp_global_token_rows(hidden_states.shape[1], self._parallel_config)
        query = _apply_tokenwise_module(attn.to_q, hidden_states, minimum_token_rows)
        query = query.unflatten(-1, (attn.heads, -1))
        query = _apply_tokenwise_module(attn.norm_q, query, minimum_token_rows)

        cp_config = getattr(self._parallel_config, "context_parallel_config", None)
        if cp_config is None:
            global_start = 0
        else:
            global_start = int(cp_config._flattened_mesh.get_local_rank()) * hidden_states.shape[1]
        global_stop = global_start + hidden_states.shape[1]

        cached_segments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        missed_segments: list[tuple[str, tuple[Hashable, ...], torch.Tensor]] = []
        hit_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)

        def local_segment(segment):
            start = max(global_start, int(segment["start"]))
            stop = min(global_stop, int(segment["stop"]))
            if start >= stop:
                return None
            offset = start - int(segment["start"])
            row_ids = tuple(segment["row_ids"][offset : offset + stop - start])
            indices = torch.arange(
                start - global_start,
                stop - global_start,
                device=hidden_states.device,
                dtype=torch.long,
            )
            return row_ids, indices

        reference = local_segment(self.layer_kv_context["reference"])
        if reference is not None:
            row_ids, indices = reference
            cached = self.layer_kv_cache.get_static(
                0,
                "reference",
                dependency=self.layer_kv_context["reference_dependency"],
                row_ids=row_ids,
            )
            if cached is None:
                missed_segments.append(("reference", row_ids, indices))
            else:
                hit_mask.index_fill_(0, indices, True)
                cached_segments.append((indices, *cached))

        for segment in self.layer_kv_context["history"]:
            local = local_segment(segment)
            if local is None:
                continue
            row_ids, indices = local
            cached = self.layer_kv_cache.get_history(
                0,
                dependency=self.layer_kv_context["history_dependency"],
                row_ids=row_ids,
            )
            if cached is None:
                missed_segments.append(("history", row_ids, indices))
            else:
                hit_mask.index_fill_(0, indices, True)
                cached_segments.append((indices, *cached))

        missing_indices = torch.nonzero(~hit_mask, as_tuple=False).flatten()
        missing_hidden = hidden_states.index_select(1, missing_indices)
        key_missing = _apply_tokenwise_module(attn.to_k, missing_hidden, minimum_token_rows)
        value_missing = _apply_tokenwise_module(attn.to_v, missing_hidden, minimum_token_rows)
        key_missing = key_missing.unflatten(-1, (attn.heads, -1))
        value_missing = value_missing.unflatten(-1, (attn.heads, -1))
        key_missing = _apply_tokenwise_module(attn.norm_k, key_missing, minimum_token_rows)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = _apply_rotary_emb(query, cos, sin)
            key_missing = _apply_rotary_emb(
                key_missing,
                cos.index_select(0, missing_indices),
                sin.index_select(0, missing_indices),
            )

        key = key_missing.new_empty(
            hidden_states.shape[0], hidden_states.shape[1], attn.heads, attn.head_dim
        )
        value = value_missing.new_empty(key.shape)
        key.index_copy_(1, missing_indices, key_missing)
        value.index_copy_(1, missing_indices, value_missing)
        for indices, cached_key, cached_value in cached_segments:
            key.index_copy_(1, indices, cached_key)
            value.index_copy_(1, indices, cached_value)

        pinned = self.layer_kv_context.get("pinned_history_row_ids", ())
        for role, row_ids, indices in missed_segments:
            segment_key = key.index_select(1, indices)
            segment_value = value.index_select(1, indices)
            if role == "reference":
                self.layer_kv_cache.put_static(
                    0,
                    "reference",
                    dependency=self.layer_kv_context["reference_dependency"],
                    row_ids=row_ids,
                    key=segment_key,
                    value=segment_value,
                )
            else:
                self.layer_kv_cache.append_history(
                    0,
                    dependency=self.layer_kv_context["history_dependency"],
                    row_ids=row_ids,
                    key=segment_key,
                    value=segment_value,
                    pinned_row_ids=pinned,
                )
        return query, key, value, minimum_token_rows


def validate_chunk_ar_descriptor(
    descriptor: Mapping[str, Any] | Any | None,
    *,
    sequence_length: int,
    maximum_dense_mask_rows: int,
) -> None:
    if maximum_dense_mask_rows <= 0:
        raise ValueError("maximum_dense_mask_rows must be positive")
    if sequence_length > maximum_dense_mask_rows:
        raise ValueError(
            f"chunk-AR dense oracle received {sequence_length} rows; "
            f"maximum is {maximum_dense_mask_rows}; use the varlen/KV-cache backend"
        )
    if descriptor is None:
        raise ValueError("chunk-AR forward requires an attention descriptor")
    if hasattr(descriptor, "backend_profile") and hasattr(descriptor, "validate"):
        descriptor.validate()
        if int(descriptor.sequence_length) != sequence_length:
            raise ValueError("chunk-AR varlen descriptor sequence length does not match the input")
        return
    if isinstance(descriptor, Mapping) and descriptor.get("backend_profile"):
        from utility.h3_depth_ar.varlen_attention import H3ARVarlenDescriptor

        H3ARVarlenDescriptor.from_dict(dict(descriptor)).validate()
        if int(descriptor.get("sequence_length", -1)) != sequence_length:
            raise ValueError("chunk-AR varlen descriptor sequence length does not match the input")
        return
    if descriptor.get("attention_backend_profile") != CHUNK_AR_BACKEND_PROFILE:
        raise ValueError("chunk-AR attention descriptor is not the bounded T0 profile")
    if int(descriptor.get("sequence_length", -1)) != sequence_length:
        raise ValueError("chunk-AR descriptor sequence length does not match the mask")


class MiniMaxH3ChunkARTransformer3DModel(MiniMaxH3Transformer3DModel):
    """Causal-role H3 transformer with explicit row/descriptor gates."""

    is_chunk_ar_role = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        ar_max_dense_mask_rows: int = 4096,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        if ar_max_dense_mask_rows <= 0:
            raise ValueError("ar_max_dense_mask_rows must be positive")
        self.ar_max_dense_mask_rows = int(ar_max_dense_mask_rows)

    def forward(
        self,
        *args,
        row_metadata: Any | None = None,
        attention_descriptor: Mapping[str, Any] | Any | None = None,
        varlen_descriptor: Any | None = None,
        varlen_context_parallel_reference: bool = False,
        varlen_context_parallel_streaming_reference: bool = False,
        varlen_context_parallel_native: bool = False,
        ar_inference_cache: Any | None = None,
        ar_kv_cache_context: Mapping[str, Any] | None = None,
        ar_max_dense_mask_rows: int | None = None,
        **kwargs,
    ):
        if varlen_descriptor is None:
            if hasattr(attention_descriptor, "attend"):
                varlen_descriptor = attention_descriptor
            elif isinstance(attention_descriptor, Mapping) and attention_descriptor.get("backend_profile"):
                from utility.h3_depth_ar.varlen_attention import H3ARVarlenDescriptor

                varlen_descriptor = H3ARVarlenDescriptor.from_dict(dict(attention_descriptor))
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        if varlen_descriptor is not None:
            if attention_mask is not None:
                raise ValueError("varlen chunk-AR forward must omit the dense attention_mask")
            if position_ids is None:
                raise ValueError("varlen chunk-AR forward requires position_ids")
            sequence_length = int(position_ids.shape[0])
            validate_chunk_ar_descriptor(
                varlen_descriptor,
                sequence_length=sequence_length,
                maximum_dense_mask_rows=sequence_length,
            )
            padding_rows = int(kwargs.get("text_padding_rows", 0))
            if padding_rows:
                text_indices = kwargs.get("text_indices")
                if not isinstance(text_indices, torch.Tensor):
                    raise ValueError("text_indices are required with text CP padding")
                _validate_masked_text_padding(varlen_descriptor, text_indices, padding_rows)
            cp_config = getattr(getattr(self, "_parallel_config", None), "context_parallel_config", None)
            cp_world_size = 1 if cp_config is None else int(cp_config._flattened_mesh.size())
            if cp_world_size > 1 and not (
                varlen_context_parallel_reference
                or varlen_context_parallel_streaming_reference
                or varlen_context_parallel_native
            ):
                raise ValueError(
                    "varlen chunk-AR with CP>1 requires an explicit reference or native backend"
                )
            if sum(
                (
                    bool(varlen_context_parallel_reference),
                    bool(varlen_context_parallel_streaming_reference),
                    bool(varlen_context_parallel_native),
                )
            ) > 1:
                raise ValueError("CP varlen backends are mutually exclusive")
            if varlen_context_parallel_native and torch.is_grad_enabled():
                raise RuntimeError("native H3 AR attention is inference-only")
            if ar_inference_cache is not None:
                if torch.is_grad_enabled():
                    raise RuntimeError("H3 AR inference cache cannot be used with gradients enabled")
                encoder_hidden_states = kwargs.get("encoder_hidden_states")
                position_ids = kwargs.get("position_ids")
                timestep = kwargs.get("timestep")
                if not all(isinstance(value, torch.Tensor) for value in (encoder_hidden_states, position_ids, timestep)):
                    raise ValueError("H3 AR inference cache requires prompt, positions, and timestep tensors")
                encoder_attention_mask = kwargs.get("encoder_attention_mask")
                kwargs["precomputed_text_embeds"] = ar_inference_cache.refined_prompt(
                    encoder_hidden_states,
                    encoder_attention_mask,
                    lambda: self.token_refiner(
                        self.context_embedder(
                            encoder_hidden_states.to(
                                next(self.context_embedder.parameters()).dtype
                            )
                        ),
                        encoder_attention_mask,
                    ),
                )
                layout_key = getattr(varlen_descriptor, "row_metadata_sha256", id(varlen_descriptor))
                kwargs["precomputed_rotary_emb"] = ar_inference_cache.rotary(
                    position_ids,
                    layout_key,
                    lambda: self.rope(position_ids),
                )
                timestep_key, temb = ar_inference_cache.timestep_embedding(
                    timestep,
                    lambda: self.time_embedder(
                        self.time_proj(timestep).to(next(self.time_embedder.parameters()).dtype)
                    ),
                )
                kwargs["precomputed_temb"] = temb
                kwargs["precomputed_block_adaln"] = ar_inference_cache.block_adaln(
                    timestep_key,
                    temb,
                    self.transformer_blocks,
                )
            if ar_kv_cache_context is not None and ar_inference_cache is None:
                raise ValueError("layer K/V context requires the H3 AR inference cache")
            processors = []
            for layer_index, block in enumerate(self.transformer_blocks):
                processor = block.attn.processor
                processors.append(processor)
                parallel_config = getattr(processor, "_parallel_config", None)
                if parallel_config is None:
                    parallel_config = getattr(self, "_parallel_config", None)
                block.attn.set_processor(
                    _VarlenAttentionProcessor(
                        varlen_descriptor,
                        parallel_config=parallel_config,
                        context_parallel_reference=varlen_context_parallel_reference,
                        context_parallel_streaming_reference=varlen_context_parallel_streaming_reference,
                        context_parallel_native=varlen_context_parallel_native,
                        layer_index=layer_index,
                        layer_kv_cache=None if ar_inference_cache is None else ar_inference_cache.layer_kv,
                        layer_kv_context=ar_kv_cache_context,
                    )
                )
            original_checkpointing_func = getattr(self, "_gradient_checkpointing_func", None)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if original_checkpointing_func is None:
                    raise RuntimeError("gradient checkpointing is enabled without a checkpoint function")

                def _checkpoint_with_bound_processor(module, *module_args):
                    processor = module.attn.processor

                    def _forward_with_bound_processor(*forward_args):
                        previous = module.attn.processor
                        module.attn.set_processor(processor)
                        try:
                            return module(*forward_args)
                        finally:
                            module.attn.set_processor(previous)

                    return original_checkpointing_func(_forward_with_bound_processor, *module_args)

                # The role wrapper restores the native processors when forward returns. Bind each checkpointed
                # block to the varlen processor it used so backward recomputation follows the identical graph.
                self._gradient_checkpointing_func = _checkpoint_with_bound_processor
            try:
                return super().forward(*args, **kwargs)
            finally:
                if original_checkpointing_func is not None:
                    self._gradient_checkpointing_func = original_checkpointing_func
                for block, processor in zip(self.transformer_blocks, processors):
                    block.attn.set_processor(processor)

        if attention_mask is None:
            raise ValueError("chunk-AR forward requires an explicit boolean attention_mask")
        if attention_mask.dtype != torch.bool or attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError("chunk-AR attention_mask must be a square boolean matrix")
        maximum = self.ar_max_dense_mask_rows if ar_max_dense_mask_rows is None else int(ar_max_dense_mask_rows)
        validate_chunk_ar_descriptor(
            attention_descriptor,
            sequence_length=int(attention_mask.shape[0]),
            maximum_dense_mask_rows=maximum,
        )
        if row_metadata is not None:
            try:
                metadata_length = len(row_metadata)
            except TypeError as error:
                raise ValueError("row_metadata must be a sized sequence") from error
            if metadata_length != attention_mask.shape[0]:
                raise ValueError("row_metadata length does not match the attention mask")
        return super().forward(*args, **kwargs)


__all__ = [
    "CHUNK_AR_BACKEND_PROFILE",
    "CHUNK_AR_CP_REFERENCE_BACKEND_PROFILE",
    "CHUNK_AR_CP_STREAMING_REFERENCE_BACKEND_PROFILE",
    "CHUNK_AR_CP_NATIVE_BACKEND_PROFILE",
    "MiniMaxH3ChunkARTransformer3DModel",
    "validate_chunk_ar_descriptor",
]
