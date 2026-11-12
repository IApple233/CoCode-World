"""MiniMax-H3 text and reference conditioning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from cocode_world.io import stable_mod
from cocode_world.prompting import SYSTEM_PROMPT, wrap_reference_presentation

TARGET_FPS = 24.0


TEXT_MAX_PRESENTATION_ROWS = 16_384


CP_SUFFIX_UNIT = " ."


TEXT_ENCODER_LAYER = 50


TEXT_ENCODER_EXECUTION_REVISION = "qwen3vl-hidden50-hook-layer49-inference-retain-0-through-49-v2"


PIXEL_MEAN = (0.485, 0.456, 0.406)


PIXEL_STD = (0.229, 0.224, 0.225)


def _fingerprint(model_root: Path, component: str, index_name: str) -> str:
    digest = hashlib.sha256()
    component_root = model_root / component
    for name in ("config.json", index_name):
        path = component_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _truncate_qwen_for_h3_hidden50(text_encoder) -> None:
    language_model = text_encoder.model.language_model
    if len(language_model.layers) <= TEXT_ENCODER_LAYER:
        raise ValueError("Qwen does not expose hidden_states[50] required by MiniMax-H3")
    language_model.layers = torch.nn.ModuleList(language_model.layers[:TEXT_ENCODER_LAYER])
    text_encoder.lm_head = None


def _load_text_components(model_root: Path, device: torch.device):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    pipeline = MiniMaxH3Ref2VATextEncoderStep().init_pipeline(str(model_root))
    pipeline.load_components(
        pretrained_model_name_or_path=str(model_root),
        dtype={"text_encoder": torch.bfloat16},
        device_map={"text_encoder": str(device)},
    )
    missing = [name for name in ("tokenizer", "processor", "text_encoder") if getattr(pipeline, name, None) is None]
    if missing:
        raise RuntimeError(f"MiniMax-H3 text cache components failed to load: {missing}")
    _truncate_qwen_for_h3_hidden50(pipeline.text_encoder)
    pipeline.text_encoder.eval().requires_grad_(False)
    torch.cuda.empty_cache()
    return pipeline


def _get_qwen3vl_hidden50_memory_efficient(
    text_encoder,
    processor,
    token_ids: list[int],
    vision_inputs: dict | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    language_model = text_encoder.model.language_model
    if len(language_model.layers) != TEXT_ENCODER_LAYER:
        raise ValueError(f"Expected {TEXT_ENCODER_LAYER} retained Qwen layers")
    captured: list[torch.Tensor] = []

    def capture(_module, _args, output) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    with torch.inference_mode():
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        token_type_factory = getattr(processor, "create_mm_token_type_ids", None)
        if callable(token_type_factory):
            token_types = token_type_factory([token_ids])
        else:
            from diffusers.modular_pipelines.minimax_h3.encoders import create_qwen3vl_mm_token_type_ids

            token_types = create_qwen3vl_mm_token_type_ids(processor, [token_ids])
        mm_token_type_ids = torch.tensor(token_types, dtype=torch.long, device=device)
        vision_kwargs = {
            name: value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
            for name, value in (vision_inputs or {}).items()
        }
        hook = getattr(text_encoder, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(text_encoder)
        handle = language_model.layers[-1].register_forward_hook(capture)
        try:
            text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=False,
                **vision_kwargs,
            )
        finally:
            handle.remove()
    if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
        raise RuntimeError(f"Expected one Qwen hidden-state capture, got {len(captured)}")
    return captured[0].to(device=device, dtype=dtype)


def _choose_prompt(
    tokenizer,
    text_step,
    processor,
    normalized_references,
    prompt: str,
    cp_degree: int,
    system_prompt: str = SYSTEM_PROMPT,
    *,
    target_rows: int,
    condition_rows: int,
) -> tuple[str, str, list[int], list[int], dict[str, torch.Tensor], int]:
    vision_inputs, image_counts, video_counts, timestamps = text_step._gather_vision_features(
        processor,
        normalized_references,
        TARGET_FPS,
    )
    for additions in range(cp_degree):
        suffix = CP_SUFFIX_UNIT * additions
        candidate = prompt + suffix
        token_ids, token_tags = text_step._build_presentation(
            tokenizer,
            candidate,
            normalized_references,
            image_counts,
            video_counts,
            timestamps,
        )
        token_ids, token_tags = wrap_reference_presentation(tokenizer, processor, token_ids, token_tags, system_prompt=system_prompt)
        packed_rows = len(token_ids) + condition_rows + target_rows
        if len(token_ids) <= TEXT_MAX_PRESENTATION_ROWS and packed_rows % cp_degree == 0:
            return candidate, suffix, token_ids, token_tags, vision_inputs, packed_rows
    raise ValueError(f"Could not make image+video Ref2VA presentation divisible by CP={cp_degree}")


def _target_seed(base_seed: int, sample_id: str) -> int:
    return stable_mod(f"{base_seed}:{sample_id}", 2**31 - 1, "h3-ref-video-target-vae-v2")


def _conditioning_prompt(row: dict[str, Any]) -> str:
    task_prompt = str(row["prompt"]).strip()
    scene_description = str(row["scene_description"]).strip()
    if not task_prompt:
        raise ValueError(f"{row.get('sample_id', '<unknown>')} prompt must be non-empty")
    if not scene_description:
        return task_prompt
    return f"{task_prompt}\nScene description: {scene_description}"


def _system_prompt(row: dict[str, Any]) -> str:
    value = row.get("system_prompt", SYSTEM_PROMPT)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("system_prompt must be non-empty")
    digest = hashlib.sha256(value.encode()).hexdigest()
    if "system_prompt" in row and row.get("system_prompt_sha256") != digest:
        raise ValueError("Explicit system_prompt requires a matching SHA256")
    return value
