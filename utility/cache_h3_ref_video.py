#!/usr/bin/env python3
"""Cache 480x1664 GTA split-screen image+video Ref2VA training samples.

Each raw manifest row points at the agent-000 and agent-001 source clips.  This
tool decodes and concatenates them in memory, extracts the target's first frame
as a frozen image reference, and caches all VAE/Qwen tensors needed by training.
No intermediate split-screen MP4 is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import errno
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "verification", ROOT / "diffusers" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from local_diffusers import use_local_diffusers


use_local_diffusers()

from data.worldscene_h3_ref_depth_v1.tools.common import (  # noqa: E402
    atomic_save_safetensors,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_mod,
)
from utility.h3_ref_video_data import (  # noqa: E402
    CachedH3RefVideoDataset,
    IMAGE_REFERENCE_LATENT_SHAPE,
    REF_VIDEO_CACHE_SCHEMA_VERSION,
    TARGET_LATENT_SHAPE,
    TARGET_VIDEO_ROWS,
)
from utility.h3_ref_video_prompt import (  # noqa: E402
    PRESENTATION_REVISION,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    wrap_reference_presentation,
)
from utility.h3_ref_video_reference import (  # noqa: E402
    EXTERNAL_REFERENCE, TARGET_REFERENCE, load_external_reference,
)


DEFAULT_MODEL_ROOT = Path("models/MiniMax-H3")
RAW_SCHEMA_VERSION = "gta-minimax-h3-split-screen-v1"
TEXT_CACHE_SCHEMA_VERSION = "h3-ref-image-video-splitscreen-text-v3"
TARGET_FRAMES = 124
TARGET_FPS = 24.0
SINGLE_HEIGHT = 480
SINGLE_WIDTH = 832
TARGET_HEIGHT = 480
TARGET_WIDTH = 1664
IMAGE_REFERENCE_ROWS = 780
PROXY_VIDEO_ROWS = 28_860
TOTAL_CONDITION_ROWS = IMAGE_REFERENCE_ROWS + PROXY_VIDEO_ROWS
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


def _source_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "target_video_sha256": row["target_video_sha256"],
        "proxy_video_sha256": row["proxy_video_sha256"],
        "split_screen": row["split_screen"],
        "reference_image_source": row["reference_image_source"],
    }
    if row["reference_image_source"] == EXTERNAL_REFERENCE:
        payload["reference_image_sha256"] = row["reference_image_sha256"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    required = {
        "schema_version",
        "sample_id",
        "prompt",
        "scene_description",
        "target_video_paths",
        "target_video_sha256",
        "proxy_video_paths",
        "proxy_video_sha256",
        "reference_image_source",
        "split_screen",
        "num_frames",
        "fps",
        "height",
        "width",
    }
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = required - set(record)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
        if record["schema_version"] != RAW_SCHEMA_VERSION:
            raise ValueError(f"{path}:{line_number} has unsupported schema {record['schema_version']!r}")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"{path}:{line_number} prompt must be a non-empty string")
        if not isinstance(record["scene_description"], str):
            raise ValueError(f"{path}:{line_number} scene_description must be a string")
        if record["reference_image_source"] == EXTERNAL_REFERENCE:
            value = Path(record["reference_image_path"])
            record["reference_image_path"] = str((path.parent / value).resolve())
            load_external_reference(record)
        elif record["reference_image_source"] != TARGET_REFERENCE:
            raise ValueError(f"{path}:{line_number} has the wrong reference image source")
        if "system_prompt" in record:
            _system_prompt(record)
        if (record["num_frames"], float(record["fps"]), record["height"], record["width"]) != (
            TARGET_FRAMES,
            TARGET_FPS,
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ):
            raise ValueError(f"{path}:{line_number} is not a 124/24/480x1664 sample")
        for paths_field, hashes_field in (
            ("target_video_paths", "target_video_sha256"),
            ("proxy_video_paths", "proxy_video_sha256"),
        ):
            paths, hashes = record[paths_field], record[hashes_field]
            if not isinstance(paths, dict) or set(paths) != {"left", "right"}:
                raise ValueError(f"{path}:{line_number} {paths_field} must have left/right")
            if not isinstance(hashes, dict) or set(hashes) != {"left", "right"}:
                raise ValueError(f"{path}:{line_number} {hashes_field} must have left/right")
            resolved_paths = {}
            for side in ("left", "right"):
                value = Path(str(paths[side]))
                resolved = value.resolve() if value.is_absolute() else (path.parent / value).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                resolved_paths[side] = str(resolved)
            record[paths_field] = resolved_paths
        record["source_media_fingerprint"] = _source_fingerprint(record)
        records.append(record)
    if not records:
        raise ValueError("Input manifest is empty")
    counts = Counter(str(record["sample_id"]) for record in records)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"Duplicate sample_id values: {duplicates[:20]}")
    return records


def _assigned(records: list[dict[str, Any]], rank: int, world_size: int, limit: int) -> list[dict[str, Any]]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank/world-size: {rank}/{world_size}")
    values = [record for index, record in enumerate(records) if index % world_size == rank]
    return values[:limit] if limit > 0 else values


def _safe_name(sample_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in sample_id)


def _text_paths(output_dir: Path, sample_id: str) -> tuple[Path, Path]:
    name = _safe_name(sample_id)
    return output_dir / "text" / f"{name}.safetensors", output_dir / "text_done" / f"{name}.json"


def _sample_paths(output_dir: Path, sample_id: str) -> tuple[Path, Path]:
    name = _safe_name(sample_id)
    return output_dir / "samples" / f"{name}.safetensors", output_dir / "sample_done" / f"{name}.json"


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _valid_done(done_path: Path, tensor_path: Path, hash_field: str, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not done_path.is_file() or not tensor_path.is_file():
        return None
    try:
        record = json.loads(done_path.read_text(encoding="utf-8"))
        if any(record.get(key) != value for key, value in expected.items()):
            return None
        if record.get(hash_field) != sha256_file(tensor_path):
            return None
        return record
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _configure_setup_pipeline():
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

    pipeline = MiniMaxH3Ref2VASetupStep().init_pipeline()
    pipeline.update_components(
        canvas_short_edge=TARGET_HEIGHT,
        canvas_max_pixels=TARGET_HEIGHT * TARGET_WIDTH,
        reference_image_short_edge=TARGET_HEIGHT,
    )
    return pipeline


def _decode_video(path: Path, *, first_frame_only: bool = False) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "video")
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
            if first_frame_only:
                break
    if not frames:
        raise ValueError(f"Video decoded no frames: {path}")
    array = np.stack(frames).astype(np.uint8, copy=False)
    expected_frames = 1 if first_frame_only else TARGET_FRAMES
    if array.shape != (expected_frames, SINGLE_HEIGHT, SINGLE_WIDTH, 3):
        raise ValueError(f"{path} decoded as {array.shape}, expected [{expected_frames},480,832,3]")
    if abs(fps - TARGET_FPS) > 1e-6:
        raise ValueError(f"{path} must be 24 fps, got {fps}")
    return array, fps


def _decode_split(paths: dict[str, str], *, first_frame_only: bool = False) -> np.ndarray:
    left, left_fps = _decode_video(Path(paths["left"]), first_frame_only=first_frame_only)
    right, right_fps = _decode_video(Path(paths["right"]), first_frame_only=first_frame_only)
    if left_fps != right_fps or left.shape != right.shape:
        raise ValueError("Left/right videos have mismatched frame grids")
    combined = np.concatenate((left, right), axis=2)
    expected_frames = 1 if first_frame_only else TARGET_FRAMES
    if combined.shape != (expected_frames, TARGET_HEIGHT, TARGET_WIDTH, 3):
        raise AssertionError(f"Unexpected split-screen shape {combined.shape}")
    return np.ascontiguousarray(combined)


def _verify_source_hashes(row: dict[str, Any]) -> None:
    if row["reference_image_source"] == EXTERNAL_REFERENCE:
        load_external_reference(row)
    for paths_field, hashes_field in (
        ("target_video_paths", "target_video_sha256"),
        ("proxy_video_paths", "proxy_video_sha256"),
    ):
        for side in ("left", "right"):
            actual = sha256_file(Path(row[paths_field][side]))
            expected = str(row[hashes_field][side])
            if actual != expected:
                raise ValueError(f"{row['sample_id']} {paths_field}.{side} SHA mismatch: {actual} != {expected}")


def _normalized_references(setup_pipe, row: dict[str, Any]):
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference

    first_frame = (load_external_reference(row) if row["reference_image_source"] == EXTERNAL_REFERENCE
                   else _decode_split(row["target_video_paths"], first_frame_only=True)[0])
    proxy_frames = _decode_split(row["proxy_video_paths"])
    references = [
        MiniMaxH3ImageReference(image=first_frame),
        MiniMaxH3VideoReference(frames=proxy_frames, fps=TARGET_FPS, audio=None),
    ]
    normalized = setup_pipe(
        references=references,
        height=TARGET_HEIGHT,
        width=TARGET_WIDTH,
        num_frames=TARGET_FRAMES,
        output="normalized_references",
    )
    if len(normalized) != 2 or [item.kind for item in normalized] != ["image", "video"]:
        raise ValueError("Ref2VA setup did not preserve [image, video] reference order")
    if normalized[1].has_audio:
        raise ValueError("Proxy video unexpectedly contains reference audio")
    image = np.asarray(normalized[0].image)
    video = np.asarray(normalized[1].frames)
    if image.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3) or image.dtype != np.uint8:
        raise ValueError(f"Normalized reference image has invalid geometry/dtype: {image.shape} {image.dtype}")
    if video.shape != (TARGET_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, 3) or video.dtype != np.uint8:
        raise ValueError(f"Normalized proxy video has invalid geometry/dtype: {video.shape} {video.dtype}")
    if not np.array_equal(image, first_frame):
        raise ValueError("Normalized first-frame reference changed despite exact 480x1664 setup")
    return normalized


def _choose_prompt(
    tokenizer,
    text_step,
    processor,
    normalized_references,
    prompt: str,
    cp_degree: int,
    system_prompt: str = SYSTEM_PROMPT,
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
        packed_rows = len(token_ids) + TOTAL_CONDITION_ROWS + TARGET_VIDEO_ROWS
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


def _text_expected(row: dict[str, Any], args: argparse.Namespace, text_hash: str, transformer_hash: str) -> dict[str, Any]:
    return {
        "schema_version": TEXT_CACHE_SCHEMA_VERSION,
        "text_encoder_hash": text_hash,
        "text_encoder_layer": TEXT_ENCODER_LAYER,
        "text_encoder_execution_revision": TEXT_ENCODER_EXECUTION_REVISION,
        "transformer_ref_config_hash": transformer_hash,
        "cache_context_parallel_degree": args.context_parallel_degree,
        "reference_kinds": ["image", "video"],
        "reference_has_audio": False,
        "reference_image_frozen": True,
        "source_media_fingerprint": row["source_media_fingerprint"],
        "original_prompt": _conditioning_prompt(row),
        "scene_description": row["scene_description"],
        "text_presentation_revision": PRESENTATION_REVISION,
        "system_prompt": _system_prompt(row),
        "system_prompt_sha256": hashlib.sha256(_system_prompt(row).encode()).hexdigest(),
    }


def run_text(args: argparse.Namespace) -> int:
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_manifest(args.manifest)
    assigned = _assigned(records, args.rank, args.world_size, args.limit)
    device = torch.device(args.device or f"cuda:{args.rank}")
    if device.type != "cuda":
        raise ValueError("Qwen cache construction requires CUDA")
    torch.cuda.set_device(device)
    model_root = args.model_root.resolve()
    text_hash = _fingerprint(model_root, "text_encoder", "model.safetensors.index.json")
    transformer_hash = _fingerprint(model_root, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json")
    setup_pipe = _configure_setup_pipeline()
    text_step = MiniMaxH3Ref2VATextEncoderStep()
    pipeline = _load_text_components(model_root, device)
    successes = failures = 0
    for index, row in enumerate(assigned, start=1):
        sample_id = str(row["sample_id"])
        tensor_path, done_path = _text_paths(output_dir, sample_id)
        expected = _text_expected(row, args, text_hash, transformer_hash)
        if _valid_done(done_path, tensor_path, "text_sha256", expected) is not None:
            successes += 1
            continue
        try:
            _verify_source_hashes(row)
            normalized = _normalized_references(setup_pipe, row)
            conditioning_prompt = _conditioning_prompt(row)
            prompt, suffix, token_ids, token_tags, vision_inputs, packed_rows = _choose_prompt(
                pipeline.tokenizer,
                text_step,
                pipeline.processor,
                normalized,
                conditioning_prompt,
                args.context_parallel_degree,
                system_prompt=_system_prompt(row),
            )
            hidden = _get_qwen3vl_hidden50_memory_efficient(
                pipeline.text_encoder,
                pipeline.processor,
                token_ids,
                vision_inputs,
                device=device,
                dtype=torch.bfloat16,
            ).squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
            tags = torch.tensor(token_tags, dtype=torch.int64)
            if hidden.shape != (len(token_ids), 5120) or tags.shape != (len(token_ids),):
                raise ValueError("Qwen image+video presentation has an unexpected shape")
            if not bool(torch.isfinite(hidden).all()) or not bool(((tags == 0) | (tags == 1)).all()):
                raise ValueError("Qwen image+video presentation contains invalid values")
            atomic_save_safetensors(tensor_path, {"encoder_hidden_states": hidden, "text_token_tags": tags})
            record = {
                **row,
                **expected,
                "sample_id": sample_id,
                "original_prompt": conditioning_prompt,
                "prompt": prompt,
                "cp_prompt_suffix": suffix,
                "text_tensor_path": _relative(tensor_path, output_dir),
                "text_sha256": sha256_file(tensor_path),
                "num_text_rows": len(token_ids),
                "num_condition_image_rows": IMAGE_REFERENCE_ROWS,
                "num_condition_proxy_video_rows": PROXY_VIDEO_ROWS,
                "num_condition_video_rows": TOTAL_CONDITION_ROWS,
                "num_target_video_rows": TARGET_VIDEO_ROWS,
                "num_packed_rows": packed_rows,
                "presentation_token_ids_sha256": hashlib.sha256(
                    np.asarray(token_ids, dtype="<i8").tobytes()
                ).hexdigest(),
                "reference_image_pixel_shape": [TARGET_HEIGHT, TARGET_WIDTH, 3],
                "reference_image_latent_shape": list(IMAGE_REFERENCE_LATENT_SHAPE),
                "reference_video_pixel_shape": [TARGET_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, 3],
                "reference_video_latent_shape": list(TARGET_LATENT_SHAPE),
                "reference_normalization": {
                    "canvas_short_edge": TARGET_HEIGHT,
                    "canvas_max_pixels": TARGET_HEIGHT * TARGET_WIDTH,
                    "reference_image_short_edge": TARGET_HEIGHT,
                },
                "audio": "disabled",
            }
            atomic_write_json(done_path, record)
            successes += 1
        except Exception:
            failures += 1
            traceback.print_exc()
            if not args.continue_on_error:
                raise
        if index % 10 == 0 or index == len(assigned):
            print(f"cache-ref-image-video-text rank={args.rank} {index}/{len(assigned)} ok={successes} bad={failures}")
    return 0 if failures == 0 else 2


def run_vae(args: argparse.Namespace) -> int:
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    output_dir = args.output_dir.resolve()
    records = _read_manifest(args.manifest)
    assigned = _assigned(records, args.rank, args.world_size, args.limit)
    device = torch.device(args.device or f"cuda:{args.rank}")
    if device.type != "cuda":
        raise ValueError("VAE cache construction requires CUDA")
    torch.cuda.set_device(device)
    model_root = args.model_root.resolve()
    vae_hash = _fingerprint(model_root, "vae", "diffusion_pytorch_model.safetensors.index.json")
    text_hash = _fingerprint(model_root, "text_encoder", "model.safetensors.index.json")
    transformer_hash = _fingerprint(model_root, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json")
    vae = AutoencoderKLMiniMaxH3.from_pretrained(model_root / "vae", low_cpu_mem_usage=True).eval().to(device)
    vae.requires_grad_(False)
    vae.enable_tiling()
    setup_pipe = _configure_setup_pipeline()
    successes = failures = 0
    for index, row in enumerate(assigned, start=1):
        sample_id = str(row["sample_id"])
        text_path, text_done_path = _text_paths(output_dir, sample_id)
        sample_path, sample_done_path = _sample_paths(output_dir, sample_id)
        text_expected = _text_expected(row, args, text_hash, transformer_hash)
        try:
            text_done = _valid_done(text_done_path, text_path, "text_sha256", text_expected)
            if text_done is None:
                raise FileNotFoundError(f"Text cache is not ready for {sample_id}")
            expected = {
                "schema_version": REF_VIDEO_CACHE_SCHEMA_VERSION,
                "text_sha256": text_done["text_sha256"],
                "vae_config_hash": vae_hash,
                "text_encoder_hash": text_hash,
                "transformer_ref_config_hash": transformer_hash,
                "cache_context_parallel_degree": args.context_parallel_degree,
                "reference_kinds": ["image", "video"],
                "reference_has_audio": False,
                "reference_image_frozen": True,
                "source_media_fingerprint": row["source_media_fingerprint"],
            }
            if _valid_done(sample_done_path, sample_path, "sha256", expected) is not None:
                successes += 1
                continue
            _verify_source_hashes(row)
            target_frames = _decode_split(row["target_video_paths"])
            normalized = _normalized_references(setup_pipe, row)
            image_frame = np.asarray(normalized[0].image)
            proxy_frames = np.asarray(normalized[1].frames)
            if row["reference_image_source"] == TARGET_REFERENCE and not np.array_equal(image_frame, target_frames[0]):
                raise ValueError("Reference image is not exactly target split-screen frame zero")
            target_pixels = torch.from_numpy(target_frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            image_pixels = torch.from_numpy(image_frame.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)
            proxy_pixels = torch.from_numpy(proxy_frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            target_seed = _target_seed(args.seed_for_target, sample_id)
            with torch.inference_mode():
                target_latent = encode_vae_condition(vae, target_pixels, PIXEL_MEAN, PIXEL_STD, target_seed)
                image_latent = encode_vae_condition(
                    vae, image_pixels, PIXEL_MEAN, PIXEL_STD, args.reference_image_vae_seed
                )
                video_latent = encode_vae_condition(
                    vae, proxy_pixels, PIXEL_MEAN, PIXEL_STD, args.reference_video_vae_seed
                )
            if tuple(target_latent.shape) != (1, *TARGET_LATENT_SHAPE):
                raise ValueError(f"Unexpected target latent shape: {tuple(target_latent.shape)}")
            if tuple(image_latent.shape) != (1, *IMAGE_REFERENCE_LATENT_SHAPE):
                raise ValueError(f"Unexpected image-reference latent shape: {tuple(image_latent.shape)}")
            if tuple(video_latent.shape) != (1, *TARGET_LATENT_SHAPE):
                raise ValueError(f"Unexpected proxy-video latent shape: {tuple(video_latent.shape)}")
            atomic_save_safetensors(
                sample_path,
                {
                    "clean_target_video_latent": target_latent.squeeze(0).to(torch.bfloat16).contiguous(),
                    "clean_reference_image_latent": image_latent.squeeze(0).to(torch.bfloat16).contiguous(),
                    "clean_reference_video_latent": video_latent.squeeze(0).to(torch.bfloat16).contiguous(),
                },
            )
            record = {
                **text_done,
                **expected,
                "tensor_path": _relative(sample_path, output_dir),
                "sha256": sha256_file(sample_path),
                "text_tensor_path": _relative(text_path, output_dir),
                "target_vae_encode_seed": target_seed,
                "reference_image_vae_encode_seed": args.reference_image_vae_seed,
                "reference_video_vae_encode_seed": args.reference_video_vae_seed,
            }
            atomic_write_json(sample_done_path, record)
            successes += 1
        except Exception:
            failures += 1
            traceback.print_exc()
            if not args.continue_on_error:
                raise
        if index % 5 == 0 or index == len(assigned):
            print(f"cache-ref-image-video-vae rank={args.rank} {index}/{len(assigned)} ok={successes} bad={failures}")
    return 0 if failures == 0 else 2


def run_merge(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    records = _read_manifest(args.manifest)
    merged = []
    for row in records:
        _, done_path = _sample_paths(output_dir, str(row["sample_id"]))
        if not done_path.is_file():
            raise FileNotFoundError(done_path)
        record = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            record.get("text_presentation_revision") != PRESENTATION_REVISION
            or record.get("system_prompt_sha256") != hashlib.sha256(_system_prompt(row).encode()).hexdigest()
            or record.get("original_prompt") != _conditioning_prompt(row)
            or record.get("source_media_fingerprint") != row["source_media_fingerprint"]
        ):
            raise ValueError(f"Stale prompt or media metadata for {row['sample_id']}; rebuild text and rebind VAE")
        merged.append(record)
    atomic_write_jsonl(output_dir / "manifest.jsonl", merged)
    report = {
        "schema_version": REF_VIDEO_CACHE_SCHEMA_VERSION,
        "records": len(merged),
        "manifest_sha256": sha256_file(output_dir / "manifest.jsonl"),
        "target_profile": {"num_frames": 124, "fps": 24.0, "height": 480, "width": 1664},
        "references": [sorted({row["reference_image_source"] for row in records}), "proxy_video"],
        "status": "PASS",
    }
    atomic_write_json(output_dir / "merge_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


def _rebind_vae_record(old, text_done, row, *, vae_hash: str) -> dict[str, Any]:
    """Reuse visual tensors only when their independent media/model contracts agree."""
    for key in (
        "sample_id", "source_media_fingerprint", "target_video_paths", "proxy_video_paths",
        "target_video_sha256", "proxy_video_sha256", "reference_image_source", "split_screen",
        "num_frames", "fps", "height", "width",
    ):
        if old.get(key) != row.get(key) or text_done.get(key) != row.get(key):
            raise ValueError(f"Cannot reuse VAE tensors: {row['sample_id']} {key} mismatch")
    if old.get("vae_config_hash") != vae_hash or old.get("schema_version") != REF_VIDEO_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Cannot reuse VAE tensors: {row['sample_id']} VAE/schema mismatch")
    for key in ("reference_kinds", "reference_has_audio", "reference_image_frozen"):
        if old.get(key) != text_done.get(key):
            raise ValueError(f"Cannot reuse VAE tensors: {row['sample_id']} {key} mismatch")
    return {
        **text_done,
        "schema_version": REF_VIDEO_CACHE_SCHEMA_VERSION,
        "vae_config_hash": vae_hash,
        **{key: old[key] for key in (
            "sha256", "target_vae_encode_seed", "reference_image_vae_encode_seed", "reference_video_vae_encode_seed",
        )},
    }


def run_reuse_vae(args: argparse.Namespace) -> int:
    """Bind newly encoded text to unchanged visual tensors in an isolated cache."""
    if args.reuse_vae_manifest is None:
        raise ValueError("reuse-vae requires --reuse-vae-manifest")
    source_manifest = args.reuse_vae_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if source_manifest.parent == output_dir:
        raise ValueError("Text recaching must use a new directory, separate from the source cache")
    old_rows = {row["sample_id"]: row for row in map(json.loads, source_manifest.read_text().splitlines())}
    records = _read_manifest(args.manifest)
    assigned = _assigned(records, args.rank, args.world_size, args.limit)
    model_root = args.model_root.resolve()
    vae_hash = _fingerprint(model_root, "vae", "diffusion_pytorch_model.safetensors.index.json")
    text_hash = _fingerprint(model_root, "text_encoder", "model.safetensors.index.json")
    transformer_hash = _fingerprint(model_root, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json")
    for index, row in enumerate(assigned, start=1):
        sample_id = row["sample_id"]
        text_path, text_done_path = _text_paths(output_dir, sample_id)
        sample_path, sample_done_path = _sample_paths(output_dir, sample_id)
        expected = _text_expected(row, args, text_hash, transformer_hash)
        text_done = _valid_done(text_done_path, text_path, "text_sha256", expected)
        if text_done is None:
            raise ValueError(f"New text cache is missing or stale: {sample_id}")
        old = old_rows[sample_id]
        record = _rebind_vae_record(old, text_done, row, vae_hash=vae_hash)
        source = (source_manifest.parent / old["tensor_path"]).resolve()
        source.relative_to(source_manifest.parent)
        if sha256_file(source) != old["sha256"]:
            raise ValueError(f"Source VAE cache hash mismatch: {source}")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_path.exists():
            if sha256_file(sample_path) != old["sha256"]:
                raise ValueError(f"Conflicting VAE file in new cache: {sample_path}")
        else:
            try:
                os.link(source, sample_path)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, sample_path)
        record["tensor_path"] = _relative(sample_path, output_dir)
        record["vae_reused_from_manifest"] = str(source_manifest)
        atomic_write_json(sample_done_path, record)
        if index % 25 == 0 or index == len(assigned):
            print(f"cache-ref-image-video-reuse-vae {index}/{len(assigned)}", flush=True)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    manifest = output_dir / "manifest.jsonl"
    dataset = CachedH3RefVideoDataset(
        manifest,
        allowed_root=output_dir,
        verify_sha256=args.verify_sha256,
        allow_external_reference=True,
    )
    count = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    for index in range(count):
        sample = dataset[index]
        sample.build_layout(context_parallel_degree=args.context_parallel_degree)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"cache-ref-image-video-validate {index + 1}/{count}")
    complete = count == len(dataset)
    report = {
        "schema_version": REF_VIDEO_CACHE_SCHEMA_VERSION,
        "records": len(dataset),
        "manifest_sha256": sha256_file(manifest),
        "validated": count,
        "verify_sha256": bool(args.verify_sha256),
        "complete": complete,
        "status": "PASS" if complete else "SMOKE",
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("text", "vae", "reuse-vae", "merge", "validate"))
    parser.add_argument("--manifest", type=Path, required=True, help="Raw split-screen pair manifest JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--reuse-vae-manifest", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--context-parallel-degree", type=int, default=4)
    parser.add_argument("--seed-for-target", type=int, default=17031)
    parser.add_argument("--reference-image-vae-seed", type=int, default=42)
    parser.add_argument("--reference-video-vae-seed", type=int, default=42)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    if args.context_parallel_degree <= 0:
        raise ValueError("context-parallel-degree must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.stage == "text":
        return run_text(args)
    if args.stage == "vae":
        return run_vae(args)
    if args.stage == "reuse-vae":
        return run_reuse_vae(args)
    if args.stage == "merge":
        return run_merge(args)
    return run_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
