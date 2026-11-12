#!/usr/bin/env python3
"""Cache native paired 720p GTA samples using the existing H3 encoders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "verification", ROOT / "diffusers" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch
from tqdm import tqdm
from local_diffusers import use_local_diffusers

use_local_diffusers()

from utility.cache_h3_ref_video import (
    _fingerprint, _load_text_components, _get_qwen3vl_hidden50_memory_efficient,
    _choose_prompt, _conditioning_prompt, _system_prompt, _target_seed,
    TEXT_ENCODER_EXECUTION_REVISION, PIXEL_MEAN, PIXEL_STD,
)
from utility.h3_ref_video_data import CachedH3RefVideoDataset, REF_VIDEO_CACHE_SCHEMA_VERSION
from utility.h3_ref_video_prompt import DEFAULT_TASK_PROMPT, PRESENTATION_REVISION, SYSTEM_PROMPT, SYSTEM_PROMPT_SHA256
from utility.measure_h3_processed720 import decode, references, reference_image_pixels
from utility.h3_ref_video_reference import EXTERNAL_REFERENCE
from utility.h3_proxy_reference import compact_profile, validate_proxy_profile, decode_proxy, text_encoder_step
from data.worldscene_h3_ref_depth_v1.tools.common import atomic_write_json, atomic_write_jsonl, atomic_save_safetensors, sha256_file

MODEL_ROOT = Path("models/MiniMax-H3")
REVISION = "gta-native720-edgepad736-ref2va-v1"
PADDING = {"top": 8, "bottom": 8, "left": 0, "right": 0, "mode": "edge"}
IMAGE_ROWS = 1840
VIDEO_ROWS = 68080


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path, *, preserve_system_prompt=False):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty unique paired cases")
    for row in rows:
        validate_proxy_profile(row)
        if (row["height"], row["width"], row["num_frames"], row["fps"]) != (720, 2560, 124, 24.0):
            raise ValueError(f"Unexpected raw profile: {row['sample_id']}")
        if row.get("prompt") != DEFAULT_TASK_PROMPT:
            raise ValueError("Prepare the manifest using the current task prompt before caching")
        if preserve_system_prompt:
            # Long CWM rollouts use system_w0/system_wn per window.  The
            # checkpoint contract is bound to the original training manifest,
            # not to the inference cache's text presentation, so retain and
            # validate the explicit row-level prompt here.
            explicit = row.get("system_prompt")
            if not isinstance(explicit, str) or not explicit.strip():
                raise ValueError(f"Missing explicit system_prompt: {row['sample_id']}")
            digest = hashlib.sha256(explicit.encode()).hexdigest()
            if row.get("system_prompt_sha256") != digest:
                raise ValueError(f"system_prompt SHA mismatch: {row['sample_id']}")
        elif row.get("system_prompt_sha256") != SYSTEM_PROMPT_SHA256:
            raise ValueError("Prepare the manifest using the current system/task prompt before caching")
        if not isinstance(row.get("scene_description"), str) or not row["scene_description"].strip():
            raise ValueError(f"Missing scene caption: {row['sample_id']}")
    return rows


def prepare(args):
    from transformers import Qwen2TokenizerFast
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_root / "tokenizer", local_files_only=True)
    for row in tqdm(rows, desc="prepare 720p prompts", mininterval=1):
        if getattr(args, "compact_proxy", False):
            row["proxy_reference_profile"] = compact_profile()
        row["prompt"] = DEFAULT_TASK_PROMPT
        if args.preserve_system_prompt:
            explicit = row.get("system_prompt")
            if not isinstance(explicit, str) or not explicit.strip():
                raise ValueError(f"Missing explicit system_prompt: {row['sample_id']}")
            row["system_prompt_sha256"] = hashlib.sha256(explicit.encode()).hexdigest()
        else:
            row.update(system_prompt=SYSTEM_PROMPT, system_prompt_sha256=SYSTEM_PROMPT_SHA256)
        if Path(row["caption_path"]).read_text().strip() != row["scene_description"].strip():
            raise ValueError(f"Caption differs: {row['sample_id']}")
        if sha256_file(Path(row["caption_path"])) != row["caption_sha256"]:
            raise ValueError(f"Caption SHA mismatch: {row['sample_id']}")
        for name in ("target_video_path", "proxy_video_path"):
            if not Path(row[name]).is_file():
                raise FileNotFoundError(row[name])
        row["scene_caption_token_count"] = len(tokenizer(_conditioning_prompt(row), add_special_tokens=False)["input_ids"])
    # Put the longest captions first so every replica in the real smoke sees a
    # worst-case length. The training order is independently shuffled below.
    pilot = sorted(rows, key=lambda row: row["scene_caption_token_count"], reverse=True)[:4]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rows))
    rows = [rows[int(index)] for index in order]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "raw_manifest.jsonl"
    if raw.exists():
        if read_rows(raw, preserve_system_prompt=args.preserve_system_prompt) != rows:
            raise ValueError("Existing prepared manifest differs")
    else:
        atomic_write_jsonl(raw, rows)
    read_rows(raw, preserve_system_prompt=args.preserve_system_prompt)
    atomic_write_jsonl(args.output_dir / "pilot_manifest.jsonl", pilot)
    report_system_prompt = (
        sorted({row["system_prompt_sha256"] for row in rows})
        if args.preserve_system_prompt else SYSTEM_PROMPT_SHA256
    )
    atomic_write_json(args.output_dir / "prepare_report.json", {
        "created_at": now(), "cases": len(rows), "manifest_sha256": sha256_file(raw),
        "system_prompt_sha256": report_system_prompt, "presentation_revision": PRESENTATION_REVISION,
        "preserve_system_prompt": bool(args.preserve_system_prompt),
        "training_order": "numpy default_rng permutation seed 42", "pilot_ids": [r["sample_id"] for r in pilot],
        "source_canvas": [720, 2560], "training_canvas": [736, 2560], "padding": PADDING,
        "proxy_reference_profile": rows[0].get("proxy_reference_profile"),
    })
    preview_system = pilot[0]["system_prompt"]
    (args.output_dir / "prompt_preview.txt").write_text("[system]\n" + preview_system + "\n\n[user]\n<Picture 1>: [first target frame]\n<Video 1>: [proxy]\n" + _conditioning_prompt(pilot[0]) + "\n")
    print(json.dumps({"cases": len(rows), "pilot_ids": [r["sample_id"] for r in pilot]}), flush=True)


def paths(cache, sample_id):
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in sample_id):
        raise ValueError("Unsafe sample ID")
    return {
        "text": cache / "text" / f"{sample_id}.safetensors",
        "text_done": cache / "text_done" / f"{sample_id}.json",
        "vae": cache / "samples" / f"{sample_id}.safetensors",
        "done": cache / "sample_done" / f"{sample_id}.json",
    }


def fingerprint(row, args):
    payload = {
        "revision": REVISION, "model_root": str(args.model_root.resolve()),
        "source": {key: row[key] for key in ("sample_id", "target_video_file_sha256", "proxy_video_file_sha256", "scene_description", "system_prompt_sha256")},
        "cp": args.context_parallel_degree, "padding": PADDING,
    }
    if row.get("reference_image_source") == EXTERNAL_REFERENCE:
        payload["external_reference_sha256"] = row["reference_image_sha256"]
    if validate_proxy_profile(row) is not None:
        payload["proxy_reference_profile"] = row["proxy_reference_profile"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def is_done(marker, tensor, expected):
    if not marker.is_file() or not tensor.is_file():
        return False
    record = json.loads(marker.read_text())
    return record.get("cache_fingerprint") == expected and record.get("tensor_size") == tensor.stat().st_size


def common_metadata(row, args):
    return {
        **row, "height": 736, "width": 2560,
        "source_canvas": [720, 2560], "canvas_padding": PADDING,
        "split_screen": {**row["split_screen"], "height": 736},
        "cache_revision": REVISION, "cache_fingerprint": fingerprint(row, args),
        "cache_context_parallel_degree": args.context_parallel_degree,
        "text_encoder_layer": 50, "text_encoder_execution_revision": TEXT_ENCODER_EXECUTION_REVISION,
        "text_presentation_revision": PRESENTATION_REVISION,
        "reference_kinds": ["image", "video"], "reference_has_audio": False,
        "reference_image_frozen": True, "audio": "disabled",
    }


def progress(args, stage, assigned, completed, started, sample_id=None, error=None):
    atomic_write_json(args.output_dir / "status" / f"{stage}-{args.rank:02d}.json", {
        "updated_at": now(), "rank": args.rank, "stage": stage, "total": len(assigned),
        "completed": completed, "sample_id": sample_id, "elapsed_seconds": time.perf_counter() - started,
        "error": error,
    })


def cache_text(args):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    # Reuse the original presentation code with counts for this canvas.
    import utility.cache_h3_ref_video as legacy
    legacy.TARGET_VIDEO_ROWS = VIDEO_ROWS
    rows = read_rows(args.manifest, preserve_system_prompt=args.preserve_system_prompt)[args.rank::args.world_size]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    model = _load_text_components(args.model_root, device)
    hashes = {
        "text_encoder_hash": _fingerprint(args.model_root, "text_encoder", "model.safetensors.index.json"),
        "transformer_ref_config_hash": _fingerprint(args.model_root, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json"),
    }
    started = time.perf_counter()
    progress(args, "text", rows, 0, started)
    for index, row in enumerate(tqdm(rows, desc=f"text {args.rank}", mininterval=10), 1):
        files = paths(args.output_dir, row["sample_id"])
        expected = fingerprint(row, args)
        if not is_done(files["text_done"], files["text"], expected):
            try:
                profile = validate_proxy_profile(row)
                proxy_rows = 5760 if profile else VIDEO_ROWS
                legacy.TOTAL_CONDITION_ROWS = IMAGE_ROWS + proxy_rows
                step = text_encoder_step(row)
                normalized = references(row)
                prompt, suffix, ids, tags, vision, packed = _choose_prompt(
                    model.tokenizer, step, model.processor, normalized, _conditioning_prompt(row),
                    args.context_parallel_degree, system_prompt=_system_prompt(row),
                )
                hidden = _get_qwen3vl_hidden50_memory_efficient(model.text_encoder, model.processor, ids, vision, device, torch.bfloat16)
                hidden = hidden.squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
                if hidden.shape != (len(ids), 5120) or not torch.isfinite(hidden).all():
                    raise ValueError("Invalid conditioner output")
                atomic_save_safetensors(files["text"], {"encoder_hidden_states": hidden, "text_token_tags": torch.tensor(tags, dtype=torch.int64)})
                metadata = {
                    **common_metadata(row, args), **hashes,
                    "original_prompt": _conditioning_prompt(row), "prompt": prompt, "cp_prompt_suffix": suffix,
                    "text_tensor_path": str(files["text"].relative_to(args.output_dir)),
                    "text_sha256": sha256_file(files["text"]), "tensor_size": files["text"].stat().st_size,
                    "num_text_rows": len(ids), "num_condition_image_rows": IMAGE_ROWS,
                    "num_condition_proxy_video_rows": proxy_rows, "num_condition_video_rows": IMAGE_ROWS + proxy_rows,
                    "num_target_video_rows": VIDEO_ROWS, "num_packed_rows": packed,
                    "presentation_token_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                }
                atomic_write_json(files["text_done"], metadata)
                del hidden, vision, normalized
            except Exception as error:
                progress(args, "text", rows, index - 1, started, row["sample_id"], repr(error))
                raise
        progress(args, "text", rows, index, started, row["sample_id"])


def cache_vae(args):
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    rows = read_rows(args.manifest, preserve_system_prompt=args.preserve_system_prompt)[args.rank::args.world_size]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model_root / "vae", low_cpu_mem_usage=True).eval().to(device)
    vae.requires_grad_(False)
    vae.enable_tiling()
    vae_hash = _fingerprint(args.model_root, "vae", "diffusion_pytorch_model.safetensors.index.json")
    started = time.perf_counter()
    progress(args, "vae", rows, 0, started)
    for index, row in enumerate(tqdm(rows, desc=f"vae {args.rank}", mininterval=10), 1):
        files = paths(args.output_dir, row["sample_id"])
        expected = fingerprint(row, args)
        if not is_done(files["done"], files["vae"], expected):
            try:
                if not is_done(files["text_done"], files["text"], expected):
                    raise ValueError(f"Text must finish before VAE: {row['sample_id']}")
                text = json.loads(files["text_done"].read_text())
                for field, hash_field in (("target_video_path", "target_video_file_sha256"), ("proxy_video_path", "proxy_video_file_sha256")):
                    if sha256_file(Path(row[field])) != row[hash_field]:
                        raise ValueError(f"Source media SHA mismatch: {row['sample_id']} {field}")
                target = decode(row["target_video_path"])
                proxy = decode_proxy(row, for_vae=True)
                profile = validate_proxy_profile(row)
                proxy_shape = tuple(profile["latent_shape"]) if profile else (24, 37, 46, 160)
                reference = reference_image_pixels(row)[None] if row.get("reference_image_source") == EXTERNAL_REFERENCE else target[:1]
                tensors = {}
                for name, pixels, seed, shape in (
                    ("clean_target_video_latent", target, _target_seed(17031, row["sample_id"]), (24, 37, 46, 160)),
                    ("clean_reference_image_latent", reference, 42, (24, 1, 46, 160)),
                    ("clean_reference_video_latent", proxy, 42, proxy_shape),
                ):
                    values = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        latent = encode_vae_condition(vae, values, PIXEL_MEAN, PIXEL_STD, seed)
                    if tuple(latent.shape) != (1, *shape) or not torch.isfinite(latent).all():
                        raise ValueError(f"Invalid {name}: {tuple(latent.shape)}")
                    tensors[name] = latent.squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
                    del values, latent
                atomic_save_safetensors(files["vae"], tensors)
                metadata = {
                    **text, "schema_version": REF_VIDEO_CACHE_SCHEMA_VERSION, "vae_config_hash": vae_hash,
                    "tensor_path": str(files["vae"].relative_to(args.output_dir)),
                    "tensor_size": files["vae"].stat().st_size, "sha256": sha256_file(files["vae"]),
                    "target_vae_encode_seed": _target_seed(17031, row["sample_id"]),
                    "reference_image_vae_encode_seed": 42, "reference_video_vae_encode_seed": 42,
                }
                atomic_write_json(files["done"], metadata)
                del tensors, target, proxy, reference
            except Exception as error:
                progress(args, "vae", rows, index - 1, started, row["sample_id"], repr(error))
                raise
        progress(args, "vae", rows, index, started, row["sample_id"])


def merge(args):
    rows = read_rows(args.manifest, preserve_system_prompt=args.preserve_system_prompt)
    records = []
    for row in rows:
        files = paths(args.output_dir, row["sample_id"])
        if not is_done(files["done"], files["vae"], fingerprint(row, args)):
            raise ValueError(f"Missing/stale cache: {row['sample_id']}")
        records.append(json.loads(files["done"].read_text()))
    manifest = args.output_dir / args.merged_name
    atomic_write_jsonl(manifest, records)
    dataset = CachedH3RefVideoDataset(manifest, allow_external_reference=any(row.get("reference_image_source") == EXTERNAL_REFERENCE for row in rows))
    from safetensors import safe_open
    for row in tqdm(dataset.records, desc="verify 720p cache", mininterval=10):
        for field, hash_field in (("tensor_path", "sha256"), ("text_tensor_path", "text_sha256")):
            path = Path(row[field])
            if sha256_file(path) != row[hash_field]:
                raise ValueError(f"Cache SHA mismatch: {path}")
        with safe_open(row["tensor_path"], framework="pt", device="cpu") as handle:
            profile = validate_proxy_profile(row)
            proxy_shape = profile["latent_shape"] if profile else [24, 37, 46, 160]
            expected_shapes = {"clean_target_video_latent": [24, 37, 46, 160], "clean_reference_video_latent": proxy_shape, "clean_reference_image_latent": [24, 1, 46, 160]}
            if set(handle.keys()) != set(expected_shapes):
                raise ValueError("Wrong VAE keys")
            for key, shape in expected_shapes.items():
                if handle.get_slice(key).get_shape() != shape or handle.get_slice(key).get_dtype() != "BF16":
                    raise ValueError(f"Wrong shape/dtype: {row['sample_id']} {key}")
        with safe_open(row["text_tensor_path"], framework="pt", device="cpu") as handle:
            if handle.get_slice("encoder_hidden_states").get_shape() != [row["num_text_rows"], 5120]:
                raise ValueError("Wrong text shape")
        if row["num_packed_rows"] % args.context_parallel_degree:
            raise ValueError("CP divisibility mismatch")
    report = {"status": "PASS", "complete": True, "verify_sha256": True, "records": len(records), "validated": len(records), "manifest_sha256": sha256_file(manifest), "manifest": str(manifest), "finished_at": now()}
    atomic_write_json(args.output_dir / (Path(args.merged_name).stem + ".validation.json"), report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "text", "vae", "merge"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--context-parallel-degree", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--merged-name", default="manifest.jsonl")
    parser.add_argument("--preserve-system-prompt", action="store_true",
                        help="Keep explicit per-row system_prompt values (CWM w0/wn rollout cache)")
    parser.add_argument("--compact-proxy", action="store_true",
                        help="Prepare half-resolution proxy references sampled every 4 source frames plus the last")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.model_root = args.model_root.resolve()
    if not 0 <= args.rank < args.world_size or args.context_parallel_degree not in (1, 2, 4, 8):
        raise ValueError("Invalid worker rank or CP degree")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "text": cache_text, "vae": cache_vae, "merge": merge}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
