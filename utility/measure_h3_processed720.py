#!/usr/bin/env python3
"""Audit a 720p split-screen dataset and measure one real CP=1 LoRA step.

This is a capacity probe, not a batch-training launcher. It reuses the existing
Ref2VA model, prompt formatter, conditioner, VAE and loss. Source videos and
captions stay untouched; the 720 rows are edge-padded to 736 in memory.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from local_diffusers import use_local_diffusers

use_local_diffusers()

from utility.h3_ref_video_prompt import (
    DEFAULT_TASK_PROMPT, PRESENTATION_REVISION, SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256, wrap_reference_presentation,
)

MODEL_ROOT = Path("models/MiniMax-H3")
PROFILE = {
    "num_frames": 124, "fps": 24.0, "source_height": 720, "width": 2560,
    "height": 736, "pad_top": 8, "pad_bottom": 8, "padding": "edge",
    "left_agent": "agent_000", "right_agent": "agent_001", "seam_column": 1280,
    "video_latent_shape": [24, 37, 46, 160],
    "image_latent_shape": [24, 1, 46, 160],
    "image_rows": 1840, "proxy_rows": 68080, "target_rows": 68080,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


def user_prompt(row):
    scene = row["scene_description"].strip()
    if not scene:
        raise ValueError(f"Missing scene caption: {row['sample_id']}")
    return DEFAULT_TASK_PROMPT + "\nScene description: " + scene


def decode(path, *, first_only=False):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if abs(float(stream.average_rate) - PROFILE["fps"]) > 1e-6:
            raise ValueError(f"Unexpected fps: {path}")
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if first_only:
                break
    result = np.stack(frames)
    expected = (1 if first_only else 124, 720, 2560, 3)
    if result.shape != expected:
        raise ValueError(f"{path}: {result.shape} != {expected}")
    padded = np.pad(result, ((0, 0), (8, 8), (0, 0), (0, 0)), mode="edge")
    if not np.array_equal(padded[:, 8:728], result):
        raise AssertionError("Padding changed source pixels")
    return np.ascontiguousarray(padded)


def reference_image_pixels(row):
    from utility.h3_ref_video_reference import EXTERNAL_REFERENCE, TARGET_REFERENCE, load_external_reference
    if row.get("reference_image_source") == EXTERNAL_REFERENCE:
        pixels = load_external_reference(row)
        if pixels.shape != (720, 2560, 3):
            raise ValueError(f"Unexpected 720p style reference: {pixels.shape}")
        return np.pad(pixels, ((8, 8), (0, 0), (0, 0)), mode="edge")
    if row.get("reference_image_source") != TARGET_REFERENCE:
        raise ValueError("Unknown reference image source")
    return decode(row["target_video_path"], first_only=True)[0]


def references(row):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    first = reference_image_pixels(row)
    from utility.h3_proxy_reference import compact_references, validate_proxy_profile
    if validate_proxy_profile(row) is not None:
        return compact_references(row, first)
    proxy = decode(row["proxy_video_path"])
    setup = MiniMaxH3Ref2VASetupStep().init_pipeline()
    setup.update_components(canvas_short_edge=736, canvas_max_pixels=736 * 2560, reference_image_short_edge=736)
    normalized = setup(
        references=[MiniMaxH3ImageReference(image=first), MiniMaxH3VideoReference(frames=proxy, fps=24.0)],
        height=736, width=2560, num_frames=124, output="normalized_references",
    )
    if not np.array_equal(np.asarray(normalized[0].image), first):
        raise AssertionError("Setup changed the padded RGB anchor")
    if not np.array_equal(np.asarray(normalized[1].frames), proxy):
        raise AssertionError("Setup changed the padded proxy")
    return normalized


def presentation(row, model_root):
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_root / "tokenizer", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(model_root / "processor", local_files_only=True)
    normalized = references(row)
    step = MiniMaxH3Ref2VATextEncoderStep()
    vision, images, videos, timestamps = step._gather_vision_features(processor, normalized, 24.0)
    ids, tags = step._build_presentation(tokenizer, user_prompt(row), normalized, images, videos, timestamps)
    ids, tags = wrap_reference_presentation(tokenizer, processor, ids, tags, system_prompt=SYSTEM_PROMPT)
    return tokenizer, processor, ids, tags, vision


def audit(args):
    from transformers import Qwen2TokenizerFast
    from tqdm import tqdm
    if (args.run_dir / "audit.json").exists():
        raise FileExistsError("Use a new run directory for a new audit")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_root / "tokenizer", local_files_only=True)
    seen, token_counts = set(), []
    for row in tqdm(rows, desc="audit media and captions", unit="case"):
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate: {sample_id}")
        seen.add(sample_id)
        if (row["num_frames"], row["fps"], row["height"], row["width"]) != (124, 24.0, 720, 2560):
            raise ValueError(f"Unexpected source profile: {sample_id}")
        split = row["split_screen"]
        if (split["layout"], split["left_agent"], split["right_agent"], split["seam_column"]) != (
            "horizontal", "agent_000", "agent_001", 1280
        ):
            raise ValueError(f"Invalid left/right layout: {sample_id}")
        for field in ("target_video_path", "proxy_video_path", "caption_path"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        caption_bytes = Path(row["caption_path"]).read_bytes()
        if caption_bytes.decode().strip() != row["scene_description"].strip():
            raise ValueError(f"Caption mismatch: {sample_id}")
        if hashlib.sha256(caption_bytes).hexdigest() != row["caption_sha256"]:
            raise ValueError(f"Caption hash mismatch: {sample_id}")
        row.update(prompt=DEFAULT_TASK_PROMPT, system_prompt=SYSTEM_PROMPT, system_prompt_sha256=SYSTEM_PROMPT_SHA256)
        token_counts.append(len(tokenizer(user_prompt(row), add_special_tokens=False)["input_ids"]))
    if not rows:
        raise ValueError("No cases")
    selected_index = max(range(len(rows)), key=token_counts.__getitem__)
    selected = rows[selected_index]
    log(f"Probe case with longest caption: {selected['sample_id']}")
    for field, hash_field in (("target_video_path", "target_video_file_sha256"), ("proxy_video_path", "proxy_video_file_sha256")):
        if sha256(selected[field]) != selected[hash_field]:
            raise ValueError(f"Probe media hash mismatch: {field}")
    _, _, ids, tags, _ = presentation(selected, args.model_root)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "raw_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.run_dir / "probe_case.json", selected)
    (args.run_dir / "prompt_preview.txt").write_text(
        "[system]\n" + SYSTEM_PROMPT + "\n\n[user]\n<Picture 1>: [padded target first frame]\n"
        "<Video 1>: [padded proxy video]\n" + user_prompt(selected) + "\n"
    )
    static_rows = len(ids) - token_counts[selected_index]
    report = {
        "created_at": utc_now(), "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest), "cases": len(rows), "profile": PROFILE,
        "requested_gpus": 4, "data_parallel_degree": 4, "context_parallel_degree": 1,
        "per_gpu_batch_size": 1, "gradient_accumulation": 1, "effective_batch_size": 4,
        "probe_case": selected["sample_id"], "probe_case_index": selected_index,
        "text_presentation_revision": PRESENTATION_REVISION, "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "max_text_rows": len(ids), "min_text_rows": min(token_counts) + static_rows,
        "max_packed_rows": len(ids) + 138000,
        "num_text_modality_rows": sum(tag == 1 for tag in tags),
        "num_vision_presentation_rows": sum(tag == 0 for tag in tags),
        "legacy_text_cache_limit": 16384, "fits_legacy_text_cache_limit": len(ids) <= 16384,
        "base_transformer_checkpoint_bytes": json.loads((args.model_root / "transformer_ref" / "diffusion_pytorch_model.safetensors.index.json").read_text())["metadata"]["total_size"],
        "training_started": False,
    }
    write_json(args.run_dir / "audit.json", report)
    log(json.dumps(report, ensure_ascii=False))


def text_cache(args):
    from safetensors.torch import save_file
    from utility.cache_h3_ref_video import _load_text_components, _get_qwen3vl_hidden50_memory_efficient
    row = json.loads((args.run_dir / "probe_case.json").read_text())
    _, processor, ids, tags, vision = presentation(row, args.model_root)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    log(f"Encoding real system + references + scene: {len(ids)} rows")
    pipeline = _load_text_components(args.model_root, device)
    hidden = _get_qwen3vl_hidden50_memory_efficient(pipeline.text_encoder, processor, ids, vision, device, torch.bfloat16)
    hidden = hidden.squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
    if hidden.shape != (len(ids), 5120) or not torch.isfinite(hidden).all():
        raise ValueError("Invalid Qwen hidden states")
    save_file({"encoder_hidden_states": hidden, "text_token_tags": torch.tensor(tags, dtype=torch.int64)}, str(args.run_dir / "probe_text.safetensors"))
    log("Text cache complete")


def vae_cache(args):
    from safetensors.torch import save_file
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from utility.cache_h3_ref_video import PIXEL_MEAN, PIXEL_STD, _target_seed
    row = json.loads((args.run_dir / "probe_case.json").read_text())
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model_root / "vae", low_cpu_mem_usage=True).eval().to(device)
    vae.requires_grad_(False)
    vae.enable_tiling()
    target = decode(row["target_video_path"])
    proxy = decode(row["proxy_video_path"])
    tensors = {}
    for name, pixels, seed, expected in (
        ("clean_target_video_latent", target, _target_seed(17031, row["sample_id"]), PROFILE["video_latent_shape"]),
        ("clean_reference_image_latent", target[:1], 42, PROFILE["image_latent_shape"]),
        ("clean_reference_video_latent", proxy, 42, PROFILE["video_latent_shape"]),
    ):
        log(f"Encoding {name} at 2560x736 (full 720p content preserved)")
        values = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        with torch.inference_mode():
            latent = encode_vae_condition(vae, values, PIXEL_MEAN, PIXEL_STD, seed)
        if tuple(latent.shape) != (1, *expected) or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid {name}: {tuple(latent.shape)}")
        tensors[name] = latent.squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
        del values, latent
        torch.cuda.empty_cache()
    save_file(tensors, str(args.run_dir / "probe_vae.safetensors"))
    log("VAE cache complete")


def train_probe(args):
    from peft import LoraConfig
    from safetensors.torch import load_file
    from diffusers import MiniMaxH3Transformer3DModel
    from train_h3_ref_video_lora import ALL_BLOCK_LORA_PATTERN, sample_h3_timestep
    from utility.h3_ref_video import H3RefVideoTrainingModel, build_ref_video_layout, patchify_ref_video_latents
    from utility.h3_distill import TrainableEMA
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = {"value": "load_model"}
    started = time.perf_counter()
    result = {
        "started_at": utc_now(), "passed": False, "probe_kind": "real_single_gpu_forward_backward_optimizer",
        "lora_rank": args.lora_rank, "lora_targets": "qkvo-ffn", "gradient_checkpointing": True,
        "activation_cpu_offload": True, "context_parallel_degree": 1, "batch_size": 1,
        "gpu": torch.cuda.get_device_name(device), "profile": PROFILE,
        "audit": str(args.run_dir / "audit.json"),
    }
    def mark(name):
        def hook(_module, _inputs):
            stage["value"] = name
            if name in ("block.0", "block.10", "block.20", "block.30", "block.40"):
                log(f"{name}; allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        return hook
    try:
        free, total = torch.cuda.mem_get_info(device)
        result.update(initial_free_bytes=free, total_gpu_bytes=total)
        if free < 75 * 2**30:
            raise RuntimeError("Probe requires an otherwise idle H100 80GB")
        log("Loading the existing full transformer_ref checkpoint")
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(args.model_root / "transformer_ref", dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
        transformer.requires_grad_(False)
        transformer.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0, target_modules=ALL_BLOCK_LORA_PATTERN, init_lora_weights=True), adapter_name="ref_video")
        transformer.enable_gradient_checkpointing()
        transformer.train()
        transformer.token_refiner.register_forward_pre_hook(mark("text_refiner"))
        for index, block in enumerate(transformer.transformer_blocks):
            block.register_forward_pre_hook(mark(f"block.{index}"))
            block.attn.register_forward_pre_hook(mark(f"block.{index}.attention"))
            block.ff.register_forward_pre_hook(mark(f"block.{index}.feed_forward"))
        model = H3RefVideoTrainingModel(transformer, reference_noise_aug=0.999)
        parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.99), weight_decay=0, foreach=False)
        ema = TrainableEMA(model, decay=0.999)
        result["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        result["model_and_ema_allocated_bytes"] = torch.cuda.memory_allocated()
        values = load_file(str(args.run_dir / "probe_vae.safetensors"))
        text_values = load_file(str(args.run_dir / "probe_text.safetensors"))
        layout = build_ref_video_layout(
            text_token_tags=text_values["text_token_tags"], num_image_reference_latent_frames=1,
            image_reference_latent_height=46, image_reference_latent_width=160,
            num_reference_latent_frames=37, reference_latent_height=46, reference_latent_width=160,
            num_target_latent_frames=37, target_latent_height=46, target_latent_width=160,
            context_parallel_degree=1,
        ).to(device)
        result["packed_rows"] = layout.sequence_length
        target, reference = patchify_ref_video_latents(
            values["clean_target_video_latent"].unsqueeze(0).to(device, torch.float32),
            values["clean_reference_video_latent"].unsqueeze(0).to(device, torch.float32), layout,
            reference_image_latent=values["clean_reference_image_latent"].unsqueeze(0).to(device, torch.float32),
        )
        prompt = text_values["encoder_hidden_states"].unsqueeze(0).to(device, torch.float32)
        generator = torch.Generator(device=device).manual_seed(42)
        target_noise = torch.randn(target.shape, device=device, dtype=torch.float32, generator=generator)
        reference_noise = torch.randn(reference.shape, device=device, dtype=torch.float32, generator=generator)
        timestep = sample_h3_timestep(generator=generator, shift=12.0, device=device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        stage["value"] = "forward"
        log(f"Real CP=1 training probe, {layout.sequence_length} packed rows")
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            output = model.flow_matching_loss(clean_target_rows=target, target_noise=target_noise,
                clean_reference_rows=reference, reference_noise=reference_noise,
                prompt_embeds=prompt, layout=layout, h3_timestep=timestep)
        if not bool(torch.isfinite(output.loss)):
            raise FloatingPointError("Non-finite loss")
        stage["value"] = "backward"
        log(f"Forward complete, loss={float(output.loss):.6f}; running backward")
        output.loss.backward()
        stage["value"] = "optimizer"
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        ema.update()
        torch.cuda.synchronize()
        result.update(passed=True, loss=float(output.loss), gradient_norm=float(norm), step_seconds=time.perf_counter() - step_started)
        stage["value"] = "complete"
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), oom=isinstance(error, torch.OutOfMemoryError))
        (args.run_dir / "probe_traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        result.update(stage=stage["value"], finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started,
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated(), max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(args.run_dir / "probe_result.json", result)
        log(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "text", "vae", "train-probe"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "datasets/GTA/processed_720/paired_manifest.jsonl")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    function = {"audit": audit, "text": text_cache, "vae": vae_cache, "train-probe": train_probe}[args.stage]
    return function(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
