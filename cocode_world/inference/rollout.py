#!/usr/bin/env python3
"""Generate long videos with overlapping MiniMax-H3 latent windows.

Each 124-frame window is denoised independently, but the handoff is kept in
latent space.  A continuation receives the preceding 12 temporal latents and
projects them to ``t * clean + (1-t) * global_noise`` before every transformer
forward.  The target noise is one deterministic case-level trajectory sliced
by absolute latent position.  A single RGB boundary frame from the preceding
window is encoded as the image reference for the next window.  All generated
windows are concatenated in latent space and decoded once at render time.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from cocode_world.inference.video import _load_decoder, _pipeline_videos
from cocode_world.inference.loading import _hashes, _init_cp8, _load_model
from cocode_world.conditioning import PIXEL_MEAN, PIXEL_STD
from cocode_world.noise import shifted_sigma_schedule
from cocode_world.flow import h3_euler_between
from cocode_world.layout import patchify_video, unpatchify_video
from cocode_world.model import H3RefVideoTrainingModel, patchify_ref_video_latents
from cocode_world.dataset import CachedH3RefVideoDataset, REF_VIDEO_CACHE_SCHEMA_VERSION
from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition


FPS = 24
WINDOW_FRAMES = 124
OVERLAP_FRAMES = 39
STRIDE_FRAMES = 85
PREFIX_LATENTS = 12
WINDOW_COUNT = 17
WINDOW_LATENTS = 37
NEW_LATENTS = 25
VIDEO_ROWS_PER_LATENT = 1840


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_tensor(path: Path, key: str, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.contiguous()}, str(temporary))
    temporary.replace(path)


def _contract(args: argparse.Namespace, dataset: CachedH3RefVideoDataset, plan: dict) -> dict:
    state = json.loads((args.checkpoint / "trainer_state.json").read_text())
    meta = state.get("metadata", {})
    if state.get("step") != 4000:
        raise ValueError("Evaluation requires the released step-4000 checkpoint")
    row = dataset.records[0]
    expected = {
        "dataset_manifest_sha256": plan["training_manifest_sha256"],
        "data_schema": REF_VIDEO_CACHE_SCHEMA_VERSION,
        "target_canvas": [736, 2560],
        "context_parallel_degree": 8,
        "reference_kinds": ["image", "video"],
        "reference_image_frozen": True,
        "reference_noise_aug": 0.999,
        "video_shift": 12.0,
        "transformer_ref_config_hash": row["transformer_ref_config_hash"],
        "dataset_vae_config_hash": row["vae_config_hash"],
        "dataset_text_encoder_hash": row["text_encoder_hash"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"checkpoint contract mismatch for {key}: {meta.get(key)!r} != {value!r}")
    if plan.get("schema_version") != "cocode-world-long-video-v1":
        raise ValueError("Unsupported evaluation plan schema")
    if plan.get("window_frames") != WINDOW_FRAMES or plan.get("overlap_frames") != OVERLAP_FRAMES or plan.get("stride_frames") != STRIDE_FRAMES:
        raise ValueError("suite geometry is not 124/39/85")
    if plan.get("fixed_prefix_latents") != PREFIX_LATENTS or plan.get("window_count") != WINDOW_COUNT:
        raise ValueError("suite must contain 17 windows with a 12-latent overlap")
    if len(dataset.records) != len(plan["cases"]) * WINDOW_COUNT:
        raise ValueError("Expected 17 cached windows for every selected case")
    expected_rows = set(plan.get("rows", []))
    if {str(r["sample_id"]) for r in dataset.records} != expected_rows:
        raise ValueError("Cached rows do not match the evaluation plan")
    if int(plan.get("output_frames", 0)) != 1440:
        raise ValueError("full60s output must be cropped to 1440 frames")
    from cocode_world.proxy import compact_profile
    if plan.get("proxy_format") != "shared-scene-v1":
        raise ValueError("Unsupported proxy format")
    for filename, expected_hash in plan["checkpoint_payloads"].items():
        if state["payloads"][filename]["sha256"] != expected_hash:
            raise ValueError(f"Wrong released checkpoint payload: {filename}")
    for record in dataset.records:
        if record.get("proxy_reference_profile") != compact_profile() or record.get("proxy_format") != "shared-scene-v1":
            raise ValueError("Wrong proxy profile/version in long cache")
    return state


def _load_long_dataset(args):
    plan = json.loads(args.plan.read_text())
    dataset = CachedH3RefVideoDataset(args.manifest, allow_external_reference=True, **_hashes(args.model_root))
    state = _contract(args, dataset, plan)
    return dataset, state, plan


def _groups(dataset: CachedH3RefVideoDataset, window_count: int, case_id: str | None = None) -> dict[str, dict[int, int]]:
    groups: dict[str, dict[int, int]] = {}
    for index, record in enumerate(dataset.records):
        sid = str(record["sample_id"])
        if "--w" not in sid:
            raise ValueError(f"missing window suffix: {sid}")
        base, suffix = sid.rsplit("--w", 1)
        wi = int(suffix)
        if wi in groups.setdefault(base, {}):
            raise ValueError(f"duplicate window {sid}")
        groups[base][wi] = index
    expected = set(range(window_count))
    if any(set(v) != expected for v in groups.values()):
        raise ValueError("each case must have windows 0..16")
    if case_id is not None:
        if case_id not in groups:
            raise ValueError(f"unknown case {case_id}")
        return {case_id: groups[case_id]}
    return groups


def _assigned_groups(dataset, plan, args):
    groups = _groups(dataset, int(plan["window_count"]), args.case_id)
    if args.case_id is not None:
        return groups
    ordered = [c["sample_id"] for c in plan["cases"]][args.node::args.nodes]
    if args.stage in ("prefix", "render"):
        ordered = ordered[args.worker::args.workers]
    return {sid:groups[sid] for sid in ordered}


def _binding(args: argparse.Namespace) -> dict:
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_state_sha256": digest(args.checkpoint / "trainer_state.json"),
        "training_manifest_sha256": json.loads(args.plan.read_text())["training_manifest_sha256"],
        "manifest_sha256": digest(args.manifest),
        "plan_sha256": digest(args.plan),
        "num_forwards": args.num_forwards,
        "prefix_latents": args.prefix_latents,
        "geometry": "window124_overlap39_stride85_prefix12_full60s17",
        "proxy_format": "shared-scene-v1", "checkpoint_step": 4000, "window_count": WINDOW_COUNT,
        "implementation_sha256": digest(Path(__file__)),
    }


def _complete(case_dir: Path, index: int, binding: dict) -> bool:
    latent_path = case_dir / f"window{index}_latent.safetensors"
    marker = case_dir / f"window{index}_meta.json"
    if not latent_path.is_file() or not marker.is_file():
        return False
    meta = json.loads(marker.read_text())
    if meta.get("binding") != binding or meta.get("latent_sha256") != digest(latent_path):
        raise ValueError(f"stale/corrupt window artifact: {latent_path}")
    if index:
        prefix = case_dir / f"prefix{index}_latent.safetensors"
        image = case_dir / f"reference{index}_image_latent.safetensors"
        if not prefix.is_file() or not image.is_file():
            return False
        if meta.get("prefix_sha256") != digest(prefix) or meta.get("reference_image_latent_sha256") != digest(image):
            raise ValueError(f"stale continuation artifacts: {case_dir} window {index}")
    return True


def _save_window(case_dir: Path, index: int, latent: torch.Tensor, seed: int, seconds: float, binding: dict, sample_id: str, **extra) -> None:
    path = case_dir / f"window{index}_latent.safetensors"
    _atomic_tensor(path, "video_latent", latent.squeeze(0).to(dtype=torch.float32, device="cpu"))
    metadata = {"index": index, "seed": seed, "sample_id": sample_id, "denoise_seconds": seconds, "binding": binding, "latent_sha256": digest(path)}
    metadata.update(extra)
    write_json(case_dir / f"window{index}_meta.json", metadata)


def _case_noises(seed: int, total_latents: int, reference_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic global target and reference noise on CPU."""
    target_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    target = torch.randn((1, 24, total_latents, 46, 160), generator=target_generator, dtype=torch.float32)
    reference_generator = torch.Generator(device="cpu").manual_seed((int(seed) + 1) % (2**63 - 1))
    reference = torch.randn((1, reference_rows, 96), generator=reference_generator, dtype=torch.float32)
    return target, reference


@torch.inference_mode()
def _denoise_window(*, model: H3RefVideoTrainingModel, ema, sample, layout, seed: int, window_index: int,
                    num_forwards: int, device: torch.device, prefix: torch.Tensor | None,
                    image_latent: torch.Tensor | None) -> tuple[torch.Tensor, float, int]:
    total_latents = WINDOW_LATENTS + (WINDOW_COUNT - 1) * NEW_LATENTS
    target_shape = sample.clean_target_video_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
    proxy_latent = sample.clean_reference_video_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
    if image_latent is None:
        image = sample.clean_reference_image_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
    else:
        image = image_latent.to(device=device, dtype=torch.float32)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if tuple(image.shape) != (1, 24, 1, 46, 160):
            raise ValueError(f"continuation image latent shape {tuple(image.shape)}")
    _, clean_reference_rows = patchify_ref_video_latents(target_shape, proxy_latent, layout, reference_image_latent=image)
    global_target, global_reference = _case_noises(seed, total_latents, layout.num_condition_video_rows)
    start = window_index * NEW_LATENTS
    target_noise = global_target[:, :, start:start + WINDOW_LATENTS].to(device=device)
    reference_noise = global_reference.to(device=device)
    noised_reference_rows = model.reference_noise_aug * clean_reference_rows + (1.0 - model.reference_noise_aug) * reference_noise
    target_rows = patchify_video(target_noise, layout.patch_size)
    prefix_rows = 0
    prefix_clean_rows = None
    prefix_noise_rows = None
    if prefix is not None:
        prefix = prefix.to(device=device, dtype=torch.float32)
        if tuple(prefix.shape) != (1, 24, PREFIX_LATENTS, 46, 160):
            raise ValueError(f"continuation prefix shape {tuple(prefix.shape)}")
        prefix_clean_rows = patchify_video(prefix, layout.patch_size)
        prefix_noise_rows = patchify_video(global_target[:, :, start:start + PREFIX_LATENTS].to(device=device), layout.patch_size)
        prefix_rows = PREFIX_LATENTS * VIDEO_ROWS_PER_LATENT
        if prefix_clean_rows.shape[1] != prefix_rows or prefix_noise_rows.shape[1] != prefix_rows:
            raise ValueError("prefix row geometry mismatch")
    prompt_embeds = sample.encoder_hidden_states.unsqueeze(0).to(device=device, dtype=torch.float32)
    schedule = shifted_sigma_schedule(num_forwards, shift=12.0, device=device)
    started = time.perf_counter()
    with ema.apply() if ema is not None else nullcontext():
        for current_sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            timestep = 1.0 - current_sigma
            next_timestep = 1.0 - next_sigma
            if prefix_rows:
                target_rows[:, :prefix_rows] = timestep * prefix_clean_rows + (1.0 - timestep) * prefix_noise_rows
            velocity = model.predict_velocity(
                noisy_target_rows=target_rows,
                noised_reference_rows=noised_reference_rows,
                prompt_embeds=prompt_embeds,
                layout=layout,
                h3_timestep=timestep,
                fixed_target_prefix_rows=0,
            )
            target_rows = h3_euler_between(target_rows, velocity, timestep, next_timestep)
            if prefix_rows:
                target_rows[:, :prefix_rows] = next_timestep * prefix_clean_rows + (1.0 - next_timestep) * prefix_noise_rows
        if prefix_rows:
            target_rows[:, :prefix_rows] = prefix_clean_rows
    if not bool(torch.isfinite(target_rows).all()):
        raise FloatingPointError(f"non-finite result for {sample.metadata['sample_id']}")
    latent = unpatchify_video(target_rows, layout).float().cpu()
    if prefix is not None and not torch.equal(latent[:, :, :PREFIX_LATENTS], prefix.cpu()):
        raise AssertionError("final continuation prefix is not exactly the carried clean latent")
    return latent, time.perf_counter() - started, prefix_rows


def run_w0(args: argparse.Namespace) -> None:
    device, rank = _init_cp8(args)
    dataset, state, plan = _load_long_dataset(args)
    groups = _assigned_groups(dataset, plan, args)
    binding = _binding(args)
    transformer, model, ema = _load_model(args, state, device)
    for base, windows in sorted(groups.items()):
        case_dir = args.output_dir / base
        if rank == 0:
            case_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        if _complete(case_dir, 0, binding):
            dist.barrier()
            continue
        sample = dataset[windows[0]]
        seed = int(sample.metadata["eval_seeds"][0])
        layout = sample.build_layout(context_parallel_degree=8).to(device)
        latent, seconds, prefix_rows = _denoise_window(model=model, ema=ema, sample=sample, layout=layout, seed=seed,
                                                        window_index=0, num_forwards=args.num_forwards, device=device,
                                                        prefix=None, image_latent=None)
        dist.barrier()
        if rank == 0:
            _save_window(case_dir, 0, latent, seed, seconds, binding, sample.metadata["sample_id"], prefix_rows=prefix_rows,
                         noise_policy="global_target_seed_case; global_reference_seed_case_plus_one", latent_shape=list(latent.shape))
            print(f"w0 {base}: {seconds:.2f}s", flush=True)
        del sample, layout, latent
        gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    del model, transformer, ema
    gc.collect(); torch.cuda.empty_cache(); dist.barrier(); dist.destroy_process_group()


@torch.inference_mode()
def _boundary_image_latent(decoder, previous_latent: torch.Tensor) -> tuple[torch.Tensor, int]:
    if previous_latent.ndim == 4:
        previous_latent = previous_latent.unsqueeze(0)
    values = _pipeline_videos(decoder(latents=previous_latent.to(decoder._execution_device), output_type="np"))
    if values.shape != (1, WINDOW_FRAMES, 736, 2560, 3) or not np.isfinite(values).all():
        raise ValueError(f"boundary decode shape {values.shape}")
    # The next global window starts at frame 85 in the previous 124-frame
    # clip.  Only this one RGB frame is re-encoded; the 39-frame history itself
    # stays in latent space and is never decode/re-encoded.
    frame = (np.clip(values[0, STRIDE_FRAMES], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pixels = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous().to(decoder._execution_device)
    encoded = encode_vae_condition(decoder.vae, pixels, PIXEL_MEAN, PIXEL_STD, encode_seed=42)
    if tuple(encoded.shape) != (1, 24, 1, 46, 160) or not bool(torch.isfinite(encoded).all()):
        raise ValueError(f"boundary image encode shape {tuple(encoded.shape)}")
    return encoded.float().cpu().contiguous(), STRIDE_FRAMES


def run_prefix(args: argparse.Namespace) -> None:
    if args.window_index < 1:
        raise ValueError("prefix requires --window-index >= 1")
    torch.set_num_threads(4); torch.cuda.set_device(0); device = torch.device("cuda", 0)
    dataset, _state, plan = _load_long_dataset(args)
    groups = _assigned_groups(dataset, plan, args)
    binding = _binding(args)
    decoder = _load_decoder(args.model_root, device)
    try:
        for base in sorted(groups):
            case_dir = args.output_dir / base
            previous_path = case_dir / f"window{args.window_index - 1}_latent.safetensors"
            if not previous_path.is_file():
                raise FileNotFoundError(previous_path)
            prefix_path = case_dir / f"prefix{args.window_index}_latent.safetensors"
            image_path = case_dir / f"reference{args.window_index}_image_latent.safetensors"
            marker = case_dir / f"prefix{args.window_index}_meta.json"
            previous_sha = digest(previous_path)
            if prefix_path.is_file() and image_path.is_file() and marker.is_file():
                old = json.loads(marker.read_text())
                if old.get("binding") == binding and old.get("previous_latent_sha256") == previous_sha and old.get("prefix_sha256") == digest(prefix_path) and old.get("reference_image_latent_sha256") == digest(image_path):
                    print(f"prefix {args.window_index} {base}: resume", flush=True)
                    continue
                raise ValueError(f"stale prefix artifacts for {base} window {args.window_index}")
            previous = load_file(str(previous_path), device="cpu")["video_latent"].unsqueeze(0).float()
            prefix = previous[:, :, -PREFIX_LATENTS:].contiguous()
            image, boundary_frame = _boundary_image_latent(decoder, previous)
            _atomic_tensor(prefix_path, "prefix_latent", prefix.squeeze(0))
            _atomic_tensor(image_path, "reference_image_latent", image.squeeze(0))
            write_json(marker, {"binding": binding, "window_index": args.window_index, "prefix_latents": PREFIX_LATENTS,
                                "overlap_frames": OVERLAP_FRAMES, "previous_latent_sha256": previous_sha,
                                "prefix_sha256": digest(prefix_path), "reference_image_latent_sha256": digest(image_path),
                                "boundary_rgb_frame_in_previous_window": boundary_frame,
                                "prefix_source": "previous_window_latent_tail_direct",
                                "reference_image_source": "previous_window_boundary_rgb_single_frame"})
            print(f"prefix {args.window_index} {base}: latent_tail=12 boundary_frame={boundary_frame}", flush=True)
            del previous, prefix, image
            torch.cuda.empty_cache()
    finally:
        del decoder; gc.collect(); torch.cuda.empty_cache()


def run_continuation(args: argparse.Namespace) -> None:
    if args.window_index < 1:
        raise ValueError("continuation requires --window-index >= 1")
    device, rank = _init_cp8(args)
    dataset, state, plan = _load_long_dataset(args)
    groups = _assigned_groups(dataset, plan, args)
    binding = _binding(args)
    transformer, model, ema = _load_model(args, state, device)
    for base, windows in sorted(groups.items()):
        case_dir = args.output_dir / base
        if _complete(case_dir, args.window_index, binding):
            dist.barrier(); continue
        prefix_path = case_dir / f"prefix{args.window_index}_latent.safetensors"
        image_path = case_dir / f"reference{args.window_index}_image_latent.safetensors"
        if not prefix_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"missing prefix/image for {base} window {args.window_index}")
        prefix_meta = json.loads((case_dir / f"prefix{args.window_index}_meta.json").read_text())
        if prefix_meta.get("binding") != binding or digest(prefix_path) != prefix_meta.get("prefix_sha256") or digest(image_path) != prefix_meta.get("reference_image_latent_sha256"):
            raise ValueError("stale prefix/image artifact")
        previous_path = case_dir / f"window{args.window_index - 1}_latent.safetensors"
        if digest(previous_path) != prefix_meta.get("previous_latent_sha256"):
            raise ValueError("previous window changed after prefix stage")
        sample = dataset[windows[args.window_index]]
        seed = int(sample.metadata["eval_seeds"][0])
        prefix = load_file(str(prefix_path), device=str(device))["prefix_latent"].unsqueeze(0).float()
        image = load_file(str(image_path), device="cpu")["reference_image_latent"].unsqueeze(0).float()
        layout = sample.build_layout(context_parallel_degree=8).to(device)
        latent, seconds, prefix_rows = _denoise_window(model=model, ema=ema, sample=sample, layout=layout, seed=seed,
                                                        window_index=args.window_index, num_forwards=args.num_forwards,
                                                        device=device, prefix=prefix, image_latent=image)
        dist.barrier()
        if rank == 0:
            _save_window(case_dir, args.window_index, latent, seed, seconds, binding, sample.metadata["sample_id"],
                         prefix_rows=prefix_rows, prefix_sha256=digest(prefix_path), reference_image_latent_sha256=digest(image_path),
                         latent_shape=list(latent.shape), reference_image_source="previous_window_boundary_rgb_single_frame",
                         global_noise_latent_start=args.window_index * NEW_LATENTS,
                         global_noise_latent_stop=args.window_index * NEW_LATENTS + WINDOW_LATENTS)
            print(f"w{args.window_index} {base}: {seconds:.2f}s prefix_rows={prefix_rows}", flush=True)
        del sample, layout, latent, prefix, image
        gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    del model, transformer, ema
    gc.collect(); torch.cuda.empty_cache(); dist.barrier(); dist.destroy_process_group()


@torch.inference_mode()
def _iter_global_decode(decoder, latent: torch.Tensor):
    """Yield decoded RGB chunks while preserving the H3 VAE temporal stitch.

    ``AutoencoderKLMiniMaxH3.decode`` already chunks the temporal latent axis
    internally, but its implementation retains every decoded chunk and then
    concatenates them before converting to RGB.  A 56-second 2560x720 result
    therefore needs more than one H100 just for the temporary float32 video.
    This is the same temporal chunk/blend procedure, streamed one finalized
    chunk at a time so the peak memory stays bounded.
    """
    vae = decoder.vae
    if latent.shape[2] % vae.tokens_chunk_size != vae.token_overlap:
        raise ValueError("streaming export requires an H3 latent length 5*n+2")
    device = decoder._execution_device
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(decoder.pixel_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(decoder.pixel_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)

    z = latent.to(device=device, dtype=torch.float32) * latents_std + latents_mean
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)

    def postprocess(chunk: torch.Tensor) -> np.ndarray:
        video = (chunk.float() * pixel_std + pixel_mean).clamp(0, 1)
        # VideoProcessor is configured with do_normalize=False for H3, so this
        # only moves the short chunk to CPU and changes it to [T,H,W,C].
        return decoder.video_processor.postprocess_video(video, output_type="np")[0]

    overlap = None
    for i in range(num_chunks):
        finalized = None
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            start = i * tokens_chunk_size
            clip = vae._decode_clip(z[:, :, start : start + tokens_chunk_size + vae.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        previous_overlap = overlap
                        overlap = None
                        chunk = vae._blend(previous_overlap, chunk, vae.frame_overlap, dim=-3)
                        del previous_overlap
                    finalized = chunk
                else:
                    overlap = chunk
        # The reference decoder leaves the VAE autocast scope before applying
        # the ImageNet RGB affine transform.  Keep that ordering so streamed
        # output matches the stock decoder numerically.
        if finalized is not None:
            yield postprocess(finalized)
            del finalized
        del clip
    if overlap is not None:
        yield postprocess(overlap)
    del z, latents_mean, latents_std, pixel_mean, pixel_std


def _stream_global_decode_video(decoder, latent: torch.Tensor, path: Path, fps: int, raw_frames: int) -> int:
    """Decode and encode the long result without materializing all frames."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    produced = 0
    try:
        for values in _iter_global_decode(decoder, latent):
            if values.ndim != 4 or values.shape[1:] != (736, 2560, 3):
                raise ValueError(f"stream decode returned {values.shape}")
            if not np.isfinite(values).all():
                raise FloatingPointError("non-finite decoded RGB values")
            remaining = raw_frames - produced
            if remaining <= 0:
                break
            frames = (np.clip(values[:remaining, 8:728], 0, 1) * 255).round().astype(np.uint8)
            for frame in frames:
                writer.append_data(frame)
            produced += int(frames.shape[0])
            if produced % 170 == 0 or produced == raw_frames:
                print(f"render export: {produced}/{raw_frames} frames", flush=True)
            if produced == raw_frames:
                break
    finally:
        writer.close()
    if produced != raw_frames:
        raise ValueError(f"stream decode returned {produced} frames, expected {raw_frames}")
    return produced


def run_render(args: argparse.Namespace) -> None:
    torch.set_num_threads(4); torch.cuda.set_device(0); device = torch.device("cuda", 0)
    dataset, _state, plan = _load_long_dataset(args)
    groups = _assigned_groups(dataset, plan, args)
    binding = _binding(args)
    decoder = _load_decoder(args.model_root, device)
    try:
        for base, windows in sorted(groups.items()):
            case_dir = args.output_dir / base
            if (case_dir / "report.json").is_file():
                old = json.loads((case_dir / "report.json").read_text())
                if old.get("binding") == binding and Path(old.get("generated_video", "")).is_file() and digest(Path(old["generated_video"])) == old.get("generated_video_sha256"):
                    print(f"render {base}: resume", flush=True)
                    continue
            latents = []
            for index in range(int(plan["window_count"])):
                path = case_dir / f"window{index}_latent.safetensors"
                if not _complete(case_dir, index, binding):
                    raise ValueError(f"incomplete window {index}: {case_dir}")
                latents.append(load_file(str(path), device="cpu")["video_latent"].unsqueeze(0).float())
            assembled = latents[0]
            for index, latent in enumerate(latents[1:], 1):
                expected = assembled[:, :, index * NEW_LATENTS:index * NEW_LATENTS + PREFIX_LATENTS]
                if not torch.equal(latent[:, :, :PREFIX_LATENTS], expected):
                    raise AssertionError(f"latent overlap mismatch at window {index}")
                assembled = torch.cat((assembled, latent[:, :, PREFIX_LATENTS:]), dim=2)
            expected_latents = WINDOW_LATENTS + (int(plan["window_count"]) - 1) * NEW_LATENTS
            if assembled.shape[2] != expected_latents:
                raise AssertionError(f"assembled latent length {assembled.shape[2]} != {expected_latents}")
            print(f"render {base}: decoding assembled latent trajectory {tuple(assembled.shape)}", flush=True)
            raw_frames = WINDOW_FRAMES + (int(plan["window_count"]) - 1) * STRIDE_FRAMES
            generated_path = case_dir / "generated_long.mp4"
            temporary = generated_path.with_suffix(".partial.mp4")
            _stream_global_decode_video(decoder, assembled, temporary, FPS, int(plan["output_frames"]))
            temporary.replace(generated_path)
            prompt_lines = []
            for index in range(int(plan["window_count"])):
                record = dataset.records[windows[index]]
                prompt_lines += [f"[window {index}]", record["scene_description"], ""]
            (case_dir / "prompt_windows.txt").write_text("\n".join(prompt_lines), encoding="utf-8")
            report = {
                "schema_version": "cocode-world-long-video-v1-result",
                "sample_id": base,
                "status": "PASS",
                "binding": binding,
                "checkpoint": str(args.checkpoint.resolve()),
                "window_count": int(plan["window_count"]),
                "window_frames": WINDOW_FRAMES,
                "overlap_frames": OVERLAP_FRAMES,
                "stride_frames": STRIDE_FRAMES,
                "prefix_latents": PREFIX_LATENTS,
                "assembled_latents": int(assembled.shape[2]),
                "raw_output_frames": raw_frames,
                "output_frames": int(plan["output_frames"]),
                "output_seconds": int(plan["output_frames"]) / FPS,
                "num_forwards": args.num_forwards,
                "video_shift": 12.0,
                "generated_video": str(generated_path.resolve()),
                "generated_video_sha256": digest(generated_path),
                "caption_provenance": "fresh Qwen v1 reviewed RGB timeline; appearance-only style correction; clipped to exact stride85 interval",
                "proxy_format": "shared-scene-v1", "checkpoint_step": 4000, "fps": 24, "width": 2560, "height": 720,
                "seed": int(dataset.records[windows[0]]["eval_seeds"][0]),
                "continuation_handoff": "VDN latent prefix projection at every flow step; global noise slice; direct latent assembly",
                "continuation_reference_image": "one boundary RGB frame from previous window encoded as the current image reference",
                "decode_policy": "one assembled latent trajectory; native VAE temporal chunks and blending streamed to MP4",
            }
            write_json(case_dir / "report.json", report)
            del assembled, latents
            torch.cuda.empty_cache()
    finally:
        del decoder; gc.collect(); torch.cuda.empty_cache()
    if not args.case_id:
        write_json(args.output_dir / f"render_node{args.node}_worker{args.worker}.json", {"status": "PASS", "case_count": len(groups), "window_count": int(plan["window_count"]), "geometry": "124/39/85/12"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("w0", "prefix", "continuation", "render"), required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--case-id")
    parser.add_argument("--node",type=int,default=0)
    parser.add_argument("--nodes",type=int,default=1)
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models/MiniMax-H3"))
    parser.add_argument("--num-forwards", type=int, default=30)
    parser.add_argument("--prefix-latents", type=int, default=PREFIX_LATENTS)
    args = parser.parse_args()
    if args.num_forwards < 2 or args.prefix_latents != PREFIX_LATENTS:
        raise ValueError("Use at least 2 denoising forwards and exactly 12 prefix latents")
    if args.window_index < 0 or args.window_index >= WINDOW_COUNT:
        raise ValueError("window-index must be in [0,16]")
    {"w0": run_w0, "prefix": run_prefix, "continuation": run_continuation, "render": run_render}[args.stage](args)


if __name__ == "__main__":
    main()
