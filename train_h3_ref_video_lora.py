"""Train split-screen MiniMax-H3 Ref2VA LoRA with frozen image/video references."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from peft import LoraConfig
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "verification", ROOT / "diffusers" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from local_diffusers import use_local_diffusers


use_local_diffusers()

from hf_hub_compat import *  # noqa: F401,F403
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models._modeling_parallel import ContextParallelConfig

from utility.h3_distill import TrainableEMA, shift_noise_ratio, sum_context_parallel_gradients
from utility.h3_distill.data import sha256_file
from utility.h3_ref_video import H3RefVideoTrainingModel, patchify_ref_video_latents
from utility.h3_ref_video_checkpoint import (
    load_ref_video_lora_checkpoint,
    save_ref_video_lora_checkpoint,
)
from utility.h3_ref_video_data import CachedH3RefVideoDataset, REF_VIDEO_CACHE_SCHEMA_VERSION
from utility.h3_decfg_adapter import inspect_decfg_adapter, load_decfg_training_adapter, validate_scale
from utility.h3_training_schedule import BufferedEpochSampler, SAMPLER_REVISION, learning_rate_for_step


DEFAULT_MODEL_ROOT = Path("models/MiniMax-H3")
TEXT_ENCODER_EXECUTION_REVISION = "qwen3vl-hidden50-hook-layer49-inference-retain-0-through-49-v2"
MAIN_QKVO_PATTERN = r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)"
ALL_BLOCK_LORA_PATTERN = (
    r"(transformer_blocks\.\d+|token_refiner\.refiner_blocks\.\d+)\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--context-parallel-degree", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument(
        "--lora-targets",
        choices=("qkvo", "qkvo-ffn"),
        default="qkvo-ffn",
        help="qkvo-ffn covers attention and feed-forward linears in all 50 main and 2 text-refiner blocks.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Linear optimizer-step warmup, followed by constant learning rate; 0 disables it.")
    parser.add_argument("--shuffle-buffer-size", type=int, default=32768,
                        help="Per-epoch shuffled index buffer; clamped to dataset size. 0 uses sequential order.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--reference-noise-aug", type=float, default=0.999)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument(
        "--decfg-adapter", "--preset-lora-path", "--preset_lora_path", type=Path,
        help="Released official Ref2VA DeCFG adapter, frozen and fused before task LoRA training; omitted at inference.",
    )
    parser.add_argument("--decfg-adapter-scale", type=float, default=1.0,
                        help="Strength of the frozen DeCFG preset (official default 1).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Fixed training seed for initialization and per-step noise (default: 42).",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--verify-data-sha256", action="store_true")
    parser.add_argument("--trust-validated-cache", action="store_true")
    parser.add_argument(
        "--activation-cpu-offload",
        action="store_true",
        help="Move autograd-saved CUDA tensors to pinned CPU memory until backward uses them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_scale(args.decfg_adapter_scale)
    if args.decfg_adapter is None and args.decfg_adapter_scale != 1.0:
        raise ValueError("--decfg-adapter-scale requires --decfg-adapter")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    positive = {
        "steps": args.steps,
        "save_every": args.save_every,
        "gradient_accumulation": args.gradient_accumulation,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "max_gradient_norm": args.max_gradient_norm,
        "video_shift": args.video_shift,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if args.warmup_steps < 0 or args.shuffle_buffer_size < 0:
        raise ValueError("warmup_steps and shuffle_buffer_size cannot be negative")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0,1)")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.reference_noise_aug <= 1:
        raise ValueError("EMA decay/reference noise augmentation is outside its valid range")


def model_component_hash(
    model_root: Path,
    component: str,
    index_name: str = "diffusion_pytorch_model.safetensors.index.json",
) -> str:
    digest = hashlib.sha256()
    root = model_root / component
    for name in ("config.json", index_name):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ParallelTopology:
    global_mesh: Any
    context_parallel_group: dist.ProcessGroup
    data_parallel_group: dist.ProcessGroup
    context_parallel_degree: int
    data_parallel_degree: int


def initialize_parallel_topology(
    *, global_world_size: int, local_world_size: int, context_parallel_degree: int = 4
) -> ParallelTopology:
    """Keep each CP group inside a node and average gradients across DP replicas."""
    if context_parallel_degree not in (1, 2, 4, 8):
        raise ValueError("CP degree must be 1, 2, 4 or 8")
    if local_world_size <= 0 or local_world_size % context_parallel_degree:
        raise ValueError(f"Local GPU count {local_world_size} must be divisible by CP={context_parallel_degree}")
    if global_world_size % local_world_size:
        raise ValueError("global world size must be divisible by the GPUs per node")
    data_parallel_degree = global_world_size // context_parallel_degree
    mesh = init_device_mesh(
        "cuda",
        (data_parallel_degree, 1, context_parallel_degree),
        mesh_dim_names=("data", "ring", "ulysses"),
    )
    context_mesh = mesh["ring", "ulysses"]
    data_mesh = mesh["data"]
    degree = int(context_mesh.size(1))
    if degree != context_parallel_degree:
        raise AssertionError(f"Expected CP={context_parallel_degree}, got CP={degree}")
    return ParallelTopology(
        global_mesh=mesh,
        context_parallel_group=context_mesh.get_group("ulysses"),
        data_parallel_group=data_mesh.get_group(),
        context_parallel_degree=degree,
        data_parallel_degree=data_parallel_degree,
    )


@torch.no_grad()
def average_data_parallel_gradients(
    parameters: list[torch.nn.Parameter], group: dist.ProcessGroup, degree: int
) -> None:
    if degree == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
            parameter.grad.div_(degree)


def sample_h3_timestep(*, generator: torch.Generator, shift: float, device: torch.device) -> torch.Tensor:
    noise_ratio = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    return 1.0 - shift_noise_ratio(noise_ratio, shift=shift)


def checkpoint_contract(
    args: argparse.Namespace,
    *,
    manifest_sha256: str,
    transformer_ref_hash: str,
    dataset_record: dict[str, Any],
    context_parallel_degree: int,
    data_parallel_degree: int,
    decfg_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "model_root": str(args.model_root.resolve()),
        "transformer_partition": "transformer_ref",
        "transformer_ref_config_hash": transformer_ref_hash,
        "dataset_manifest": str(args.manifest.resolve()),
        "dataset_manifest_sha256": manifest_sha256,
        "data_schema": REF_VIDEO_CACHE_SCHEMA_VERSION,
        "dataset_vae_config_hash": dataset_record["vae_config_hash"],
        "dataset_text_encoder_hash": dataset_record["text_encoder_hash"],
        "dataset_transformer_ref_config_hash": dataset_record["transformer_ref_config_hash"],
        "text_encoder_layer": 50,
        "text_encoder_execution_revision": TEXT_ENCODER_EXECUTION_REVISION,
        "reference_kinds": ["image", "video"],
        "reference_order": "target_first_frame_then_proxy_video",
        "reference_image_frozen": True,
        "reference_audio": "disabled",
        "reference_noise_aug": args.reference_noise_aug,
        "context_parallel_degree": context_parallel_degree,
        "data_parallel_degree": data_parallel_degree,
        "global_world_size": context_parallel_degree * data_parallel_degree,
        "gradient_accumulation": args.gradient_accumulation,
        "activation_cpu_offload": bool(args.activation_cpu_offload),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": args.lora_targets,
        "objective": "target_rows_h3_flow_velocity_mse",
        "time_distribution": "uniform_noise_ratio_then_shift",
        "video_shift": args.video_shift,
        "learning_rate": args.learning_rate,
        "lr_schedule": "linear_warmup_then_constant" if args.warmup_steps else "constant",
        "warmup_steps": args.warmup_steps,
        "sample_order_revision": SAMPLER_REVISION if args.shuffle_buffer_size else "sequential-v1",
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "epoch_tail_policy": "continuous_global_sample_stream",
        "weight_decay": args.weight_decay,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "ema_decay": args.ema_decay,
        "max_gradient_norm": args.max_gradient_norm,
        "attention": "full_bidirectional",
        "seed": args.seed,
        "audio": "disabled",
        "depth_control": "removed",
        "target_canvas": [int(dataset_record["height"]), int(dataset_record["width"])],
        "split_screen": "agent_000_left_agent_001_right",
    }
    if decfg_contract is not None:
        contract.update(decfg_contract)
    return contract


def validate_resume_contract(checkpoint: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    mismatches = {
        key: {"checkpoint": metadata.get(key), "requested": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    # Old checkpoints have no preset. A strict optimizer resume cannot silently
    # enable, remove or switch this training-only base. Evaluation is separate.
    for key in set(metadata) | set(expected):
        if key.startswith("decfg_adapter_") and metadata.get(key) != expected.get(key):
            mismatches[key] = {"checkpoint": metadata.get(key), "requested": expected.get(key)}
    if mismatches:
        raise ValueError(f"Resume checkpoint contract mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return manifest


def _broadcast_rank_zero_error(error: str | None, *, device: torch.device) -> None:
    values = [error]
    dist.broadcast_object_list(values, src=0, device=device)
    if values[0] is not None:
        raise RuntimeError(str(values[0]))


def main() -> int:
    args = parse_args()
    validate_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout_seconds = int(os.environ.get("H3_PROCESS_GROUP_TIMEOUT_SECONDS", "3600"))
    if timeout_seconds < 600:
        raise ValueError("H3_PROCESS_GROUP_TIMEOUT_SECONDS must be at least 600")
    dist.init_process_group(
        "nccl",
        device_id=device,
        timeout=timedelta(seconds=timeout_seconds),
    )
    rank, global_world_size = dist.get_rank(), dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", global_world_size))
    topology = initialize_parallel_topology(
        global_world_size=global_world_size,
        local_world_size=local_world_size,
        context_parallel_degree=args.context_parallel_degree,
    )
    cp_degree = topology.context_parallel_degree
    dp_degree = topology.data_parallel_degree
    dp_rank = dist.get_rank(topology.data_parallel_group)
    # Allocate NCCL subgroup collectives before the long-sequence allocator
    # pool fills the device. A lazy NVLS/all-reduce allocation after backward
    # cannot reclaim memory reserved by PyTorch.
    collective_probe = torch.ones(1, device=device, dtype=torch.float32)
    for group, degree in ((topology.context_parallel_group, cp_degree),
                          (topology.data_parallel_group, dp_degree)):
        collective_probe.fill_(1)
        dist.all_reduce(collective_probe, group=group)
        if collective_probe.item() != degree:
            raise RuntimeError("Parallel subgroup warmup returned the wrong size")
    del collective_probe
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if rank == 0:
        print(f"[h3-ref-video-lora] seed={args.seed} DP={dp_degree} CP={cp_degree} "
              f"effective_batch={dp_degree * args.gradient_accumulation}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_root = args.model_root.resolve()
    decfg_contract = inspect_decfg_adapter(args.decfg_adapter, scale=args.decfg_adapter_scale) if args.decfg_adapter else None
    transformer_hash = model_component_hash(model_root, "transformer_ref")
    vae_hash = model_component_hash(model_root, "vae")
    text_encoder_hash = model_component_hash(
        model_root,
        "text_encoder",
        "model.safetensors.index.json",
    )
    dataset = CachedH3RefVideoDataset(
        args.manifest,
        expected_vae_hash=vae_hash,
        expected_text_encoder_hash=text_encoder_hash,
        expected_transformer_ref_hash=transformer_hash,
        verify_sha256=args.verify_data_sha256,
        trust_validated_paths=args.trust_validated_cache,
    )
    sampler = BufferedEpochSampler(len(dataset), seed=args.seed, buffer_size=args.shuffle_buffer_size)
    # Fail before model loading if any cached sequence cannot be divided across
    # the requested context-parallel ranks.
    for record in dataset.records:
        if int(record["num_packed_rows"]) % cp_degree:
            raise ValueError(
                f"{record['sample_id']} has {record['num_packed_rows']} packed rows, incompatible with CP={cp_degree}"
            )
    manifest_hash = sha256_file(args.manifest)
    contract = checkpoint_contract(
        args,
        manifest_sha256=manifest_hash,
        transformer_ref_hash=transformer_hash,
        dataset_record=dataset.records[0],
        context_parallel_degree=cp_degree,
        data_parallel_degree=dp_degree,
        decfg_contract=decfg_contract,
    )
    if args.resume is not None:
        validate_resume_contract(args.resume, contract)

    output_error = None
    if rank == 0:
        try:
            if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise FileExistsError(f"Refusing to reuse non-empty output directory {args.output_dir}")
        except BaseException as error:
            output_error = f"Output preflight failed: {error!r}"
    _broadcast_rank_zero_error(output_error, device=device)

    transformer_ref = MiniMaxH3Transformer3DModel.from_pretrained(
        model_root / "transformer_ref",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    transformer_ref.requires_grad_(False)
    if args.decfg_adapter is not None:
        report = load_decfg_training_adapter(
            transformer_ref, args.decfg_adapter, scale=args.decfg_adapter_scale, expected_contract=decfg_contract,
        )
        if rank == 0:
            print("[h3-ref-video-lora] released DeCFG preset fused: " + json.dumps(report, sort_keys=True), flush=True)
    target_pattern = MAIN_QKVO_PATTERN if args.lora_targets == "qkvo" else ALL_BLOCK_LORA_PATTERN
    transformer_ref.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_pattern,
            init_lora_weights=True,
        ),
        adapter_name="ref_video",
    )
    trainable = [parameter for parameter in transformer_ref.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in transformer_ref.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable_names):
        raise AssertionError("Exactly the Ref2VA LoRA parameters must be trainable")
    for parameter in trainable:
        dist.broadcast(parameter.data, src=0)
    transformer_ref.enable_gradient_checkpointing()
    if cp_degree > 1:
        transformer_ref.enable_parallelism(
            config=ContextParallelConfig(ulysses_degree=cp_degree, mesh=topology.global_mesh)
        )
    transformer_ref.train()
    training_model = H3RefVideoTrainingModel(
        transformer_ref,
        reference_noise_aug=args.reference_noise_aug,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        foreach=False,
    )
    ema = TrainableEMA(training_model, decay=args.ema_decay)
    start_step = 0
    if args.resume is not None:
        validate_resume_contract(args.resume, contract)
        restored = load_ref_video_lora_checkpoint(
            args.resume,
            model=training_model,
            ema=ema,
            optimizer=optimizer,
        )
        start_step = int(restored["step"])

    publish_error = None
    if rank == 0:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "run_contract.json").write_text(
                json.dumps(
                    {
                        **contract,
                        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                        "trainable_tensors": len(trainable),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except BaseException as error:
            publish_error = f"Rank 0 failed to publish run contract: {error!r}"
    _broadcast_rank_zero_error(publish_error, device=device)
    dist.barrier()
    metrics_path = args.output_dir / "train_metrics.jsonl"
    progress = (
        tqdm(
            total=args.steps - start_step,
            desc=f"train steps {start_step + 1}-{args.steps}",
            unit="step",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        if rank == 0
        else None
    )

    try:
        for step in range(start_step, args.steps):
            learning_rate = learning_rate_for_step(args.learning_rate, step, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            sample_indices = sampler.step_indices(step, replicas=dp_degree, accumulation=args.gradient_accumulation)
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_started = time.perf_counter()
            sample_ids: list[str] = []
            torch.cuda.reset_peak_memory_stats(device)
            for microbatch in range(args.gradient_accumulation):
                global_microbatch = step * args.gradient_accumulation + microbatch
                dataset_index = sample_indices[dp_rank][microbatch]
                sample = dataset[dataset_index]
                layout = sample.build_layout(context_parallel_degree=cp_degree).to(device)
                target_latent = sample.clean_target_video_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
                reference_image_latent = sample.clean_reference_image_latent.unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                reference_video_latent = sample.clean_reference_video_latent.unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                clean_target_rows, clean_reference_rows = patchify_ref_video_latents(
                    target_latent,
                    reference_video_latent,
                    layout,
                    reference_image_latent=reference_image_latent,
                )
                prompt = sample.encoder_hidden_states.unsqueeze(0).to(device=device, dtype=torch.float32)
                seed = args.seed + global_microbatch * 3 + dp_rank * 1_000_003
                generator = torch.Generator(device=device).manual_seed(seed)
                target_noise = torch.randn(
                    clean_target_rows.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                reference_noise = torch.randn(
                    clean_reference_rows.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                timestep = sample_h3_timestep(generator=generator, shift=args.video_shift, device=device)
                saved_tensor_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if args.activation_cpu_offload
                    else nullcontext()
                )
                with saved_tensor_context:
                    output = training_model.flow_matching_loss(
                        clean_target_rows=clean_target_rows,
                        target_noise=target_noise,
                        clean_reference_rows=clean_reference_rows,
                        reference_noise=reference_noise,
                        prompt_embeds=prompt,
                        layout=layout,
                        h3_timestep=timestep,
                    )
                if not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError(f"Non-finite Ref2VA loss at step {step + 1}")
                (output.loss / args.gradient_accumulation).backward()
                total_loss += output.loss.detach().float() / args.gradient_accumulation
                sample_ids.append(str(sample.metadata["sample_id"]))

            sum_context_parallel_gradients(trainable, topology.context_parallel_group)
            average_data_parallel_gradients(trainable, topology.data_parallel_group, dp_degree)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_gradient_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Non-finite LoRA gradient norm at step {step + 1}")
            optimizer.step()
            ema.update()
            local_finite = torch.tensor(
                int(all(bool(torch.isfinite(parameter).all()) for parameter in trainable)),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
            if not bool(local_finite.item()):
                raise FloatingPointError(f"Non-finite LoRA weights at step {step + 1}")
            torch.cuda.synchronize(device)
            resources = torch.tensor(
                [
                    float(total_loss),
                    float(grad_norm),
                    time.perf_counter() - step_started,
                    torch.cuda.max_memory_allocated(device),
                ],
                dtype=torch.float64,
                device=device,
            )
            gathered = [torch.empty_like(resources) for _ in range(global_world_size)]
            dist.all_gather(gathered, resources)
            values = torch.stack(gathered).cpu()
            for replica in range(dp_degree):
                replica_values = values[replica * cp_degree : (replica + 1) * cp_degree]
                if float(replica_values[:, 0].max() - replica_values[:, 0].min()) > 1e-5:
                    raise RuntimeError("CP ranks within one replica produced different losses")
            if rank == 0:
                all_sample_ids = [
                    [str(dataset.records[index]["sample_id"]) for index in indices]
                    for indices in sample_indices
                ]
                samples_seen = (step + 1) * args.gradient_accumulation * dp_degree
                metrics = {
                    "step": step + 1,
                    "learning_rate": learning_rate,
                    "epoch": (step * args.gradient_accumulation * dp_degree) // len(dataset) + 1,
                    "epoch_end": (samples_seen - 1) // len(dataset) + 1,
                    "epochs_completed": samples_seen / len(dataset),
                    "global_samples_seen": samples_seen,
                    "sample_ids_rank0": sample_ids,
                    "data_parallel_degree": dp_degree,
                    "context_parallel_degree": cp_degree,
                    "effective_batch_size": dp_degree * args.gradient_accumulation,
                    "sample_ids_by_replica": all_sample_ids,
                    "loss": float(values[:, 0].mean()),
                    "gradient_norm": float(values[:, 1].mean()),
                    "step_seconds_max": float(values[:, 2].max()),
                    "peak_memory_allocated_bytes_per_rank": [int(value) for value in values[:, 3]],
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    lr=f"{learning_rate:.2e}",
                    epoch=metrics["epoch"],
                    grad=f"{metrics['gradient_norm']:.4g}",
                    refresh=False,
                )
                progress.update(1)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                dist.barrier()
                if rank == 0:
                    save_ref_video_lora_checkpoint(
                        args.output_dir / f"step-{step + 1:06d}",
                        model=training_model,
                        ema=ema,
                        optimizer=optimizer,
                        step=step + 1,
                        metadata={
                            **contract,
                            "global_samples_seen": (step + 1) * args.gradient_accumulation * dp_degree,
                            "dataset_epoch": ((step + 1) * args.gradient_accumulation * dp_degree) // len(dataset),
                            "dataset_cursor": (
                                (step + 1) * args.gradient_accumulation * dp_degree
                            )
                            % len(dataset),
                        },
                    )
                dist.barrier()
    finally:
        if progress is not None:
            progress.close()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
