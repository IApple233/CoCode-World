from __future__ import annotations
import argparse
from datetime import timedelta
from pathlib import Path
import os
import torch
import torch.distributed as dist
from peft import LoraConfig
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models._modeling_parallel import ContextParallelConfig
from infer_h3_ref_video_lora import _component_hash, ALL_BLOCK_LORA_PATTERN, MAIN_QKVO_PATTERN
from utility.h3_ref_video import H3RefVideoTrainingModel
from utility.h3_distill import TrainableEMA
from utility.h3_ref_video_checkpoint import load_ref_video_lora_checkpoint


def _hashes(model_root: Path) -> dict[str, str]:
    return {
        "expected_transformer_ref_hash": _component_hash(
            model_root, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json"
        ),
        "expected_vae_hash": _component_hash(
            model_root, "vae", "diffusion_pytorch_model.safetensors.index.json"
        ),
        "expected_text_encoder_hash": _component_hash(
            model_root, "text_encoder", "model.safetensors.index.json"
        ),
    }

def _load_model(args: argparse.Namespace, state: dict, device: torch.device):
    contract = state["metadata"]
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model_root / "transformer_ref", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(contract["lora_rank"]),
            lora_alpha=float(contract["lora_alpha"]),
            lora_dropout=0.0,
            target_modules=MAIN_QKVO_PATTERN
            if contract["lora_targets"] == "qkvo"
            else ALL_BLOCK_LORA_PATTERN,
        ),
        adapter_name="ref_video",
    )
    transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=8))
    transformer.eval()
    model = H3RefVideoTrainingModel(transformer, reference_noise_aug=0.999).eval()
    ema = TrainableEMA(model, decay=float(contract["ema_decay"]))
    load_ref_video_lora_checkpoint(args.checkpoint, model=model, ema=ema, optimizer=None, restore_rng=False)
    return transformer, model, ema

def _init_cp8(args: argparse.Namespace) -> tuple[torch.device, int]:
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=4))
    if dist.get_world_size() != 8:
        raise ValueError("The staged rollout requires one CP8 process group")
    return device, rank
