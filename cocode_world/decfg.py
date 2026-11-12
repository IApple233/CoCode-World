"""Load the released Ref2VA DeCFG preset as frozen training-only weights.

Source: DiffSynth-Studio/MiniMax-H3-TrainingAdapter, model_ref2va.safetensors.
The preset is merged before adding the task LoRA, like DiffSynth's
--preset_lora_path. Checkpoints contain only the learned task LoRA; evaluation
loads the original H3 base and deliberately does not load this preset.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_ADAPTER_PATH = (
    Path(__file__).resolve().parents[1] / "models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_ref2va.safetensors"
)
RELEASE_SHA256 = "9361683349c1bf78f206c382d91b1e01b91fcddc2408e85ab39c9faa51f0d3d1"
RELEASE_REVISION = "bdddb151d0dfb62b1dfa9e2f8ae6a6e1b179e098"
RELEASE_MODEL_ID = "DiffSynth-Studio/MiniMax-H3-TrainingAdapter"
CONVERSION_REVISION = "official-h3-head-role-dim-qkv-split-swiglu-swap-v1"
FUSION_REVISION = "bf16-matmul-add-like-diffsynth-preset-v1"
USAGE = "frozen_training_preset_excluded_from_task_lora_and_inference"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_scale(scale: float) -> None:
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("decfg_adapter_scale must be finite and positive; omit the adapter path to disable it")


def inspect_decfg_adapter(path: Path, *, scale: float = 1.0) -> dict:
    """Bind the exact public Ref2VA release, rejecting FL2VA/Comfy variants."""
    validate_scale(scale)
    path = Path(path).resolve()
    digest = file_sha256(path)
    if digest != RELEASE_SHA256:
        raise ValueError(
            "DeCFG adapter SHA256 does not match the released official Ref2VA weights. "
            "Use model_ref2va.safetensors, not the FL2VA or Comfy-layout variant."
        )
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected = source_modules({"num_layers": 50, "num_refiner_layers": 2})
        wanted = {f"{name}.lora_{part}.default.weight" for name in expected for part in ("A", "B")}
        if keys != wanted:
            raise ValueError("Unexpected released DeCFG adapter tensor set")
    return {
        "decfg_adapter_path": str(path), "decfg_adapter_sha256": digest,
        "decfg_adapter_scale": float(scale), "decfg_adapter_model_id": RELEASE_MODEL_ID,
        "decfg_adapter_release_revision": RELEASE_REVISION,
        "decfg_adapter_conversion_revision": CONVERSION_REVISION,
        "decfg_adapter_fusion_revision": FUSION_REVISION,
        "decfg_adapter_usage": USAGE,
    }


def source_modules(config) -> list[str]:
    return [f"{prefix}.{index}.{layer}"
            for prefix, count in (("blocks", config["num_layers"]), ("token_refiner.blocks", config["num_refiner_layers"]))
            for index in range(count)
            for layer in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]


def convert_pair(name: str, a: torch.Tensor, b: torch.Tensor, config):
    """Convert LoRA factors without materializing a full 33B weight delta.

    The official adapter stores B's QKV rows as [head, role, dim]. Diffusers
    uses separate Q/K/V matrices. FFN gate/value halves also reverse. A has
    input coordinates only and is not permuted. Scaling is alpha/rank=1 for
    this published DiffSynth LoRA, independent of the task LoRA's own rank.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or a.shape[0] != b.shape[1]:
        raise ValueError(f"Invalid DeCFG A/B shapes: {name}")
    if not a.is_floating_point() or not b.is_floating_point() or a.dtype != b.dtype:
        raise ValueError(f"Invalid DeCFG A/B dtype: {name}")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError(f"Nonfinite DeCFG A/B: {name}")
    target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1) if name.startswith("token_refiner.blocks.") else name.replace("blocks.", "transformer_blocks.", 1)
    heads, dim, hidden, ffn = (config[key] for key in ("num_attention_heads", "attention_head_dim", "hidden_size", "ffn_dim"))
    inner = heads * dim
    if name.endswith(".attn.qkv_proj"):
        expected = (3 * inner, hidden)
        if (b.shape[0], a.shape[1]) != expected:
            raise ValueError(f"DeCFG QKV dimensions do not match H3: {name}")
        grouped = b.reshape(heads, 3, dim, b.shape[1])
        return [(target.removesuffix("qkv_proj") + f"to_{role}", a, grouped[:, index].reshape(inner, b.shape[1]).contiguous())
                for index, role in enumerate("qkv")]
    if name.endswith(".attn.out_proj"):
        expected, target = (hidden, inner), target.removesuffix("out_proj") + "to_out.0"
    elif name.endswith(".mlp.fc1"):
        expected, target = (2 * ffn, hidden), target.removesuffix("mlp.fc1") + "ff.net.0.proj"
        if b.shape[0] != 2 * ffn:
            raise ValueError(f"DeCFG FFN dimensions do not match H3: {name}")
        gate, value = b.chunk(2, dim=0)
        b = torch.cat((value, gate)).contiguous()
    elif name.endswith(".mlp.fc2"):
        expected, target = (hidden, ffn), target.removesuffix("mlp.fc2") + "ff.net.2"
    else:
        raise ValueError(f"Unsupported DeCFG target: {name}")
    if (b.shape[0], a.shape[1]) != expected:
        raise ValueError(f"DeCFG dimensions do not match H3: {name}")
    return [(target, a, b)]


@torch.no_grad()
def fuse_pairs(model, pairs, *, scale: float = 1.0, chunk_rows: int = 1024) -> int:
    """One-time bounded-memory fusion, before creating the trainable task LoRA."""
    validate_scale(scale)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    modules = dict(model.named_modules())
    seen = set()
    for name, a, b in pairs:
        if name in seen:
            raise ValueError(f"Duplicate converted DeCFG target: {name}")
        seen.add(name)
        layer = modules.get(name)
        if not isinstance(layer, torch.nn.Linear) or hasattr(layer, "lora_A"):
            raise ValueError(f"DeCFG must be fused into a plain Linear before task LoRA creation: {name}")
        weight = layer.weight
        if weight.requires_grad or weight.is_meta:
            raise ValueError(f"DeCFG base must be materialized and frozen: {name}")
        if weight.shape != (b.shape[0], a.shape[1]):
            raise ValueError(f"DeCFG target weight shape mismatch: {name}")
        a = a.to(device=weight.device, dtype=weight.dtype)
        for start in range(0, b.shape[0], chunk_rows):
            stop = min(start + chunk_rows, b.shape[0])
            part = b[start:stop].to(device=weight.device, dtype=weight.dtype)
            # Match DiffSynth's preset fusion arithmetic in the base dtype.
            delta = (part @ a) * scale
            merged = weight[start:stop] + delta
            if not bool(torch.isfinite(merged).all()):
                raise FloatingPointError(f"Nonfinite fused DeCFG weight: {name}")
            weight[start:stop].copy_(merged)
    return len(seen)


def load_decfg_training_adapter(model, path: Path, *, scale: float = 1.0, expected_contract: dict | None = None) -> dict:
    """Fuse the checked official release once; never called by evaluation."""
    if getattr(model, "_h3_decfg_training_adapter", None) is not None:
        raise ValueError("DeCFG training adapter has already been fused into this model")
    contract = inspect_decfg_adapter(path, scale=scale)
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("DeCFG adapter changed after training preflight")
    count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in source_modules(model.config):
            a = handle.get_tensor(f"{name}.lora_A.default.weight")
            b = handle.get_tensor(f"{name}.lora_B.default.weight")
            count += fuse_pairs(model, convert_pair(name, a, b, model.config), scale=scale)
    if count != 6 * (model.config.num_layers + model.config.num_refiner_layers):
        raise ValueError("Incomplete DeCFG fusion")
    report = {**contract, "fused_linear_layers": count, "source_lora_rank": 64}
    model._h3_decfg_training_adapter = report
    return report
