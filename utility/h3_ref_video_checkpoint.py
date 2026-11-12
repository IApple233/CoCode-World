"""Atomic LoRA checkpoints for MiniMax-H3 proxy-video Ref2VA training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import torch
from safetensors.torch import load_file, save_file

from utility.h3_distill import TrainableEMA


CHECKPOINT_VERSION = 1
PAYLOAD_FILES = ("lora.safetensors", "ema.safetensors", "optimizer.pt", "rng_state.pt")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _trainable_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not parameters:
        raise ValueError("The Ref2VA transformer has no trainable LoRA parameters")
    invalid = [name for name in parameters if "lora_" not in name]
    if invalid:
        raise ValueError(f"Non-LoRA parameters unexpectedly require gradients: {invalid[:8]}")
    return parameters


def _name_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def save_ref_video_lora_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    ema: TrainableEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    """Atomically publish LoRA, EMA, optimizer, RNG, and their hashes."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{directory.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        parameters = _trainable_parameters(model)
        state = {name: value.detach().cpu().contiguous() for name, value in parameters.items()}
        save_file(state, temporary / "lora.safetensors")
        save_file(ema.state_dict(), temporary / "ema.safetensors")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        rng_state: dict[str, Any] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        torch.save(rng_state, temporary / "rng_state.pt")
        payloads = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256_file(temporary / name),
            }
            for name in PAYLOAD_FILES
        }
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": int(step),
            "trainable_tensors": len(state),
            "trainable_parameters": sum(value.numel() for value in state.values()),
            "parameter_names_sha256": _name_hash(list(state)),
            "payloads": payloads,
            "metadata": metadata,
        }
        (temporary / "trainer_state.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return directory


def load_ref_video_lora_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    ema: TrainableEMA,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Verify and restore one proxy-video Ref2VA LoRA checkpoint."""
    directory = Path(directory)
    manifest = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {manifest.get('checkpoint_version')}")
    payloads = manifest.get("payloads", {})
    if set(payloads) != set(PAYLOAD_FILES):
        raise ValueError("Checkpoint payload set is incomplete or unexpected")
    for name in PAYLOAD_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(payloads[name]["bytes"]) or _sha256_file(path) != payloads[name]["sha256"]:
            raise ValueError(f"Checkpoint payload hash/size mismatch: {name}")

    parameters = _trainable_parameters(model)
    state = load_file(directory / "lora.safetensors")
    if set(state) != set(parameters):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(f"LoRA state mismatch; missing={missing}, unexpected={unexpected}")
    if _name_hash(list(parameters)) != manifest.get("parameter_names_sha256"):
        raise ValueError("LoRA parameter-name fingerprint mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"LoRA shape mismatch for {name}: {value.shape} != {parameter.shape}")
            parameter.copy_(value.to(parameter))
    ema.load_state_dict(load_file(directory / "ema.safetensors"))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
    if restore_rng:
        rng_state = torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"])
    return manifest


__all__ = ["load_ref_video_lora_checkpoint", "save_ref_video_lora_checkpoint"]
