"""Resumable LoRA/EMA checkpoints for H3 distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .rcm_adapter import TrainableEMA


CHECKPOINT_VERSION = 1


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("The model has no trainable adapter parameters to checkpoint.")
    return state


def load_trainable_weights(model: torch.nn.Module, state_path: str | Path) -> None:
    """Load a safetensors state into the model's currently trainable adapter parameters."""
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = load_file(state_path)
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(f"Adapter state mismatch; missing={missing}, unexpected={unexpected}.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"Adapter shape mismatch for {name}: {value.shape} != {parameter.shape}.")
            parameter.copy_(value.to(parameter))


def save_training_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    ema: TrainableEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save trainable weights, EMA, optimizer, RNG and a versioned JSON manifest."""
    if step < 0:
        raise ValueError("step cannot be negative.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter_state = _trainable_state(model)
    save_file(adapter_state, directory / "adapter.safetensors")
    save_file(ema.state_dict(), directory / "ema.safetensors")
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    rng_state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng_state, directory / "rng_state.pt")
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(step),
        "trainable_tensors": len(adapter_state),
        "trainable_parameters": sum(tensor.numel() for tensor in adapter_state.values()),
        "metadata": metadata or {},
    }
    (directory / "trainer_state.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_training_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    ema: TrainableEMA,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its validated manifest."""
    directory = Path(directory)
    manifest = json.loads((directory / "trainer_state.json").read_text())
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {manifest.get('checkpoint_version')}.")

    load_trainable_weights(model, directory / "adapter.safetensors")
    ema.load_state_dict(load_file(directory / "ema.safetensors"))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
    if restore_rng:
        rng_state = torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"])
    return manifest
