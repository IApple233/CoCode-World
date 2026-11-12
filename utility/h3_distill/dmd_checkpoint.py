"""Atomic, role-explicit checkpoint schema v2 for H3 DMD/DFD."""

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

from .dfd import NamedRngStreams
from .role_adapters import H3DmdRoleManager, RoleScopedEMA


DMD_CHECKPOINT_VERSION = 2
DMD_PAYLOAD_FILES = (
    "student.safetensors",
    "fake_score.safetensors",
    "student_ema.safetensors",
    "student_optimizer.pt",
    "fake_score_optimizer.pt",
    "rng_state.pt",
)
REQUIRED_COUNTERS = (
    "outer_iteration",
    "student_updates",
    "fake_score_updates",
    "samples_seen",
    "transformer_forwards",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_name_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def _validate_counters(counters: dict[str, Any]) -> None:
    missing = set(REQUIRED_COUNTERS) - set(counters)
    if missing:
        raise ValueError(f"Checkpoint counters are missing {sorted(missing)}.")
    for name in REQUIRED_COUNTERS[:-1]:
        if not isinstance(counters[name], int) or counters[name] < 0:
            raise ValueError(f"Counter {name} must be a non-negative integer.")
    forwards = counters["transformer_forwards"]
    expected_roles = {"teacher", "student", "student_ema", "fake_score"}
    if set(forwards) != expected_roles or any(not isinstance(value, int) or value < 0 for value in forwards.values()):
        raise ValueError("transformer_forwards must contain non-negative counters for all four roles.")


def save_dmd_training_checkpoint(
    directory: str | Path,
    *,
    roles: H3DmdRoleManager,
    student_ema: RoleScopedEMA,
    student_optimizer: torch.optim.Optimizer,
    fake_score_optimizer: torch.optim.Optimizer,
    rng_streams: NamedRngStreams,
    counters: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a new v2 checkpoint atomically and refuse to replace any existing directory."""
    _validate_counters(counters)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint {directory}.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{directory.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        student_state = roles.adapter_state("student")
        fake_state = roles.adapter_state("fake_score")
        save_file(student_state, temporary / "student.safetensors")
        save_file(fake_state, temporary / "fake_score.safetensors")
        save_file(student_ema.state_dict(), temporary / "student_ema.safetensors")
        torch.save(student_optimizer.state_dict(), temporary / "student_optimizer.pt")
        torch.save(fake_score_optimizer.state_dict(), temporary / "fake_score_optimizer.pt")
        torch.save(rng_streams.state_dict(), temporary / "rng_state.pt")

        payloads = {
            name: {"bytes": (temporary / name).stat().st_size, "sha256": sha256_file(temporary / name)}
            for name in DMD_PAYLOAD_FILES
        }
        manifest = {
            "checkpoint_version": DMD_CHECKPOINT_VERSION,
            "counters": counters,
            "student_tensors": len(student_state),
            "student_parameters": sum(value.numel() for value in student_state.values()),
            "fake_score_tensors": len(fake_state),
            "fake_score_parameters": sum(value.numel() for value in fake_state.values()),
            "student_parameter_names_sha256": parameter_name_sha256(list(student_state)),
            "fake_score_parameter_names_sha256": parameter_name_sha256(list(fake_state)),
            "rng_draw_counts": rng_streams.draw_counts,
            "payloads": payloads,
            "metadata": metadata or {},
        }
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (temporary / "trainer_state.json").write_text(serialized, encoding="utf-8")
        os.rename(temporary, directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return directory


def load_dmd_training_checkpoint(
    directory: str | Path,
    *,
    roles: H3DmdRoleManager,
    student_ema: RoleScopedEMA,
    student_optimizer: torch.optim.Optimizer | None = None,
    fake_score_optimizer: torch.optim.Optimizer | None = None,
    rng_streams: NamedRngStreams | None = None,
) -> dict[str, Any]:
    """Verify every payload hash before restoring a schema-v2 checkpoint."""
    directory = Path(directory)
    manifest = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_version") != DMD_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported DMD checkpoint version: {manifest.get('checkpoint_version')}.")
    _validate_counters(manifest.get("counters", {}))
    payloads = manifest.get("payloads", {})
    if set(payloads) != set(DMD_PAYLOAD_FILES):
        raise ValueError("Checkpoint manifest has an incomplete or unexpected payload set.")
    for name in DMD_PAYLOAD_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(payloads[name]["bytes"]) or sha256_file(path) != payloads[name]["sha256"]:
            raise ValueError(f"Checkpoint payload hash/size mismatch for {name}.")

    roles.load_adapter_state("student", load_file(directory / "student.safetensors"))
    roles.load_adapter_state("fake_score", load_file(directory / "fake_score.safetensors"))
    student_ema.load_state_dict(load_file(directory / "student_ema.safetensors"))
    if student_optimizer is not None:
        student_optimizer.load_state_dict(
            torch.load(directory / "student_optimizer.pt", map_location="cpu", weights_only=True)
        )
    if fake_score_optimizer is not None:
        fake_score_optimizer.load_state_dict(
            torch.load(directory / "fake_score_optimizer.pt", map_location="cpu", weights_only=True)
        )
    if rng_streams is not None:
        rng_streams.load_state_dict(torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=True))
        if rng_streams.draw_counts != manifest["rng_draw_counts"]:
            raise ValueError("Restored RNG draw counters differ from the checkpoint manifest.")
    return manifest
