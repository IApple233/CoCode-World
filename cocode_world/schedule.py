"""Reproducible, CPU-only sampling and schedules for H3 training."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import random


DEFAULT_EVALUATION_PACKAGE = "examples/evaluation"
DEFAULT_TRAINING_SETTINGS = {
    "epochs": 3,
    "lora_rank": 64,
    "lora_alpha": 64.0,
    "learning_rate": 1e-5,
    "warmup_steps": 200,
    "shuffle_buffer_size": 32768,
    "save_every": 500,
    "eval_every": 1000,
    "evaluate_final": True,
    "evaluation_package": DEFAULT_EVALUATION_PACKAGE,
    "evaluation_count": 30,
    "exclude_evaluation_source_samples": True,
}
SAMPLER_REVISION = "epoch-buffer-shuffle-seed-plus-epoch-v1"
OPTIONAL_TRAINING_SETTINGS = {"decfg_adapter", "decfg_adapter_scale"}


def load_training_settings(run_dir: Path) -> dict:
    path = Path(run_dir) / "training_config.json"
    overrides = json.loads(path.read_text()) if path.exists() else {}
    unknown = set(overrides) - set(DEFAULT_TRAINING_SETTINGS) - OPTIONAL_TRAINING_SETTINGS
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    settings = {**DEFAULT_TRAINING_SETTINGS, **overrides}
    for name in ("epochs", "lora_rank", "save_every", "eval_every", "evaluation_count"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("warmup_steps", "shuffle_buffer_size"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("lora_alpha", "learning_rate"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("evaluate_final", "exclude_evaluation_source_samples"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be boolean")
    if not isinstance(settings["evaluation_package"], str) or not settings["evaluation_package"]:
        raise ValueError("evaluation_package must name a dataset directory")
    adapter = settings.get("decfg_adapter")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        raise ValueError("decfg_adapter must be a nonempty path or null")
    scale = settings.get("decfg_adapter_scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("decfg_adapter_scale must be finite and positive")
    if adapter is None and scale != 1.0:
        raise ValueError("decfg_adapter_scale requires decfg_adapter")
    return settings


def buffered_epoch_indices(size: int, *, seed: int, epoch: int, buffer_size: int):
    """Visit every index once, resetting the shuffle RNG at each epoch.

    Buffer entries are integer indices, never decoded videos or latent tensors.
    A buffer covering the dataset is a full permutation; smaller buffers bound
    the streaming shuffle's lookahead. Zero preserves legacy sequential order.
    """
    if size < 1 or epoch < 0 or buffer_size < 0:
        raise ValueError("Invalid epoch sampler dimensions")
    if buffer_size == 0:
        yield from range(size)
        return
    rng = random.Random(seed + epoch)
    buffer = list(range(min(size, buffer_size)))
    for index in range(len(buffer), size):
        slot = rng.randrange(len(buffer))
        yield buffer[slot]
        buffer[slot] = index
    rng.shuffle(buffer)
    yield from buffer


class BufferedEpochSampler:
    """Map a global sample offset to a deterministic shuffled dataset index.

    All CP peers and DP replicas use the same order. DP takes disjoint offsets
    from that global stream. Epoch tails flow into the next epoch without
    dropping/padding records. Resuming needs only the saved optimizer step,
    DP degree, accumulation, dataset length, seed and buffer-size contract.
    """

    def __init__(self, size: int, *, seed: int, buffer_size: int):
        if size < 1 or buffer_size < 0:
            raise ValueError("Invalid epoch sampler dimensions")
        self.size = size
        self.seed = seed
        self.buffer_size = buffer_size
        # Cache only integer orders for the two epochs a boundary batch can use.
        self._order = lru_cache(maxsize=2)(self._epoch_order)

    def _epoch_order(self, epoch: int) -> tuple[int, ...]:
        return tuple(buffered_epoch_indices(
            self.size, seed=self.seed, epoch=epoch, buffer_size=self.buffer_size,
        ))

    def index_at(self, global_sample_offset: int) -> int:
        if global_sample_offset < 0:
            raise ValueError("Global sample offset cannot be negative")
        epoch, offset = divmod(global_sample_offset, self.size)
        return self._order(epoch)[offset]

    def step_indices(self, step: int, *, replicas: int, accumulation: int) -> list[list[int]]:
        if step < 0 or replicas < 1 or accumulation < 1:
            raise ValueError("Invalid distributed sample assignment")
        return [[self.index_at((step * accumulation + microbatch) * replicas + replica)
                 for microbatch in range(accumulation)] for replica in range(replicas)]


def learning_rate_for_step(base_lr: float, step: int, warmup_steps: int) -> float:
    """LR for a zero-based optimizer step, derived directly from resumed step."""
    if not math.isfinite(base_lr) or base_lr <= 0 or step < 0 or warmup_steps < 0:
        raise ValueError("Invalid learning-rate schedule")
    return base_lr * min(1.0, (step + 1) / warmup_steps) if warmup_steps else base_lr


def training_environment(settings: dict, *, preview: bool = False) -> list[str]:
    values = {
        "LORA_RANK": settings["lora_rank"],
        "LORA_ALPHA": settings["lora_alpha"],
        "LEARNING_RATE": settings["learning_rate"],
        "WARMUP_STEPS": settings["warmup_steps"],
        "SHUFFLE_BUFFER_SIZE": settings["shuffle_buffer_size"],
        "SAVE_EVERY": 2 if preview else settings["save_every"],
        "DECFG_ADAPTER": settings.get("decfg_adapter") or "",
        "DECFG_ADAPTER_SCALE": settings.get("decfg_adapter_scale", 1.0),
    }
    return [f"H3_REF_VIDEO_{name}={value}" for name, value in values.items()]
