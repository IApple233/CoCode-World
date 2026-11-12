"""Deterministic data-forcing decisions and RNG streams for H3 DMD/DFD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Literal

import torch

from .schedules import shift_noise_ratio


UpdateRole = Literal["student", "fake_score"]
RNG_STREAM_NAMES = (
    "data_order",
    "student_noise",
    "student_grid_index",
    "score_noise",
    "score_time",
    "dfd_mask",
    "evaluation_seed",
)
DEFAULT_STUDENT_GRID_U = (1.0, 36.0 / 37.0, 12.0 / 13.0, 0.8)


@dataclass(frozen=True)
class DfdBatchPlan:
    """All scalar random choices needed by one outer iteration."""

    outer_iteration: int
    update_role: UpdateRole
    data_index: int
    use_real_teacher_score: bool
    mask_draw: float | None
    student_grid_index: int
    student_u: float
    score_u: float
    student_noise_seed: int
    score_noise_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DfdBatchPlan":
        return cls(**value)


@dataclass(frozen=True)
class DfdOuterDecision:
    """One update role and one Bernoulli decision shared by all accumulated microbatches."""

    outer_iteration: int
    update_role: UpdateRole
    use_real_teacher_score: bool
    mask_draw: float | None


class NamedRngStreams:
    """Independent CPU generators with serializable state and draw counters."""

    def __init__(self, base_seed: int, names: tuple[str, ...] = RNG_STREAM_NAMES) -> None:
        if base_seed < 0:
            raise ValueError("base_seed cannot be negative.")
        if not names or len(set(names)) != len(names):
            raise ValueError("RNG stream names must be non-empty and unique.")
        self.base_seed = int(base_seed)
        self.names = tuple(names)
        self._generators: dict[str, torch.Generator] = {}
        self._draw_counts = {name: 0 for name in self.names}
        for name in self.names:
            digest = hashlib.sha256(f"{self.base_seed}:{name}".encode("ascii")).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            self._generators[name] = torch.Generator(device="cpu").manual_seed(seed)

    def _generator(self, name: str) -> torch.Generator:
        if name not in self._generators:
            raise KeyError(f"Unknown RNG stream {name!r}; expected one of {self.names}.")
        return self._generators[name]

    def rand(self, name: str) -> float:
        result = float(torch.rand((), generator=self._generator(name), dtype=torch.float64))
        self._draw_counts[name] += 1
        return result

    def randint(self, name: str, high: int) -> int:
        if high <= 0:
            raise ValueError("randint high must be positive.")
        result = int(torch.randint(high, (), generator=self._generator(name), dtype=torch.int64))
        self._draw_counts[name] += 1
        return result

    def next_seed(self, name: str) -> int:
        return self.randint(name, 2**63 - 1)

    @property
    def draw_counts(self) -> dict[str, int]:
        return dict(self._draw_counts)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "names": list(self.names),
            "draw_counts": self.draw_counts,
            "generator_states": {name: generator.get_state().cpu() for name, generator in self._generators.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("base_seed", -1)) != self.base_seed:
            raise ValueError("RNG base seed mismatch.")
        if tuple(state_dict.get("names", ())) != self.names:
            raise ValueError("RNG stream name/order mismatch.")
        states = state_dict.get("generator_states", {})
        counts = state_dict.get("draw_counts", {})
        if set(states) != set(self.names) or set(counts) != set(self.names):
            raise ValueError("RNG checkpoint does not contain the exact named streams.")
        for name in self.names:
            count = int(counts[name])
            if count < 0:
                raise ValueError(f"Negative RNG draw count for {name}.")
            self._generators[name].set_state(states[name].cpu())
            self._draw_counts[name] = count


def update_role_for_iteration(outer_iteration: int, student_update_frequency: int = 5) -> UpdateRole:
    """Match upstream's zero-based one-student/four-fake update schedule."""
    if outer_iteration < 0:
        raise ValueError("outer_iteration cannot be negative.")
    if student_update_frequency < 1:
        raise ValueError("student_update_frequency must be positive.")
    return "student" if outer_iteration % student_update_frequency == 0 else "fake_score"


def build_dfd_teacher_input(
    generated: torch.Tensor,
    real: torch.Tensor,
    use_real: bool,
) -> torch.Tensor:
    """Route the teacher score to generated data (DMD) or paired real data (DFD)."""
    if generated.shape != real.shape:
        raise ValueError(f"generated and real shapes differ: {tuple(generated.shape)} != {tuple(real.shape)}.")
    if generated.device != real.device:
        raise ValueError("generated and real must be on the same device.")
    if generated.dtype != real.dtype:
        raise ValueError("generated and real must have the same dtype.")
    if not use_real:
        return generated
    # Teacher scoring is always no-grad. Returning real directly also avoids the
    # one-ULP cancellation error in upstream's straight-through expression.
    return real.detach()


def make_dfd_batch_plan(
    outer_iteration: int,
    *,
    streams: NamedRngStreams,
    dataset_size: int,
    dfd_probability: float,
    student_grid_u: tuple[float, ...] = DEFAULT_STUDENT_GRID_U,
    student_update_frequency: int = 5,
    score_time_min: float = 0.001,
    score_time_max: float = 0.999,
    score_time_shift: float = 12.0,
) -> DfdBatchPlan:
    """Draw one reproducible single-microbatch outer plan."""
    decision = make_dfd_outer_decision(
        outer_iteration,
        streams=streams,
        dfd_probability=dfd_probability,
        student_update_frequency=student_update_frequency,
    )
    return make_dfd_microbatch_plan(
        decision,
        streams=streams,
        dataset_size=dataset_size,
        student_grid_u=student_grid_u,
        score_time_min=score_time_min,
        score_time_max=score_time_max,
        score_time_shift=score_time_shift,
    )


def make_dfd_outer_decision(
    outer_iteration: int,
    *,
    streams: NamedRngStreams,
    dfd_probability: float,
    student_update_frequency: int = 5,
) -> DfdOuterDecision:
    """Draw the only DFD Bernoulli used by an accumulated outer iteration."""
    if not 0 <= dfd_probability <= 1:
        raise ValueError("dfd_probability must be in [0, 1].")
    role = update_role_for_iteration(outer_iteration, student_update_frequency)
    mask_draw: float | None = None
    use_real = False
    if role == "student":
        mask_draw = streams.rand("dfd_mask")
        use_real = mask_draw < dfd_probability
    return DfdOuterDecision(outer_iteration, role, use_real, mask_draw)


def make_dfd_microbatch_plan(
    decision: DfdOuterDecision,
    *,
    streams: NamedRngStreams,
    dataset_size: int,
    student_grid_u: tuple[float, ...] = DEFAULT_STUDENT_GRID_U,
    score_time_min: float = 0.001,
    score_time_max: float = 0.999,
    score_time_shift: float = 12.0,
) -> DfdBatchPlan:
    """Draw data/time/noise choices while reusing the outer iteration's one mask."""
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if not student_grid_u or any(not 0 < value <= 1 for value in student_grid_u):
        raise ValueError("student_grid_u must contain positive values no greater than one.")
    if not 0 <= score_time_min < score_time_max <= 1:
        raise ValueError("score time bounds must satisfy 0 <= min < max <= 1.")

    data_index = streams.randint("data_order", dataset_size)
    grid_index = streams.randint("student_grid_index", len(student_grid_u))
    student_noise_seed = streams.next_seed("student_noise")
    score_noise_seed = streams.next_seed("score_noise")
    raw_score_u = score_time_min + (score_time_max - score_time_min) * streams.rand("score_time")
    shifted_score_u = float(shift_noise_ratio(torch.tensor(raw_score_u, dtype=torch.float64), score_time_shift))
    score_u = min(score_time_max, max(score_time_min, shifted_score_u))

    return DfdBatchPlan(
        outer_iteration=decision.outer_iteration,
        update_role=decision.update_role,
        data_index=data_index,
        use_real_teacher_score=decision.use_real_teacher_score,
        mask_draw=decision.mask_draw,
        student_grid_index=grid_index,
        student_u=float(student_grid_u[grid_index]),
        score_u=score_u,
        student_noise_seed=student_noise_seed,
        score_noise_seed=score_noise_seed,
    )


def mask_sequence_sha256(plans: list[DfdBatchPlan]) -> str:
    """Hash only student-update Bernoulli outcomes as an auditable golden fixture."""
    bits = "".join("1" if plan.use_real_teacher_score else "0" for plan in plans if plan.update_role == "student")
    return hashlib.sha256(bits.encode("ascii")).hexdigest()
