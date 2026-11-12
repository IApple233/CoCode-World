"""Role-isolated student and fake-score LoRA adapters for H3 DMD/DFD."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Iterator, Literal

import torch
from safetensors.torch import load_file

from .consistency import h3_clean_from_velocity
from .packed_video import H3VideoLayout


DmdRole = Literal["teacher", "student", "student_ema", "fake_score"]


def _belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
    return adapter_name in parameter_name.split(".") and ("lora_" in parameter_name or "modules_to_save" in parameter_name)


class RoleScopedEMA:
    """FP32 EMA containing exactly one named student adapter."""

    def __init__(
        self,
        parameters: dict[str, torch.nn.Parameter],
        *,
        decay: float = 0.999,
    ) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1).")
        if not parameters:
            raise ValueError("RoleScopedEMA requires a non-empty parameter mapping.")
        self.decay = float(decay)
        self._parameters = dict(parameters)
        self.shadow = {
            name: parameter.detach().to(device=parameter.device, dtype=torch.float32).clone()
            for name, parameter in self._parameters.items()
        }

    @torch.no_grad()
    def update(self) -> None:
        for name, parameter in self._parameters.items():
            self.shadow[name].lerp_(parameter.detach().float(), 1.0 - self.decay)

    @contextmanager
    def apply(self) -> Iterator[None]:
        original = {name: parameter.data for name, parameter in self._parameters.items()}
        staged = {
            name: shadow.to(device=self._parameters[name].device, dtype=self._parameters[name].dtype)
            for name, shadow in self.shadow.items()
        }
        try:
            for name, parameter in self._parameters.items():
                parameter.data = staged[name]
            yield
        finally:
            for name, parameter in self._parameters.items():
                parameter.data = original[name]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if set(state_dict) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(self.shadow))
            raise ValueError(f"Student EMA state mismatch; missing={missing}, unexpected={unexpected}.")
        for name, value in state_dict.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"Student EMA shape mismatch for {name}: {value.shape} != {self.shadow[name].shape}.")
            self.shadow[name].copy_(value.to(self.shadow[name]))


class H3DmdRoleManager:
    """Own fixed parameter sets and make every H3 forward use exactly one role."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        student_adapter: str = "student",
        fake_score_adapter: str = "fake_score",
    ) -> None:
        self.model = model
        self.student_adapter = student_adapter
        self.fake_score_adapter = fake_score_adapter
        configured = set(getattr(model, "peft_config", {}))
        expected = {student_adapter, fake_score_adapter}
        if not expected.issubset(configured):
            raise ValueError(f"Missing named adapters {sorted(expected - configured)}; configured={sorted(configured)}.")
        self._adapter_parameters = {
            student_adapter: {
                name: parameter
                for name, parameter in model.named_parameters()
                if _belongs_to_adapter(name, student_adapter)
            },
            fake_score_adapter: {
                name: parameter
                for name, parameter in model.named_parameters()
                if _belongs_to_adapter(name, fake_score_adapter)
            },
        }
        for name, parameters in self._adapter_parameters.items():
            if not parameters:
                raise ValueError(f"Adapter {name!r} has no parameters.")
        overlap = set(self._adapter_parameters[student_adapter]) & set(self._adapter_parameters[fake_score_adapter])
        if overlap:
            raise AssertionError(f"Named adapter parameter sets overlap: {sorted(overlap)}.")
        self._all_adapter_names = set().union(*(set(values) for values in self._adapter_parameters.values()))
        self._base_parameters = {
            name: parameter for name, parameter in model.named_parameters() if name not in self._all_adapter_names
        }
        self.active_role: DmdRole | None = None
        self.freeze_all()

    @property
    def student_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self._adapter_parameters[self.student_adapter])

    @property
    def fake_score_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self._adapter_parameters[self.fake_score_adapter])

    @property
    def base_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self._base_parameters)

    def freeze_all(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def activate(self, role: DmdRole, *, trainable: bool = False) -> None:
        """Persistently activate one role so gradients survive until backward."""
        self.freeze_all()
        if role == "teacher":
            self.model.disable_adapters()
            self.active_role = role
            return
        adapter_name = self.student_adapter if role in ("student", "student_ema") else self.fake_score_adapter
        if role not in ("student", "student_ema", "fake_score"):
            raise ValueError(f"Unknown DMD role {role!r}.")
        if role == "student_ema" and trainable:
            raise ValueError("student_ema cannot be trainable.")
        self.model.enable_adapters()
        self.model.set_adapter(adapter_name)
        self.freeze_all()  # PEFT set_adapter may mark the active role trainable.
        if trainable:
            for parameter in self._adapter_parameters[adapter_name].values():
                parameter.requires_grad_(True)
        if any(parameter.requires_grad for parameter in self._base_parameters.values()):
            raise AssertionError("A base H3 parameter became trainable during role activation.")
        self.active_role = role

    def parameters_for_role(self, role: Literal["student", "fake_score"]) -> list[torch.nn.Parameter]:
        adapter_name = self.student_adapter if role == "student" else self.fake_score_adapter
        return list(self._adapter_parameters[adapter_name].values())

    def adapter_state(self, role: Literal["student", "fake_score"]) -> dict[str, torch.Tensor]:
        adapter_name = self.student_adapter if role == "student" else self.fake_score_adapter
        return {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in self._adapter_parameters[adapter_name].items()
        }

    @torch.no_grad()
    def load_adapter_state(
        self,
        role: Literal["student", "fake_score"],
        state_dict: dict[str, torch.Tensor],
        *,
        source_adapter: str | None = None,
    ) -> None:
        adapter_name = self.student_adapter if role == "student" else self.fake_score_adapter
        if source_adapter is not None and source_adapter != adapter_name:
            source_token = f".{source_adapter}."
            target_token = f".{adapter_name}."
            state_dict = {name.replace(source_token, target_token): value for name, value in state_dict.items()}
        parameters = self._adapter_parameters[adapter_name]
        if set(state_dict) != set(parameters):
            missing = sorted(set(parameters) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(parameters))
            raise ValueError(f"{role} adapter state mismatch; missing={missing}, unexpected={unexpected}.")
        for name, parameter in parameters.items():
            value = state_dict[name]
            if value.shape != parameter.shape:
                raise ValueError(f"{role} shape mismatch for {name}: {value.shape} != {parameter.shape}.")
            parameter.copy_(value.to(parameter))

    def load_legacy_student_safetensors(self, path: str | Path, *, source_adapter: str = "default") -> None:
        self.load_adapter_state("student", load_file(path), source_adapter=source_adapter)

    def base_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self._base_parameters.items()):
            digest.update(name.encode("utf-8"))
            value = parameter.detach().contiguous().view(torch.uint8).cpu()
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()


@dataclass
class H3DmdAdapter:
    """Bind one physical H3 base to teacher, student, EMA and fake-score roles."""

    model: torch.nn.Module
    layout: H3VideoLayout
    prompt_embeds: torch.Tensor
    roles: H3DmdRoleManager
    student_ema: RoleScopedEMA | None = None
    forward_counts: dict[str, int] = field(
        default_factory=lambda: {"teacher": 0, "student": 0, "student_ema": 0, "fake_score": 0}
    )

    def __post_init__(self) -> None:
        if self.roles.model is not self.model:
            raise ValueError("H3DmdAdapter and role manager must reference the same physical model.")
        if self.prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [B,L,text_dim].")
        if self.prompt_embeds.shape[1] != self.layout.text_indices.numel():
            raise ValueError("prompt_embeds length does not match the H3 text rows.")

    def reset_forward_counts(self) -> None:
        for role in self.forward_counts:
            self.forward_counts[role] = 0

    def predict_h3_velocity(
        self,
        sample: torch.Tensor,
        u: float | torch.Tensor,
        role: DmdRole,
        *,
        trainable: bool = False,
    ) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[1:] != (self.layout.num_video_rows, self.layout.video_row_width):
            raise ValueError(
                f"sample must have shape [B,{self.layout.num_video_rows},{self.layout.video_row_width}], "
                f"got {tuple(sample.shape)}."
            )
        if sample.shape[0] != self.prompt_embeds.shape[0]:
            raise ValueError("sample and prompt_embeds batch sizes differ.")
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device=sample.device).flatten()
        if u_tensor.numel() != 1 or not bool(torch.isfinite(u_tensor).all()) or not 0 <= float(u_tensor[0]) <= 1:
            raise ValueError("H3 requires one shared finite RF time u in [0, 1] per microbatch.")
        if role == "teacher" and trainable:
            raise ValueError("The teacher cannot be trainable.")
        if role == "student_ema" and self.student_ema is None:
            raise ValueError("student_ema was requested without an EMA state.")

        self.roles.activate(role, trainable=trainable)
        model_kwargs = self.layout.structural_kwargs(1.0 - u_tensor, sample.device)
        empty_audio = torch.empty(
            sample.shape[0], 0, self.layout.audio_channels, device=sample.device, dtype=sample.dtype
        )
        ema_context = self.student_ema.apply() if role == "student_ema" else _null_context()
        with ema_context:
            video_velocity, audio_velocity = self.model(
                hidden_states=sample,
                audio_hidden_states=empty_audio,
                encoder_hidden_states=self.prompt_embeds.to(sample.device),
                return_dict=False,
                **model_kwargs,
            )
        if video_velocity.shape != sample.shape:
            raise RuntimeError(f"Unexpected H3 video output shape {tuple(video_velocity.shape)}.")
        if audio_velocity.shape != empty_audio.shape:
            raise RuntimeError(f"Unexpected H3 empty-audio output shape {tuple(audio_velocity.shape)}.")
        self.forward_counts[role] += 1
        return video_velocity

    def predict_clean(
        self,
        sample: torch.Tensor,
        u: float | torch.Tensor,
        role: DmdRole,
        *,
        trainable: bool = False,
    ) -> torch.Tensor:
        velocity = self.predict_h3_velocity(sample, u, role, trainable=trainable)
        return h3_clean_from_velocity(sample, velocity, 1.0 - torch.as_tensor(u, device=sample.device))


@contextmanager
def _null_context() -> Iterator[None]:
    yield
