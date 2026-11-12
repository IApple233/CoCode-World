"""Architecture-preserving role adapter for MiniMax-H3 rCM distillation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch

from .consistency import h3_clean_from_velocity
from .packed_video import H3VideoLayout


H3Role = Literal["teacher", "student", "student_ema"]


class TrainableEMA:
    """FP32 EMA shadow of the model's trainable adapter parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self._parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not self._parameters:
            raise ValueError("Cannot construct an EMA before trainable adapter parameters exist.")
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
            name: value.to(device=self._parameters[name].device, dtype=self._parameters[name].dtype)
            for name, value in self.shadow.items()
        }
        try:
            for name, parameter in self._parameters.items():
                parameter.data = staged[name]
            yield
        finally:
            for name, parameter in self._parameters.items():
                parameter.data = original[name]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if set(state_dict) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(self.shadow))
            raise ValueError(f"EMA state mismatch; missing={missing}, unexpected={unexpected}.")
        for name, tensor in state_dict.items():
            if tensor.shape != self.shadow[name].shape:
                raise ValueError(f"EMA shape mismatch for {name}: {tensor.shape} != {self.shadow[name].shape}.")
            self.shadow[name].copy_(tensor.to(self.shadow[name]))


@dataclass
class H3RcmAdapter:
    """Bind one H3 model, video-only layout and cached text condition to rCM coordinates."""

    model: torch.nn.Module
    layout: H3VideoLayout
    prompt_embeds: torch.Tensor
    ema: TrainableEMA | None = None

    def __post_init__(self) -> None:
        if self.prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [B,L,text_dim].")
        if self.prompt_embeds.shape[1] != self.layout.text_indices.numel():
            raise ValueError("prompt_embeds length does not match the H3 text rows.")

    @contextmanager
    def _role(self, role: H3Role) -> Iterator[None]:
        has_adapters = bool(getattr(self.model, "peft_config", {}))
        if role == "teacher":
            if not has_adapters:
                yield
                return
            self.model.disable_adapters()
            try:
                yield
            finally:
                self.model.enable_adapters()
            return
        if role == "student_ema":
            if self.ema is None:
                raise ValueError("student_ema was requested without an EMA state.")
            self.model.enable_adapters()
            with self.ema.apply():
                yield
            return
        if role != "student":
            raise ValueError(f"Unknown H3 role: {role!r}.")
        if has_adapters:
            self.model.enable_adapters()
        yield

    def predict_h3_velocity(self, sample: torch.Tensor, u: float | torch.Tensor, role: H3Role) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[1:] != (self.layout.num_video_rows, self.layout.video_row_width):
            raise ValueError(
                f"sample must have shape [B,{self.layout.num_video_rows},{self.layout.video_row_width}], "
                f"got {tuple(sample.shape)}."
            )
        if sample.shape[0] != self.prompt_embeds.shape[0]:
            raise ValueError("sample and prompt_embeds batch sizes differ.")
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device=sample.device).flatten()
        if u_tensor.numel() != 1 or not bool(torch.isfinite(u_tensor).all()) or not 0 <= float(u_tensor[0]) <= 1:
            raise ValueError("H3 currently requires one shared finite RF time u in [0, 1] per microbatch.")
        model_kwargs = self.layout.structural_kwargs(1.0 - u_tensor, sample.device)
        empty_audio = torch.empty(
            sample.shape[0], 0, self.layout.audio_channels, device=sample.device, dtype=sample.dtype
        )
        with self._role(role):
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
        return video_velocity

    def predict_rcm_velocity(self, sample: torch.Tensor, u: float | torch.Tensor, role: H3Role) -> torch.Tensor:
        return -self.predict_h3_velocity(sample, u, role)

    def predict_clean(self, sample: torch.Tensor, u: float | torch.Tensor, role: H3Role) -> torch.Tensor:
        h3_velocity = self.predict_h3_velocity(sample, u, role)
        return h3_clean_from_velocity(sample, h3_velocity, 1.0 - torch.as_tensor(u, device=sample.device))
