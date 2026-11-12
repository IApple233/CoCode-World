"""Exponential moving averages of trainable adapter parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

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
