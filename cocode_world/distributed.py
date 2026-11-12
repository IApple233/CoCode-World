"""Context-parallel gradient synchronization."""

from __future__ import annotations

import torch
import torch.distributed as dist

@torch.no_grad()
def sum_context_parallel_gradients(
    parameters: list[torch.nn.Parameter], group: dist.ProcessGroup | None = None
) -> None:
    """Sum replicated parameter gradients whose sequence contributions are split across CP ranks."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
