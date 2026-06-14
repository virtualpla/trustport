from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from trustport.berths.consignments import BatchTensors
from trustport.manifest import OptimConfig
from trustport.voyage.objective import MultiTaskObjective


def _warmup_cosine(total_steps: int, warmup: int) -> Callable[[int], float]:
    def curve(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return curve


class Helmsman:
    def __init__(
        self,
        model: nn.Module,
        objective: MultiTaskObjective,
        config: OptimConfig,
        total_steps: int,
    ) -> None:
        self.model = model
        self.objective = objective
        self.config = config
        params = list(model.parameters()) + list(objective.parameters())
        self.optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        warmup = int(config.warmup_ratio * total_steps)
        curve = _warmup_cosine(total_steps, warmup)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=curve)
        self.ema: dict[str, torch.Tensor] = {}
        if config.ema_decay > 0.0:
            self.ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _update_ema(self) -> None:
        if not self.ema:
            return
        decay = self.config.ema_decay
        for key, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                self.ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.ema[key].copy_(value)

    def step(self, batch: BatchTensors) -> float:
        self.model.train()
        outputs = self.model(batch)
        loss, _ = self.objective(outputs["heads"], batch)
        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self._update_ema()
        return float(loss.item())

    def fit(self, batches: Iterable[BatchTensors], steps: int) -> list[float]:
        history: list[float] = []
        stream: Iterator[BatchTensors] = _cycle(batches)
        for _ in range(steps):
            history.append(self.step(next(stream)))
        return history


def _cycle(batches: Iterable[BatchTensors]) -> Iterator[BatchTensors]:
    cache: list[BatchTensors] = list(batches)
    while True:
        yield from cache
