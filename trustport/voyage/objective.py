from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from trustport.berths.consignments import BatchTensors
from trustport.customs.seals import format_penalty
from trustport.quay.types import TASKS, TaskHeads


class LossParts(TypedDict):
    total: float
    ner: float
    re: float
    classification: float
    qa: float
    format: float


class MultiTaskObjective(nn.Module):
    def __init__(
        self,
        init_weights: tuple[float, float, float, float],
        format_gamma: float,
        use_struct_out: bool,
    ) -> None:
        super().__init__()
        log_sigma = [-0.5 * torch.log(torch.tensor(w)) for w in init_weights]
        self.log_sigma = nn.Parameter(torch.stack(log_sigma))
        self.format_gamma = format_gamma
        self.use_struct_out = use_struct_out

    def task_losses(self, heads: TaskHeads, batch: BatchTensors) -> dict[str, torch.Tensor]:
        ner_logits = heads["ner"]
        ner = F.cross_entropy(
            ner_logits.reshape(-1, ner_logits.shape[-1]), batch.ner_labels.reshape(-1)
        )
        relation = F.cross_entropy(heads["re"], batch.relation)
        category = F.cross_entropy(heads["classification"], batch.category)
        answer = F.cross_entropy(heads["qa"], batch.answer)
        return {"ner": ner, "re": relation, "classification": category, "qa": answer}

    def forward(self, heads: TaskHeads, batch: BatchTensors) -> tuple[torch.Tensor, LossParts]:
        losses = self.task_losses(heads, batch)
        precision = torch.exp(-self.log_sigma)
        total = heads["ner"].new_zeros(())
        for i, task in enumerate(TASKS):
            total = total + precision[i] * losses[task] + 0.5 * self.log_sigma[i]
        fmt = format_penalty(heads) if self.use_struct_out else heads["ner"].new_zeros(())
        total = total + self.format_gamma * fmt
        parts = LossParts(
            total=float(total.item()),
            ner=float(losses["ner"].item()),
            re=float(losses["re"].item()),
            classification=float(losses["classification"].item()),
            qa=float(losses["qa"].item()),
            format=float(fmt.item()),
        )
        return total, parts

    def normalized_weights(self) -> tuple[float, float, float, float]:
        precision = torch.exp(-self.log_sigma)
        weights = precision / precision.sum()
        values = [float(v) for v in weights.tolist()]
        return (values[0], values[1], values[2], values[3])
