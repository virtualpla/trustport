from __future__ import annotations

import math

import torch
from torch import nn


def lora_delta(a: torch.Tensor, b: torch.Tensor, alpha: int, rank: int) -> torch.Tensor:
    scale = alpha / rank
    weight: torch.Tensor = scale * (b @ a)
    return weight


class LoraLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
        enabled: bool,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(not enabled)
        self.base.bias.requires_grad_(not enabled)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.drop = nn.Dropout(dropout)
        if enabled:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out: torch.Tensor = self.base(x)
        if not self.enabled:
            return base_out
        update: torch.Tensor = self.drop(x) @ self.lora_a.t() @ self.lora_b.t()
        return base_out + self.scale * update

    def fused_weight(self) -> torch.Tensor:
        delta = lora_delta(self.lora_a, self.lora_b, self.alpha, self.rank)
        fused: torch.Tensor = self.base.weight + delta
        return fused
