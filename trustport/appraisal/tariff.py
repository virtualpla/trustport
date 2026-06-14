from __future__ import annotations

import torch
from torch import nn


def pmc_index(dim_scores: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=dim_scores.dtype, device=dim_scores.device)
    blended: torch.Tensor = (dim_scores * w).sum(dim=-1)
    return 100.0 * blended


class DimensionScorer(nn.Module):
    def __init__(self, feature_dim: int, kbi_dim: int, anchor_dim: int = 8) -> None:
        super().__init__()
        self.project = nn.Linear(feature_dim + kbi_dim + anchor_dim, 3)
        self.anchor_dim = anchor_dim

    def forward(
        self,
        features: torch.Tensor,
        kbi: torch.Tensor,
        anchors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchors is None:
            anchors = features.new_zeros((features.shape[0], self.anchor_dim))
        joined = torch.cat([features, kbi, anchors], dim=-1)
        scores: torch.Tensor = torch.sigmoid(self.project(joined))
        return scores
