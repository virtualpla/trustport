from __future__ import annotations

import torch
from torch import nn

from trustport.consolidation.crossties import hadamard_interactions


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        policy_dim: int,
        kbi_dim: int,
        fused_dim: int,
        pathways: int,
    ) -> None:
        super().__init__()
        self.to_clinical = nn.Linear(clinical_dim, fused_dim)
        self.to_policy = nn.Linear(policy_dim, fused_dim)
        self.to_kbi = nn.Linear(kbi_dim, fused_dim)
        self.gate = nn.Linear(fused_dim, 1)
        self.cross = nn.Linear(3 * fused_dim, fused_dim, bias=False)
        self.pathway = nn.Linear(fused_dim, pathways)

    def forward(
        self,
        clinical: torch.Tensor,
        policy: torch.Tensor,
        kbi: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fc = self.to_clinical(clinical)
        fp = self.to_policy(policy)
        fk = self.to_kbi(kbi)
        stacked = torch.stack([fc, fp, fk], dim=1)
        gate_logits = self.gate(torch.tanh(stacked)).squeeze(-1)
        if present is not None:
            gate_logits = gate_logits.masked_fill(~present, float("-inf"))
        weights = torch.softmax(gate_logits, dim=1)
        weighted = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        interactions = hadamard_interactions(fc, fp, fk)
        fused: torch.Tensor = weighted + self.cross(interactions)
        return fused, weights

    def pathway_logits(self, fused: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.pathway(fused)
        return logits


def pathway_clusters(logits: torch.Tensor) -> torch.Tensor:
    labels: torch.Tensor = logits.argmax(dim=-1)
    return labels
