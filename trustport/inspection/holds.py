from __future__ import annotations

import torch


def reference_grounding(predicted: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
    pred = (predicted > 0).float()
    ref = (retrieved > 0).float()
    intersection = (pred * ref).sum(dim=-1)
    union = torch.clamp(pred.sum(dim=-1) + ref.sum(dim=-1) - intersection, min=1.0)
    return intersection / union


def self_consistency(passes: torch.Tensor) -> torch.Tensor:
    k = passes.shape[0]
    batch = passes.shape[1]
    agreement = torch.empty(batch, dtype=torch.float32)
    for i in range(batch):
        column = passes[:, i]
        _, counts = torch.unique(column, return_counts=True)
        agreement[i] = counts.max().item() / k
    return agreement


def schema_signal(conformance: torch.Tensor) -> torch.Tensor:
    return conformance.clamp(0.0, 1.0)


def hallucination_score(
    reference: torch.Tensor,
    consistency: torch.Tensor,
    schema: torch.Tensor,
) -> torch.Tensor:
    blended = (reference + consistency + schema) / 3.0
    return 1.0 - blended
