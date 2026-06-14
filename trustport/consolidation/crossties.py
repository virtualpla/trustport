from __future__ import annotations

import torch


def hadamard_interactions(
    clinical: torch.Tensor,
    policy: torch.Tensor,
    kbi: torch.Tensor,
) -> torch.Tensor:
    cp = clinical * policy
    pk = policy * kbi
    ck = clinical * kbi
    return torch.cat([cp, pk, ck], dim=-1)
