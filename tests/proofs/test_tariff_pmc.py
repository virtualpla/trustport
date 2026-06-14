from __future__ import annotations

import torch

from trustport.appraisal.tariff import DimensionScorer, pmc_index


def test_pmc_index_full_score() -> None:
    scores = torch.ones(4, 3)
    value = pmc_index(scores, (0.35, 0.30, 0.35))
    assert torch.allclose(value, torch.full((4,), 100.0))


def test_pmc_index_weighting() -> None:
    scores = torch.tensor([[1.0, 0.0, 0.0]])
    value = pmc_index(scores, (0.35, 0.30, 0.35))
    assert abs(float(value.item()) - 35.0) < 1e-4


def test_dimension_scorer_bounds() -> None:
    scorer = DimensionScorer(feature_dim=16, kbi_dim=8)
    out = scorer(torch.randn(5, 16), torch.randn(5, 8))
    assert out.shape == (5, 3)
    assert torch.all(out > 0.0) and torch.all(out < 1.0)
