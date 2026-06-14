from __future__ import annotations

import torch

from trustport.consolidation.containerize import CrossModalFusion, pathway_clusters
from trustport.consolidation.crossties import hadamard_interactions


def test_hadamard_shape() -> None:
    out = hadamard_interactions(torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8))
    assert out.shape == (3, 24)


def test_fusion_shape_and_weights() -> None:
    fusion = CrossModalFusion(8, 8, 6, 8, 4)
    fused, weights = fusion(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 6))
    assert fused.shape == (4, 8)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_modality_dropout_downweights() -> None:
    fusion = CrossModalFusion(8, 8, 6, 8, 4)
    present = torch.tensor([[True, True, False]])
    _, weights = fusion(torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 6), present=present)
    assert float(weights[0, 2].item()) < 1e-6


def test_fusion_gradient_flow() -> None:
    fusion = CrossModalFusion(8, 8, 6, 8, 4)
    fused, _ = fusion(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 6))
    fused.sum().backward()
    grad = fusion.to_clinical.weight.grad
    assert grad is not None and torch.any(grad != 0.0)


def test_pathway_labels_range() -> None:
    labels = pathway_clusters(torch.randn(10, 4))
    assert int(labels.min().item()) >= 0 and int(labels.max().item()) < 4
