from __future__ import annotations

import torch

from trustport.customs.ledger import LoraLinear, lora_delta


def test_lora_delta_scaling() -> None:
    rank, alpha = 4, 16
    a = torch.randn(rank, 8)
    b = torch.randn(6, rank)
    delta = lora_delta(a, b, alpha, rank)
    expected = (alpha / rank) * (b @ a)
    assert torch.allclose(delta, expected)
    assert delta.shape == (6, 8)


def test_lora_forward_shape() -> None:
    layer = LoraLinear(8, 6, rank=4, alpha=16, dropout=0.0, enabled=True)
    out = layer(torch.randn(3, 8))
    assert out.shape == (3, 6)


def test_lora_disabled_matches_base() -> None:
    layer = LoraLinear(8, 6, rank=4, alpha=16, dropout=0.0, enabled=False)
    x = torch.randn(5, 8)
    assert torch.allclose(layer(x), layer.base(x))


def test_lora_base_frozen_when_enabled() -> None:
    layer = LoraLinear(8, 6, rank=4, alpha=16, dropout=0.0, enabled=True)
    assert not layer.base.weight.requires_grad
    assert layer.lora_a.requires_grad
