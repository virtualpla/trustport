from __future__ import annotations

import torch

from trustport.inspection.clearance import flag_for_review, stage_detection, trust_score
from trustport.inspection.holds import hallucination_score, self_consistency


def test_hallucination_zero_when_grounded() -> None:
    ones = torch.ones(4)
    assert torch.allclose(hallucination_score(ones, ones, ones), torch.zeros(4))


def test_hallucination_in_unit_interval() -> None:
    h = hallucination_score(torch.rand(20), torch.rand(20), torch.rand(20))
    assert torch.all(h >= 0.0) and torch.all(h <= 1.0)


def test_trust_multiplicative_decay() -> None:
    h = torch.full((5, 3), 0.2)
    c = torch.full((5, 3), 0.9)
    short = trust_score(h[:2], c[:2])
    full = trust_score(h, c)
    assert torch.all(full <= short + 1e-6)


def test_flagging_threshold() -> None:
    trust = torch.tensor([0.4, 0.7, 0.55, 0.9])
    flags = flag_for_review(trust, tau=0.6)
    assert flags.tolist() == [True, False, True, False]


def test_self_consistency_levels() -> None:
    unanimous = self_consistency(torch.zeros(3, 2, dtype=torch.long))
    assert torch.allclose(unanimous, torch.ones(2))
    split = self_consistency(torch.tensor([[0], [0], [1], [1]]))
    assert abs(float(split[0].item()) - 0.5) < 1e-6


def test_stage_detection_counts() -> None:
    errors = torch.tensor([1, 0, 1, 0])
    flags = torch.tensor([1, 0, 0, 1])
    report = stage_detection(errors, flags)
    assert abs(report["detection_rate"] - 0.5) < 1e-6
    assert abs(report["false_positive_rate"] - 0.5) < 1e-6
