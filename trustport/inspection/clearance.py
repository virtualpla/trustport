from __future__ import annotations

from typing import TypedDict

import torch


class StageReport(TypedDict):
    error_rate: float
    detection_rate: float
    false_positive_rate: float


def trust_score(stage_h: torch.Tensor, stage_c: torch.Tensor) -> torch.Tensor:
    contributions = (1.0 - stage_h) * stage_c
    product: torch.Tensor = contributions.clamp(0.0, 1.0).prod(dim=0)
    return product


def flag_for_review(trust: torch.Tensor, tau: float = 0.6) -> torch.Tensor:
    return trust < tau


def stage_detection(error_flags: torch.Tensor, review_flags: torch.Tensor) -> StageReport:
    errors = error_flags.bool()
    flagged = review_flags.bool()
    total = error_flags.shape[0]
    error_count = int(errors.sum().item())
    correct_count = total - error_count
    detected = int((errors & flagged).sum().item())
    false_positive = int((~errors & flagged).sum().item())
    detection_rate = detected / error_count if error_count else 0.0
    false_positive_rate = false_positive / correct_count if correct_count else 0.0
    return StageReport(
        error_rate=error_count / total if total else 0.0,
        detection_rate=detection_rate,
        false_positive_rate=false_positive_rate,
    )
