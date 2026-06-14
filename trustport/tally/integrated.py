from __future__ import annotations

from collections.abc import Mapping

STAGE_WEIGHTS: dict[str, float] = {"ner": 0.35, "re": 0.40, "classification": 0.25}


def integrated_f1(
    f1_by_stage: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    table = dict(STAGE_WEIGHTS if weights is None else weights)
    stages = [s for s in table if s in f1_by_stage]
    total_weight = sum(table[s] for s in stages)
    if total_weight <= 0.0:
        raise ValueError("stage weights must sum to a positive value")
    accumulator = 0.0
    for stage in stages:
        score = f1_by_stage[stage]
        if score <= 0.0:
            return 0.0
        accumulator += (table[stage] / total_weight) / score
    return 1.0 / accumulator


def mean_task_f1(f1_by_stage: Mapping[str, float]) -> float:
    if not f1_by_stage:
        return 0.0
    return sum(f1_by_stage.values()) / len(f1_by_stage)
