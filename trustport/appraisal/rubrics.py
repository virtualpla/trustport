from __future__ import annotations

from dataclasses import dataclass

from trustport.manifest import PmcConfig

DIMENSION_NAMES: tuple[str, str, str] = ("policy", "measure", "content")


@dataclass(frozen=True)
class DimensionRubric:
    name: str
    weight: float
    anchors: tuple[str, ...]


DIMENSION_WEIGHTS: dict[str, float] = {"policy": 0.35, "measure": 0.30, "content": 0.35}


def dimension_weights(config: PmcConfig) -> tuple[float, float, float]:
    return (config.alpha_policy, config.alpha_measure, config.alpha_content)


def rubric_table(config: PmcConfig) -> tuple[DimensionRubric, ...]:
    return (
        DimensionRubric("policy", config.alpha_policy, ("implementation feasibility",)),
        DimensionRubric("measure", config.alpha_measure, ("specific interventions",)),
        DimensionRubric("content", config.alpha_content, ("technical quality",)),
    )
