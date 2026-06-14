from __future__ import annotations

from trustport.tally.integrated import integrated_f1, mean_task_f1


def test_equal_stages_returns_value() -> None:
    value = integrated_f1({"ner": 0.8, "re": 0.8, "classification": 0.8})
    assert abs(value - 0.8) < 1e-9


def test_within_min_max() -> None:
    scores = {"ner": 0.95, "re": 0.70, "classification": 0.85}
    value = integrated_f1(scores)
    assert min(scores.values()) <= value <= max(scores.values())


def test_weakest_stage_dominates() -> None:
    strong = integrated_f1({"ner": 0.95, "re": 0.90, "classification": 0.92})
    weak = integrated_f1({"ner": 0.95, "re": 0.40, "classification": 0.92})
    assert weak < strong
    assert weak < 0.70


def test_zero_stage_collapses() -> None:
    assert integrated_f1({"ner": 0.0, "re": 0.9, "classification": 0.9}) == 0.0


def test_mean_task_f1() -> None:
    assert abs(mean_task_f1({"a": 0.6, "b": 0.8}) - 0.7) < 1e-9
