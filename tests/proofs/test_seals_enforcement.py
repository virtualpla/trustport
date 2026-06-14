from __future__ import annotations

import torch

from trustport.customs.seals import (
    TASK_SCHEMA,
    assemble_record,
    format_penalty,
    schema_conformance,
    validate_against_schema,
)
from trustport.quay.types import TaskHeads


def _heads(peak: float) -> TaskHeads:
    base = torch.zeros(2, 4)
    base[:, 0] = peak
    ner = torch.zeros(2, 5, 5)
    ner[:, :, 0] = peak
    return TaskHeads(ner=ner, re=base.clone(), classification=base.clone(), qa=base.clone())


def test_valid_record_passes() -> None:
    record = {"entities": [1, 2], "relation": 3, "category": 1, "answer": 0}
    assert validate_against_schema(record, TASK_SCHEMA) == []


def test_missing_field_flagged() -> None:
    errors = validate_against_schema({"entities": []}, TASK_SCHEMA)
    assert any("relation" in e for e in errors)


def test_out_of_range_flagged() -> None:
    record = {"entities": [], "relation": 999, "category": 0, "answer": 0}
    assert any("relation" in e for e in validate_against_schema(record, TASK_SCHEMA))


def test_format_penalty_lower_for_confident() -> None:
    confident = format_penalty(_heads(8.0))
    diffuse = format_penalty(_heads(0.0))
    assert confident.item() < diffuse.item()
    assert confident.item() >= 0.0


def test_assembled_records_conform() -> None:
    records = assemble_record(_heads(8.0))
    flags = schema_conformance(records, TASK_SCHEMA)
    assert torch.all(flags == 1.0)
