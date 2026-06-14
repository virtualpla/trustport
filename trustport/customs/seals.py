from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from trustport.quay.types import TASKS, TaskHeads

TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["entities", "relation", "category", "answer"],
    "properties": {
        "entities": {"type": "array", "item_min": 0, "item_max": 64},
        "relation": {"type": "integer", "min": 0, "max": 64},
        "category": {"type": "integer", "min": 0, "max": 64},
        "answer": {"type": "integer", "min": 0, "max": 64},
    },
}


def validate_against_schema(obj: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            return ["root is not an object"]
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"missing required field {key}")
        props: dict[str, Any] = schema.get("properties", {})
        for key, spec in props.items():
            if key not in obj:
                continue
            errors.extend(_check_field(key, obj[key], spec))
    return errors


def _check_field(key: str, value: Any, spec: dict[str, Any]) -> list[str]:
    kind = spec.get("type")
    out: list[str] = []
    if kind == "integer":
        if not isinstance(value, int):
            out.append(f"{key} is not an integer")
            return out
        if "min" in spec and value < spec["min"]:
            out.append(f"{key} below minimum")
        if "max" in spec and value > spec["max"]:
            out.append(f"{key} above maximum")
    elif kind == "array":
        if not isinstance(value, list):
            out.append(f"{key} is not an array")
            return out
        if "item_min" in spec and len(value) < spec["item_min"]:
            out.append(f"{key} too short")
        if "item_max" in spec and len(value) > spec["item_max"]:
            out.append(f"{key} too long")
    return out


def assemble_record(heads: TaskHeads) -> list[dict[str, Any]]:
    ner = heads["ner"].argmax(dim=-1)
    relation = heads["re"].argmax(dim=-1)
    category = heads["classification"].argmax(dim=-1)
    answer = heads["qa"].argmax(dim=-1)
    records: list[dict[str, Any]] = []
    for i in range(ner.shape[0]):
        entities = [int(t) for t in ner[i].tolist() if int(t) != 0]
        records.append(
            {
                "entities": entities,
                "relation": int(relation[i].item()),
                "category": int(category[i].item()),
                "answer": int(answer[i].item()),
            }
        )
    return records


def schema_conformance(records: list[dict[str, Any]], schema: dict[str, Any]) -> torch.Tensor:
    flags = [1.0 if not validate_against_schema(r, schema) else 0.0 for r in records]
    return torch.tensor(flags, dtype=torch.float32)


def format_penalty(heads: TaskHeads) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for task in TASKS:
        logits = heads[task]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        terms.append(entropy.mean())
    return torch.stack(terms).mean()


def enforcement_gain(f1_with: float, f1_without: float) -> float:
    return f1_with - f1_without
