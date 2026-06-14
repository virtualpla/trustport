from __future__ import annotations

from typing import Literal, TypedDict

import torch

Task = Literal["ner", "re", "classification", "qa"]
ModalityName = Literal["clinical", "policy", "kbi"]
LanguageCode = Literal["en", "zh"]

TASKS: tuple[Task, ...] = ("ner", "re", "classification", "qa")
MODALITIES: tuple[ModalityName, ...] = ("clinical", "policy", "kbi")


class TaskHeads(TypedDict):
    ner: torch.Tensor
    re: torch.Tensor
    classification: torch.Tensor
    qa: torch.Tensor


class StageSignals(TypedDict):
    reference: torch.Tensor
    consistency: torch.Tensor
    schema: torch.Tensor


class PmcDimensions(TypedDict):
    policy: torch.Tensor
    measure: torch.Tensor
    content: torch.Tensor
