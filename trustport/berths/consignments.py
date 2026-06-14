from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from trustport.manifest import Consignment
from trustport.quay.types import MODALITIES


@dataclass
class BatchTensors:
    token_ids: torch.Tensor
    ner_labels: torch.Tensor
    relation: torch.Tensor
    category: torch.Tensor
    answer: torch.Tensor
    pmc_targets: torch.Tensor
    kbi: torch.Tensor
    modality: torch.Tensor

    def to(self, device: torch.device) -> BatchTensors:
        return BatchTensors(
            token_ids=self.token_ids.to(device),
            ner_labels=self.ner_labels.to(device),
            relation=self.relation.to(device),
            category=self.category.to(device),
            answer=self.answer.to(device),
            pmc_targets=self.pmc_targets.to(device),
            kbi=self.kbi.to(device),
            modality=self.modality.to(device),
        )

    @property
    def size(self) -> int:
        return int(self.token_ids.shape[0])


def collate(items: Sequence[Consignment]) -> BatchTensors:
    modality_index = {name: idx for idx, name in enumerate(MODALITIES)}
    token_ids = torch.tensor([c.token_ids for c in items], dtype=torch.long)
    ner_labels = torch.tensor([c.ner_labels for c in items], dtype=torch.long)
    relation = torch.tensor([c.relation for c in items], dtype=torch.long)
    category = torch.tensor([c.category for c in items], dtype=torch.long)
    answer = torch.tensor([c.answer for c in items], dtype=torch.long)
    pmc_targets = torch.tensor([c.pmc_targets for c in items], dtype=torch.float32)
    kbi = torch.tensor([c.kbi for c in items], dtype=torch.float32)
    modality = torch.tensor([modality_index[c.modality] for c in items], dtype=torch.long)
    return BatchTensors(
        token_ids=token_ids,
        ner_labels=ner_labels,
        relation=relation,
        category=category,
        answer=answer,
        pmc_targets=pmc_targets,
        kbi=kbi,
        modality=modality,
    )
