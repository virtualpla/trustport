from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trustport.manifest import Consignment
from trustport.quay.beacon import derived_seed
from trustport.quay.types import MODALITIES, LanguageCode

DATASETS: dict[str, dict[str, object]] = {
    "bc5cdr": {"language": "en", "task": "ner", "classes": 2, "source": "BioCreative V"},
    "n2c2": {"language": "en", "task": "ner", "classes": 16, "source": "n2c2 2022 track-1"},
    "chemprot": {"language": "en", "task": "re", "classes": 13, "source": "BioCreative VI"},
    "medqa": {"language": "en", "task": "qa", "classes": 4, "source": "MedBench"},
    "climedbench": {"language": "zh", "task": "qa", "classes": 14, "source": "CliMedBench"},
    "ccks17": {"language": "zh", "task": "ner", "classes": 5, "source": "CCKS-2017"},
    "ccks19": {"language": "zh", "task": "ner", "classes": 6, "source": "CCKS-2019"},
    "policy": {"language": "zh", "task": "scoring", "classes": 3, "source": "NHC gazette"},
}


@dataclass
class SyntheticArrivals:
    seed: int
    seq_len: int
    vocab_size: int
    ner_tags: int
    classes: int
    relations: int
    answers: int
    kbi_dim: int

    def cohort(self, n: int) -> list[Consignment]:
        rng = np.random.default_rng(self.seed)
        step = max(1, (self.vocab_size // self.ner_tags) // max(1, self.classes))
        projection = rng.standard_normal((3, self.kbi_dim)) / np.sqrt(self.kbi_dim)
        items: list[Consignment] = []
        for i in range(n):
            theme = int(rng.integers(0, self.classes))
            ner = rng.integers(0, self.ner_tags, size=self.seq_len)
            jitter = rng.integers(0, step, size=self.seq_len)
            mult = theme * step + jitter
            tokens = (ner + self.ner_tags * mult).astype(np.int64)
            tokens = np.clip(tokens, 0, self.vocab_size - 1)
            kbi = rng.standard_normal(self.kbi_dim) + float(theme)
            logits = projection @ kbi
            pmc = 1.0 / (1.0 + np.exp(-logits))
            modality = MODALITIES[i % len(MODALITIES)]
            language: LanguageCode = "zh" if theme % 2 == 0 else "en"
            items.append(
                Consignment(
                    doc_id=f"doc-{derived_seed(self.seed, str(i)):08x}",
                    language=language,
                    modality=modality,
                    token_ids=tuple(int(t) for t in tokens),
                    ner_labels=tuple(int(t) % self.ner_tags for t in tokens),
                    relation=int((theme * 5 + 1) % self.relations),
                    category=theme,
                    answer=int(theme % self.answers),
                    pmc_targets=(float(pmc[0]), float(pmc[1]), float(pmc[2])),
                    kbi=tuple(float(v) for v in kbi),
                )
            )
        return items


def load_manifest(path: str | Path) -> list[Consignment]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(blob, list):
        raise ValueError("manifest must be a JSON array of records")
    items: list[Consignment] = []
    for row in blob:
        pt = [float(v) for v in row["pmc_targets"]]
        items.append(
            Consignment(
                doc_id=str(row["doc_id"]),
                language=row["language"],
                modality=row["modality"],
                token_ids=tuple(int(t) for t in row["token_ids"]),
                ner_labels=tuple(int(t) for t in row["ner_labels"]),
                relation=int(row["relation"]),
                category=int(row["category"]),
                answer=int(row["answer"]),
                pmc_targets=(pt[0], pt[1], pt[2]),
                kbi=tuple(float(v) for v in row["kbi"]),
            )
        )
    return items
