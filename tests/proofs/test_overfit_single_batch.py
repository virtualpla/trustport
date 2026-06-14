from __future__ import annotations

from pathlib import Path

import torch

from trustport.berths.arrivals import SyntheticArrivals
from trustport.berths.consignments import collate
from trustport.harbor import build_harbor
from trustport.quay.beacon import fix_entropy
from trustport.voyage.helmsman import Helmsman
from trustport.voyage.objective import MultiTaskObjective
from trustport.wheelhouse.settings import load_experiment

ROOT = Path(__file__).resolve().parents[2]


def test_overfits_single_batch() -> None:
    config = load_experiment(ROOT / "configs" / "experiment" / "_smoke.yaml")
    fix_entropy(config.seed)
    arrivals = SyntheticArrivals(
        seed=config.seed,
        seq_len=config.data.seq_len,
        vocab_size=config.backbone.vocab_size,
        ner_tags=config.tasks.ner_tags,
        classes=config.tasks.classes,
        relations=config.tasks.relations,
        answers=config.tasks.answers,
        kbi_dim=config.fusion.kbi_dim,
    )
    batch = collate(arrivals.cohort(8))
    model = build_harbor(config)
    objective = MultiTaskObjective(
        config.init_task_weights, config.format_gamma, config.toggles.struct_out
    )
    helmsman = Helmsman(model, objective, config.optim, total_steps=300)
    for _ in range(300):
        helmsman.step(batch)
    model.eval()
    with torch.no_grad():
        pred = model(batch)["heads"]["ner"].argmax(dim=-1)
    accuracy = float((pred == batch.ner_labels).float().mean().item())
    assert accuracy > 0.9
