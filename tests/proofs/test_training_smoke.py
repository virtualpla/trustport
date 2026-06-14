from __future__ import annotations

from pathlib import Path

import torch

from trustport.berths.arrivals import SyntheticArrivals
from trustport.berths.stowage import iter_batches
from trustport.harbor import build_harbor
from trustport.quay.beacon import fix_entropy
from trustport.voyage.helmsman import Helmsman
from trustport.voyage.logbook import restore_checkpoint, write_checkpoint
from trustport.voyage.objective import MultiTaskObjective
from trustport.wheelhouse.settings import load_experiment

ROOT = Path(__file__).resolve().parents[2]


def _smoke_batches(config) -> list:
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
    return list(iter_batches(arrivals.cohort(config.data.documents), config.optim.batch_size, True))


def test_smoke_loss_decreases() -> None:
    config = load_experiment(ROOT / "configs" / "experiment" / "_smoke.yaml")
    fix_entropy(config.seed)
    model = build_harbor(config)
    objective = MultiTaskObjective(
        config.init_task_weights, config.format_gamma, config.toggles.struct_out
    )
    helmsman = Helmsman(model, objective, config.optim, total_steps=24)
    history = helmsman.fit(_smoke_batches(config), 24)
    assert history[-1] < history[0]


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = load_experiment(ROOT / "configs" / "experiment" / "_smoke.yaml")
    fix_entropy(config.seed)
    model = build_harbor(config)
    objective = MultiTaskObjective(
        config.init_task_weights, config.format_gamma, config.toggles.struct_out
    )
    helmsman = Helmsman(model, objective, config.optim, total_steps=4)
    helmsman.fit(_smoke_batches(config), 4)
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(model, helmsman.optimizer, 4, config.seed, path)
    fresh = build_harbor(config)
    blob = restore_checkpoint(fresh, None, path)
    assert blob["seed"] == config.seed
    batch = _smoke_batches(config)[0]
    model.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch)["pmc"], fresh(batch)["pmc"])
