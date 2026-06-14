from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from trustport.berths.arrivals import SyntheticArrivals
from trustport.berths.consignments import BatchTensors
from trustport.berths.stowage import iter_batches
from trustport.harbor import Harbor, build_harbor
from trustport.inspection.clearance import flag_for_review, trust_score
from trustport.manifest import Consignment, ExperimentConfig
from trustport.quay.beacon import fix_entropy
from trustport.quay.moorings import pick_device
from trustport.quay.signals import get_logger
from trustport.tally.integrated import integrated_f1
from trustport.voyage.helmsman import Helmsman
from trustport.voyage.logbook import restore_checkpoint, write_checkpoint
from trustport.voyage.objective import MultiTaskObjective
from trustport.wheelhouse.export import export_onnx
from trustport.wheelhouse.settings import load_experiment

_LOG = get_logger("wheelhouse")


@dataclass
class TrainArgs:
    config: str = "configs/experiment/main.yaml"
    out: str = "runs/main"
    steps: int = 50
    seed: int = 42
    device: str = "auto"


@dataclass
class AppraiseArgs:
    config: str = "configs/experiment/main.yaml"
    documents: int = 8
    device: str = "auto"


@dataclass
class FuseArgs:
    config: str = "configs/experiment/main.yaml"
    documents: int = 8
    device: str = "auto"


@dataclass
class InspectArgs:
    config: str = "configs/experiment/main.yaml"
    documents: int = 32
    tau: float = 0.6
    device: str = "auto"


@dataclass
class EvaluateArgs:
    config: str = "configs/experiment/main.yaml"
    checkpoint: str = ""
    device: str = "auto"


@dataclass
class ExportArgs:
    config: str = "configs/experiment/main.yaml"
    out: str = "runs/main/ner.onnx"
    device: str = "auto"


def _cohort(config: ExperimentConfig, count: int | None = None) -> list[Consignment]:
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
    return arrivals.cohort(config.data.documents if count is None else count)


def _batches(
    config: ExperimentConfig, device: torch.device, count: int | None = None
) -> list[BatchTensors]:
    cohort = _cohort(config, count)
    return [b.to(device) for b in iter_batches(cohort, config.optim.batch_size, drop_last=True)]


def _stage_f1(model: Harbor, batches: list[BatchTensors]) -> dict[str, float]:
    model.eval()
    hits = {"ner": 0.0, "re": 0.0, "classification": 0.0}
    seen = 0
    with torch.no_grad():
        for batch in batches:
            out = model(batch)
            ner_pred = out["heads"]["ner"].argmax(dim=-1)
            hits["ner"] += float((ner_pred == batch.ner_labels).float().mean().item()) * batch.size
            hits["re"] += float((out["heads"]["re"].argmax(dim=-1) == batch.relation).sum().item())
            hits["classification"] += float(
                (out["heads"]["classification"].argmax(dim=-1) == batch.category).sum().item()
            )
            seen += batch.size
    return {key: (value / seen if seen else 0.0) for key, value in hits.items()}


def run_train(args: TrainArgs) -> list[float]:
    config = load_experiment(args.config).with_seed(args.seed)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    objective = MultiTaskObjective(
        config.init_task_weights, config.format_gamma, config.toggles.struct_out
    ).to(device)
    batches = _batches(config, device)
    helmsman = Helmsman(model, objective, config.optim, total_steps=args.steps)
    history = helmsman.fit(batches, args.steps)
    write_checkpoint(
        model, helmsman.optimizer, args.steps, config.seed, Path(args.out) / "checkpoint.pt"
    )
    weights = objective.normalized_weights()
    _LOG.info("trained steps=%d final_loss=%.4f weights=%s", args.steps, history[-1], weights)
    return history


def run_appraise(args: AppraiseArgs) -> torch.Tensor:
    config = load_experiment(args.config)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    batches = _batches(config, device, count=args.documents)
    model.eval()
    with torch.no_grad():
        scores: torch.Tensor = model(batches[0])["pmc"]
    _LOG.info("pmc_index mean=%.2f", float(scores.mean().item()))
    return scores


def run_fuse(args: FuseArgs) -> torch.Tensor:
    config = load_experiment(args.config)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    batches = _batches(config, device, count=args.documents)
    model.eval()
    with torch.no_grad():
        pathways: torch.Tensor = model(batches[0])["pathways"]
    _LOG.info("pathway clusters=%s", pathways.tolist())
    return pathways


def run_inspect(args: InspectArgs) -> float:
    config = load_experiment(args.config)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    batches = _batches(config, device, count=args.documents)
    model.eval()
    with torch.no_grad():
        out = model(batches[0])
    rows = out["heads"]["ner"].shape[0]
    stages = torch.rand(5, rows)
    consistency = torch.rand(5, rows).clamp(0.5, 1.0)
    trust = trust_score(stages * 0.2, consistency)
    flagged = flag_for_review(trust, args.tau)
    rate = float(flagged.float().mean().item())
    _LOG.info("flagged_fraction=%.3f", rate)
    return rate


def run_evaluate(args: EvaluateArgs) -> float:
    config = load_experiment(args.config)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    if args.checkpoint:
        restore_checkpoint(model, None, args.checkpoint)
    batches = _batches(config, device)
    scores = _stage_f1(model, batches)
    value = integrated_f1(scores)
    _LOG.info("integrated_f1=%.4f stages=%s", value, scores)
    return value


def run_export(args: ExportArgs) -> str:
    config = load_experiment(args.config)
    fix_entropy(config.seed)
    device = pick_device(args.device)
    model = build_harbor(config).to(device)
    sample = _batches(config, device, count=config.optim.batch_size)[0].token_ids
    export_onnx(model, sample, args.out)
    _LOG.info("exported onnx to %s", args.out)
    return args.out
