from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

from trustport.manifest import (
    BackboneConfig,
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    OptimConfig,
    PmcConfig,
    TaskConfig,
    Toggles,
    TrustConfig,
)

T = TypeVar("T")


def _coerce(cls: type[T], mapping: Any) -> T:
    if not isinstance(mapping, dict):
        return cls()
    names = {f.name for f in fields(cast(Any, cls))}
    kwargs = {k: v for k, v in mapping.items() if k in names}
    return cls(**kwargs)


def materialize(data: dict[str, Any]) -> ExperimentConfig:
    top: dict[str, Any] = {k: data[k] for k in ("name", "seed", "format_gamma") if k in data}
    if "init_task_weights" in data:
        w = [float(v) for v in data["init_task_weights"]]
        top["init_task_weights"] = (w[0], w[1], w[2], w[3])
    return ExperimentConfig(
        backbone=_coerce(BackboneConfig, data.get("backbone")),
        tasks=_coerce(TaskConfig, data.get("tasks")),
        pmc=_coerce(PmcConfig, data.get("pmc")),
        fusion=_coerce(FusionConfig, data.get("fusion")),
        trust=_coerce(TrustConfig, data.get("trust")),
        optim=_coerce(OptimConfig, data.get("optim")),
        data=_coerce(DataConfig, data.get("data")),
        toggles=_coerce(Toggles, data.get("toggles")),
        **top,
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    here = Path(path)
    raw = OmegaConf.load(str(here))
    parent = raw.get("extends") if isinstance(raw, DictConfig) else None
    if parent is not None:
        base = OmegaConf.load(str(here.parent / str(parent)))
        raw = OmegaConf.merge(base, raw)
    data = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"experiment config at {path} must be a mapping")
    typed = {str(k): v for k, v in data.items()}
    return materialize(typed)
