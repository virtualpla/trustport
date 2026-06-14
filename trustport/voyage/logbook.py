from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from trustport.quay.beacon import fix_entropy
from trustport.quay.cargo_io import load_blob, stow_blob


def write_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    seed: int,
    path: str | Path,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "seed": seed,
    }
    stow_blob(payload, path)


def restore_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    path: str | Path,
) -> dict[str, Any]:
    blob = load_blob(path)
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    fix_entropy(int(blob["seed"]))
    return blob
