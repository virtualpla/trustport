from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def stow_blob(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_blob(path: str | Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"expected a mapping checkpoint at {path}")
    return blob
