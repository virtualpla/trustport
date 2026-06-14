from __future__ import annotations

import random

import numpy as np
import torch


def fix_entropy(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_seed(seed: int, salt: str) -> int:
    acc = seed & 0xFFFFFFFF
    for ch in salt:
        acc = (acc * 1099511628211 + ord(ch)) & 0xFFFFFFFF
    return acc
