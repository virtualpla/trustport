from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def icc_2_1(ratings: Sequence[Sequence[float]]) -> float:
    matrix = np.asarray(ratings, dtype=np.float64)
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    sst = float(((matrix - grand) ** 2).sum())
    ssr = float(k * ((row_means - grand) ** 2).sum())
    ssc = float(n * ((col_means - grand) ** 2).sum())
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom == 0.0:
        return 0.0
    return float((msr - mse) / denom)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    return float(np.abs(a - b).mean())


def cohen_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    categories = np.unique(np.concatenate([a, b]))
    index = {int(c): i for i, c in enumerate(categories)}
    m = len(categories)
    confusion = np.zeros((m, m), dtype=np.float64)
    for ai, bi in zip(a, b, strict=False):
        confusion[index[int(ai)], index[int(bi)]] += 1.0
    total = confusion.sum()
    observed = float(np.trace(confusion) / total)
    rows = confusion.sum(axis=1) / total
    cols = confusion.sum(axis=0) / total
    expected = float((rows * cols).sum())
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 5,
) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = conf.shape[0]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf > lo) & (conf <= hi) if lo > 0.0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(hit[mask].mean())
        ece += (count / total) * abs(avg_acc - avg_conf)
    return ece
