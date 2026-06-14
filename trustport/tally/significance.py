from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    nx, ny = x.shape[0], y.shape[0]
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    pooled_std = float(np.sqrt(pooled_var))
    if pooled_std == 0.0:
        return 0.0
    return float(x.mean() - y.mean()) / pooled_std


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def delong_test(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> tuple[float, float, float]:
    truth = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-truth)
    positives = int(truth.sum())
    stacked = np.vstack(
        (np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64))
    )[:, order]
    m = positives
    n = stacked.shape[1] - m
    pos = stacked[:, :m]
    neg = stacked[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(stacked[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0.0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def bca_bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    sample = np.asarray(data, dtype=np.float64)
    n = sample.shape[0]
    rng = np.random.default_rng(seed)
    theta_hat = statistic(sample)
    replicates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        replicates[i] = statistic(sample[idx])
    proportion = float((replicates < theta_hat).mean())
    proportion = min(max(proportion, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = float(stats.norm.ppf(proportion))
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = statistic(np.delete(sample, i))
    jack_mean = jack.mean()
    num = float(((jack_mean - jack) ** 3).sum())
    den = float(6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5))
    accel = num / den if den != 0.0 else 0.0
    z_lo = float(stats.norm.ppf(alpha / 2.0))
    z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))
    a_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    lo = float(np.quantile(replicates, float(a_lo)))
    hi = float(np.quantile(replicates, float(a_hi)))
    return lo, hi


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    values = list(pvalues)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if values[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    values = list(pvalues)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    reject = [False] * m
    max_rank = -1
    for rank, idx in enumerate(order, start=1):
        if values[idx] <= alpha * rank / m:
            max_rank = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_rank:
            reject[idx] = True
    return reject


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    diff = x - y
    observed = float(abs(diff.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.shape[0])
        if abs(float((signs * diff).mean())) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)
