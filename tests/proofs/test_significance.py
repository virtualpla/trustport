from __future__ import annotations

import numpy as np

from trustport.tally.significance import (
    bca_bootstrap_ci,
    benjamini_hochberg,
    cohens_d,
    delong_test,
    holm_bonferroni,
    permutation_test,
)


def test_cohens_d_sign() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)
    assert cohens_d(a.tolist(), b.tolist()) > 0.5


def test_delong_orders_aucs() -> None:
    rng = np.random.default_rng(2)
    labels = [0] * 50 + [1] * 50
    strong = [rng.normal(0.0, 1.0) for _ in range(50)] + [rng.normal(2.0, 1.0) for _ in range(50)]
    weak = [rng.normal(0.0, 1.0) for _ in range(50)] + [rng.normal(0.5, 1.0) for _ in range(50)]
    auc_a, auc_b, p = delong_test(labels, strong, weak)
    assert auc_a > auc_b
    assert 0.0 <= p <= 1.0


def test_delong_identical_high_p() -> None:
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.2, 0.1, 0.8, 0.9, 0.3, 0.7]
    _, _, p = delong_test(labels, scores, scores)
    assert p > 0.99


def test_bootstrap_brackets_mean() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(5.0, 1.0, size=200)
    lo, hi = bca_bootstrap_ci(data.tolist(), lambda s: float(np.mean(s)), n_boot=400, seed=1)
    assert lo < 5.0 < hi


def test_holm_rejects_small() -> None:
    flags = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
    assert flags[0] is True and flags[2] is False


def test_bh_monotone() -> None:
    flags = benjamini_hochberg([0.001, 0.01, 0.2, 0.9], alpha=0.05)
    assert flags[0] is True and flags[3] is False


def test_permutation_ranges() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 1.0, size=60)
    same = permutation_test(base.tolist(), base.tolist(), n_perm=500, seed=0)
    shifted = permutation_test((base + 2.0).tolist(), base.tolist(), n_perm=500, seed=0)
    assert same > 0.5
    assert shifted < 0.05
