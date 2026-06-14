from __future__ import annotations

import numpy as np

from trustport.tally.agreement import (
    cohen_kappa,
    expected_calibration_error,
    icc_2_1,
    mae,
    pearson_r,
)


def test_icc_high_agreement() -> None:
    rng = np.random.default_rng(0)
    subjects = rng.normal(size=(30, 1))
    ratings = np.hstack([subjects + rng.normal(scale=0.02, size=(30, 1)) for _ in range(3)])
    assert icc_2_1(ratings.tolist()) > 0.9


def test_pearson_perfect() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    assert abs(pearson_r(x, [2.0, 4.0, 6.0, 8.0]) - 1.0) < 1e-9


def test_mae_basic() -> None:
    assert abs(mae([1.0, 2.0], [1.5, 2.5]) - 0.5) < 1e-9


def test_kappa_perfect_and_chance() -> None:
    assert abs(cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) - 1.0) < 1e-9
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, size=400)
    b = rng.integers(0, 3, size=400)
    assert abs(cohen_kappa(a.tolist(), b.tolist())) < 0.15


def test_ece_calibrated_small() -> None:
    conf = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
    correct = [1 if np.random.default_rng(i).random() < c else 0 for i, c in enumerate(conf)]
    value = expected_calibration_error(conf, correct, n_bins=5)
    assert 0.0 <= value <= 1.0
