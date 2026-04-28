"""Tests pour ``aphid_spatial.evaluation.metrics``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from aphid_spatial.evaluation.metrics import (
    auc_pr,
    auc_roc,
    brier,
    calibration_curve_data,
    evaluate_all,
    log_loss_clipped,
    mae_prob,
    rmse_prob,
)


def test_brier_perfect() -> None:
    y = np.array([0, 1, 0, 1, 1])
    assert brier(y, y.astype(float)) == pytest.approx(0.0)


def test_brier_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.uniform(size=100)
    assert brier(y, p) == pytest.approx(skm.brier_score_loss(y, p))


def test_auc_perfect_prediction() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.1, 0.9, 1.0])
    assert auc_roc(y, p) == pytest.approx(1.0)
    assert auc_pr(y, p) == pytest.approx(1.0)


def test_auc_single_class_returns_nan() -> None:
    y = np.zeros(10, dtype=int)
    p = np.linspace(0, 1, 10)
    assert np.isnan(auc_roc(y, p))
    assert np.isnan(auc_pr(y, p))


def test_log_loss_handles_extremes() -> None:
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    # Sans clipping : log(0) = -inf. Avec clipping, valeur finie
    val = log_loss_clipped(y, p)
    assert np.isfinite(val)


def test_log_loss_matches_sklearn_when_safe() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=50)
    p = rng.uniform(0.05, 0.95, size=50)
    sk = skm.log_loss(y, p, labels=[0, 1])
    assert log_loss_clipped(y, p) == pytest.approx(sk, rel=1e-6)


def test_mae_rmse_zero_when_equal() -> None:
    p = np.array([0.1, 0.5, 0.9])
    assert mae_prob(p, p) == 0.0
    assert rmse_prob(p, p) == 0.0


def test_calibration_curve_shapes() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(size=200)
    out = calibration_curve_data(y, p, n_bins=10)
    assert out["bin_edges"].shape == (11,)
    assert out["mean_pred"].shape == (10,)
    assert out["frac_pos"].shape == (10,)
    assert out["count"].shape == (10,)
    assert int(out["count"].sum()) == 200


def test_evaluate_all_keys() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p_pred = rng.uniform(size=200)
    p_true = rng.uniform(size=200)
    out = evaluate_all(y, p_pred, p_true)
    for key in ("auc_roc", "auc_pr", "brier", "log_loss", "mae_prob", "rmse_prob"):
        assert key in out
