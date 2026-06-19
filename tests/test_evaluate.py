"""Tests for evaluation metrics."""

import numpy as np

from abalone.evaluate import (
    bootstrap_rmse_ci,
    compute_metrics,
    residual_summary,
)


def test_compute_metrics_perfect_prediction() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_true)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_bootstrap_rmse_ci_bounds() -> None:
    y_true = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    y_pred = np.array([7.5, 9.5, 10.5, 10.0, 12.5])
    ci = bootstrap_rmse_ci(y_true, y_pred, n_samples=200, random_state=42)
    assert ci["rmse_ci_95_lower"] <= ci["rmse_bootstrap_mean"]
    assert ci["rmse_ci_95_upper"] >= ci["rmse_bootstrap_mean"]


def test_residual_summary() -> None:
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([9.0, 11.5, 11.0])
    summary = residual_summary(y_true, y_pred)
    assert summary["mean"] == 0.5
    assert summary["std"] > 0
