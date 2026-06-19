"""Hold-out evaluation, bootstrap uncertainty, and diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from abalone.config import (
    BOOTSTRAP_SAMPLES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
)
from abalone.data import get_train_test_split
from abalone.pipeline import load_pipeline


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_rmse_ci(
    y_true: Any,
    y_pred: Any,
    n_samples: int = BOOTSTRAP_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap 95% CI for RMSE on the test set."""
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_obs = len(y_true_arr)

    boot_rmse = []
    for _ in range(n_samples):
        indices = rng.integers(0, n_obs, n_obs)
        residuals = y_true_arr[indices] - y_pred_arr[indices]
        boot_rmse.append(float(np.sqrt(np.mean(residuals**2))))

    lower, upper = np.percentile(boot_rmse, [2.5, 97.5])
    return {
        "rmse_bootstrap_mean": float(np.mean(boot_rmse)),
        "rmse_ci_95_lower": float(lower),
        "rmse_ci_95_upper": float(upper),
        "bootstrap_samples": n_samples,
    }


def residual_summary(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Summarize residual distribution for diagnostics."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
    }


def evaluate_model(
    pipeline: Pipeline,
    X_test: Any,
    y_test: Any,
) -> dict[str, Any]:
    """Evaluate a fitted pipeline on hold-out data."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "test": compute_metrics(y_test, y_pred),
        "bootstrap_rmse": bootstrap_rmse_ci(y_test, y_pred),
        "residuals": residual_summary(y_test, y_pred),
    }
    return metrics


def merge_metrics_file(evaluation: dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Merge evaluation results into the metrics JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as metrics_file:
            payload = json.load(metrics_file)
    else:
        payload = {}

    payload["evaluation"] = evaluation
    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)


def save_model_diagnostic_plots(
    y_true: Any,
    y_pred: Any,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Save predicted-vs-actual and residual diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    residuals = y_true_arr - y_pred_arr

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.5, edgecolors="none")
    min_val = min(y_true_arr.min(), y_pred_arr.min())
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)
    ax.set_xlabel("Actual age")
    ax.set_ylabel("Predicted age")
    ax.set_title("Predicted vs actual")
    _save_figure(fig, output_dir / "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred_arr, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(0.0, color="r", linestyle="--", lw=1)
    ax.set_xlabel("Predicted age")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted")
    _save_figure(fig, output_dir / "residuals_vs_predicted.png")


def run_evaluation(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load model, evaluate on test split, save metrics and diagnostic plots."""
    pipeline = load_pipeline(model_path)
    _, X_test, _, y_test = get_train_test_split()
    y_pred = pipeline.predict(X_test)

    evaluation = evaluate_model(pipeline, X_test, y_test)
    merge_metrics_file(evaluation)
    save_model_diagnostic_plots(y_test, y_pred)

    return evaluation


def main() -> None:
    """Evaluate the saved model and write metrics to disk."""
    evaluation = run_evaluation()
    test_metrics = evaluation["test"]
    ci = evaluation["bootstrap_rmse"]
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R2:   {test_metrics['r2']:.4f}")
    print(
        "RMSE 95% CI: "
        f"[{ci['rmse_ci_95_lower']:.4f}, {ci['rmse_ci_95_upper']:.4f}]"
    )
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
