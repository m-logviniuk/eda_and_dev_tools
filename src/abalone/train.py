"""Model training and cross-validated model selection."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from abalone.config import CV_FOLDS, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from abalone.data import get_train_test_split
from abalone.pipeline import build_pipeline, save_pipeline


def get_candidate_estimators() -> dict[str, Any]:
    """Return named candidate regressors for model comparison."""
    return {
        "ridge": Ridge(random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cross_validate_candidate(
    name: str,
    estimator: Any,
    X_train: Any,
    y_train: Any,
) -> dict[str, float]:
    """Run cross-validation and return RMSE summary statistics."""
    pipeline = build_pipeline(estimator)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -scores["test_score"]
    return {
        "name": name,
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "rmse_folds": [float(value) for value in rmse_scores],
    }


def select_best_candidate(cv_results: list[dict[str, float]]) -> str:
    """Select the candidate with the lowest mean CV RMSE."""
    return min(cv_results, key=lambda row: row["rmse_mean"])["name"]


def train_and_select_model() -> tuple[Pipeline, dict[str, Any]]:
    """Compare candidates, fit the winner, and persist the pipeline."""
    X_train, X_test, y_train, y_test = get_train_test_split()

    candidates = get_candidate_estimators()
    cv_results = [
        cross_validate_candidate(name, estimator, X_train, y_train)
        for name, estimator in candidates.items()
    ]
    best_name = select_best_candidate(cv_results)

    best_pipeline = build_pipeline(candidates[best_name])
    best_pipeline.fit(X_train, y_train)
    save_pipeline(best_pipeline, MODEL_PATH)

    training_summary = {
        "selected_model": best_name,
        "cv_folds": CV_FOLDS,
        "cv_results": {
            row["name"]: {
                "rmse_mean": row["rmse_mean"],
                "rmse_std": row["rmse_std"],
                "rmse_folds": row["rmse_folds"],
            }
            for row in cv_results
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump({"training": training_summary}, metrics_file, indent=2)

    return best_pipeline, {
        "X_test": X_test,
        "y_test": y_test,
        "training_summary": training_summary,
    }


def main() -> None:
    """Train models, select the best candidate, and save the pipeline."""
    pipeline, info = train_and_select_model()
    selected = info["training_summary"]["selected_model"]
    cv_rmse = info["training_summary"]["cv_results"][selected]["rmse_mean"]
    print(f"Selected model: {selected}")
    print(f"CV RMSE (mean): {cv_rmse:.4f}")
    print(f"Saved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
