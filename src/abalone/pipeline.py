"""Sklearn pipeline construction and persistence."""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from abalone.preprocess import build_preprocessor


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Create a sklearn Pipeline with preprocessing and a regressor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ],
    )


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Persist a fitted pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Pipeline:
    """Load a fitted pipeline from disk."""
    return joblib.load(path)
