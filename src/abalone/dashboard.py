"""ExplainerDashboard configuration and serving."""

from __future__ import annotations

from pathlib import Path

import dill
import numpy as np
import pandas as pd
from explainerdashboard import ExplainerDashboard, RegressionExplainer
from explainerdashboard.explainers import BaseExplainer

from abalone.config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_YAML_PATH,
    EXPLAINER_PATH,
    EXPLAINER_SAMPLE_SIZE,
    MODEL_PATH,
)
from abalone.data import get_train_test_split
from abalone.pipeline import load_pipeline

DASHBOARD_TITLE = "Abalone Age Prediction"
DASHBOARD_DESCRIPTION = (
    "Interactive exploration of the fitted regression pipeline "
    "for predicting abalone age from physical measurements."
)
DASHBOARD_KWARGS = {
    "shap_interaction": False,
    "decision_trees": False,
}


def _sample_explainer_data(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int = EXPLAINER_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= sample_size:
        return X, y
    sampled = X.sample(n=sample_size, random_state=42)
    return sampled, y.loc[sampled.index]


def _prepare_explainer_inputs(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return writable copies suitable for sklearn metrics and SHAP."""
    return X.copy(), np.asarray(y, dtype=np.float64).copy()


def _normalize_explainer_targets(explainer: BaseExplainer) -> None:
    """
    Convert target arrays to writable numpy arrays.

    Pandas Series loaded from dill can trigger sklearn metrics errors on recent
    sklearn versions (read-only WRITEABLE flag).
    """
    explainer.y = np.asarray(explainer.y, dtype=np.float64).copy()


def _build_explainer(
    model_path: Path = MODEL_PATH,
) -> RegressionExplainer:
    """Fit a RegressionExplainer on a sample of the hold-out test set."""
    pipeline = load_pipeline(model_path)
    _, X_test, _, y_test = get_train_test_split()
    X_sample, y_sample = _sample_explainer_data(X_test, y_test)
    X_sample, y_array = _prepare_explainer_inputs(X_sample, y_sample)
    explainer = RegressionExplainer(pipeline, X_sample, y_array)
    _normalize_explainer_targets(explainer)
    return explainer


def _load_explainer(explainer_path: Path = EXPLAINER_PATH) -> RegressionExplainer:
    """Load a persisted explainer and normalize target arrays."""
    with explainer_path.open("rb") as explainer_file:
        explainer = dill.load(explainer_file)
    _normalize_explainer_targets(explainer)
    return explainer


def build_dashboard_config(
    model_path: Path = MODEL_PATH,
    yaml_path: Path = DASHBOARD_YAML_PATH,
    explainer_path: Path = EXPLAINER_PATH,
) -> Path:
    """Create dashboard YAML and explainer artifact from a trained pipeline."""
    explainer = _build_explainer(model_path)
    dashboard = ExplainerDashboard(
        explainer,
        title=DASHBOARD_TITLE,
        description=DASHBOARD_DESCRIPTION,
        **DASHBOARD_KWARGS,
    )

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.to_yaml(
        str(yaml_path),
        explainerfile=explainer_path.name,
        dump_explainer=True,
        explainerfile_absolute_path=False,
    )
    return yaml_path


def serve_dashboard(
    host: str = DASHBOARD_HOST,
    port: int = DASHBOARD_PORT,
) -> None:
    """Run the ExplainerDashboard web application."""
    # Rebuild from model.joblib instead of loading explainer.dill.
    # Cached dill files are tied to Python/sklearn/numba versions and break in Docker.
    explainer = _build_explainer()

    dashboard = ExplainerDashboard(
        explainer,
        title=DASHBOARD_TITLE,
        description=DASHBOARD_DESCRIPTION,
        **DASHBOARD_KWARGS,
    )
    dashboard.run(host=host, port=port, use_waitress=True)


def main() -> None:
    """Serve the dashboard (rebuilds explainer from the saved pipeline)."""
    serve_dashboard()


if __name__ == "__main__":
    main()
