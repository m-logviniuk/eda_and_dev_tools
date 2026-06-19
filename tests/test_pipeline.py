"""Tests for sklearn pipeline construction."""

from sklearn.linear_model import Ridge

from abalone.config import MODEL_PATH
from abalone.pipeline import build_pipeline, load_pipeline, save_pipeline


def test_pipeline_fit_predict(synthetic_regression_data) -> None:
    X, y = synthetic_regression_data
    pipeline = build_pipeline(Ridge())
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


def test_pipeline_save_load_roundtrip(synthetic_regression_data, tmp_path) -> None:
    X, y = synthetic_regression_data
    pipeline = build_pipeline(Ridge())
    pipeline.fit(X, y)

    model_path = tmp_path / "model.joblib"
    save_pipeline(pipeline, model_path)
    loaded = load_pipeline(model_path)
    assert loaded.predict(X).shape == pipeline.predict(X).shape


def test_committed_model_loads_if_present() -> None:
    if not MODEL_PATH.exists():
        return
    pipeline = load_pipeline(MODEL_PATH)
    assert hasattr(pipeline, "predict")
