"""Tests for dashboard explainer helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from abalone.dashboard import _normalize_explainer_targets, _prepare_explainer_inputs


class _FakeExplainer:
    def __init__(self, y, preds):
        self.y = y
        self.preds = preds


def test_prepare_explainer_inputs_returns_writable_numpy() -> None:
    X = pd.DataFrame({"Sex": ["M", "F"], "Length": [0.5, 0.6]})
    y = pd.Series([10.0, 11.0], name="Age")
    X_out, y_out = _prepare_explainer_inputs(X, y)
    assert X_out is not X
    assert y_out.flags.writeable
    assert isinstance(y_out, np.ndarray)


def test_normalize_explainer_targets_fixes_pandas_series() -> None:
    explainer = _FakeExplainer(
        pd.Series([10.0, 11.0], name="Age"),
        np.array([9.5, 11.5]),
    )
    _normalize_explainer_targets(explainer)
    assert isinstance(explainer.y, np.ndarray)
    assert explainer.y.flags.writeable
    assert mean_squared_error(explainer.y, explainer.preds) > 0
