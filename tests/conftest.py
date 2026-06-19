"""Shared pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def raw_abalone_df() -> pd.DataFrame:
    """Minimal raw dataframe matching the UCI abalone schema."""
    return pd.DataFrame(
        {
            "Sex": ["M", "F", "I", "f", "M"],
            "Length": [0.5, 0.55, 0.43, 0.6, 0.48],
            "Diameter": [0.4, 0.42, 0.35, 0.45, None],
            "Height": [0.15, 0.0, 0.12, 0.14, 0.13],
            "Whole weight": [0.5, 0.55, 0.4, 0.7, 0.45],
            "Shucked weight": [0.22, 0.25, 0.18, 0.3, 0.2],
            "Viscera weight": [0.12, 0.14, 0.1, 0.18, 0.11],
            "Shell weight": [0.15, None, 0.12, 0.2, 0.13],
            "Rings": [8, 10, 6, 12, 7],
        }
    )


@pytest.fixture
def cleaned_abalone_df(raw_abalone_df: pd.DataFrame) -> pd.DataFrame:
    from abalone.data import clean_raw_data

    return clean_raw_data(raw_abalone_df)


@pytest.fixture
def regression_xy(cleaned_abalone_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    from abalone.data import get_X_y

    return get_X_y(cleaned_abalone_df)


@pytest.fixture
def synthetic_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Larger synthetic dataset for pipeline fitting tests."""
    rows = []
    for sex in ["M", "F", "I"]:
        for ring in range(5, 15):
            rows.append(
                {
                    "Sex": sex,
                    "Length": 0.4 + ring * 0.02,
                    "Diameter": 0.3 + ring * 0.015,
                    "Height": 0.1 + ring * 0.005,
                    "Whole_weight": 0.3 + ring * 0.05,
                    "Shucked_weight": 0.15 + ring * 0.02,
                    "Viscera_weight": 0.08 + ring * 0.01,
                    "Shell_weight": 0.1 + ring * 0.015,
                    "Rings": ring,
                }
            )
    df = pd.DataFrame(rows)
    from abalone.data import clean_raw_data, get_X_y

    cleaned = clean_raw_data(df)
    return get_X_y(cleaned)
