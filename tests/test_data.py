"""Tests for data loading and cleaning."""

import pandas as pd

from abalone.data import (
    clean_raw_data,
    get_X_y,
    load_raw_data,
)


def test_clean_raw_data_fixes_sex_and_target(raw_abalone_df: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_abalone_df)
    assert "f" not in cleaned["Sex"].values
    assert "Rings" not in cleaned.columns
    assert cleaned["Age"].iloc[0] == 8 + 1.5
    assert cleaned["Height"].min() > 0


def test_clean_raw_data_fills_missing(raw_abalone_df: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_abalone_df)
    assert cleaned["Diameter"].isna().sum() == 0
    assert cleaned["Whole_weight"].isna().sum() == 0
    assert cleaned["Shell_weight"].isna().sum() == 0


def test_get_X_y_columns(cleaned_abalone_df: pd.DataFrame) -> None:
    X, y = get_X_y(cleaned_abalone_df)
    assert list(X.columns) == [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    ]
    assert y.name == "Age"


def test_load_raw_data_from_local(tmp_path) -> None:
    csv_path = tmp_path / "abalone.csv"
    pd.DataFrame(
        {
            "Sex": ["M"],
            "Length": [0.5],
            "Diameter": [0.4],
            "Height": [0.15],
            "Whole weight": [0.5],
            "Shucked weight": [0.22],
            "Viscera weight": [0.12],
            "Shell weight": [0.15],
            "Rings": [8],
        }
    ).to_csv(csv_path, index=False)
    loaded = load_raw_data(csv_path)
    assert len(loaded) == 1
