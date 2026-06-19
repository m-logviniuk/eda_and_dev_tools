"""Load and clean abalone dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from abalone.config import (
    CATEGORICAL_FEATURES,
    DATA_URL,
    DEFAULT_LOCAL_DATA,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

COLUMN_RENAMES = {
    "Whole weight": "Whole_weight",
    "Shucked weight": "Shucked_weight",
    "Viscera weight": "Viscera_weight",
    "Shell weight": "Shell_weight",
}

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_raw_data(source: str | Path | None = None) -> pd.DataFrame:
    """Load raw abalone CSV from a local path or the default remote URL."""
    if source is not None:
        return pd.read_csv(source)

    if DEFAULT_LOCAL_DATA.exists():
        return pd.read_csv(DEFAULT_LOCAL_DATA)

    return pd.read_csv(DATA_URL)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps from the original bootcamp EDA."""
    cleaned = df.rename(columns=COLUMN_RENAMES).copy()
    cleaned["Sex"] = cleaned["Sex"].replace("f", "F")

    cleaned.fillna(
        {
            "Diameter": cleaned["Diameter"].median(),
            "Whole_weight": cleaned["Whole_weight"].median(),
            "Shell_weight": cleaned["Shell_weight"].median(),
        },
        inplace=True,
    )
    cleaned["Height"] = cleaned["Height"].replace(0.0, cleaned["Height"].median())

    cleaned[TARGET_COLUMN] = cleaned["Rings"] + 1.5
    cleaned.drop(columns=["Rings"], inplace=True)

    return cleaned


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataframe into features and target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def get_train_test_split(
    source: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and split data into train and test sets."""
    df = clean_raw_data(load_raw_data(source))
    X, y = get_X_y(df)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
