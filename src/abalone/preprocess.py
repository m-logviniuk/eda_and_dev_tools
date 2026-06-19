"""Feature preprocessing for abalone regression."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from abalone.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "scaling",
                MinMaxScaler(),
                NUMERIC_FEATURES,
            ),
        ],
    )
