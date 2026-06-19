"""Project paths and constants."""

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    """
    Resolve the project root for dev (editable install) and production (Docker).

    Lookup order:
    1. ABALONE_PROJECT_ROOT environment variable
    2. Parent of `src/` when running from the repository layout
    3. Current working directory if it contains `artifacts/`
    4. Fallback to repository-style layout relative to this file
    """
    env_root = os.environ.get("ABALONE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd = Path.cwd()
    if (cwd / "artifacts").is_dir():
        return cwd

    file_root = Path(__file__).resolve().parent.parent.parent
    if (file_root / "artifacts").is_dir() or (file_root / "src").is_dir():
        return file_root

    return file_root


PROJECT_ROOT = _resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
DASHBOARD_YAML_PATH = ARTIFACTS_DIR / "dashboard.yaml"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.dill"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"

DATA_URL = (
    "https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools"
    "/main/datasets/abalone.csv"
)
DEFAULT_LOCAL_DATA = DATA_DIR / "abalone.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
BOOTSTRAP_SAMPLES = 1000

CATEGORICAL_FEATURES = ["Sex"]
NUMERIC_FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
]
TARGET_COLUMN = "Age"

DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 9050
EXPLAINER_SAMPLE_SIZE = 200
