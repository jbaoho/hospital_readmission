"""Project configuration and shared constants."""

from __future__ import annotations

from pathlib import Path


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_PATH = RESULTS_DIR / "metrics.csv"

DATASET_CSV = "diabetic_data.csv"
IDS_MAPPING_CSV = "IDS_mapping.csv"

TARGET_COLUMN = "readmitted"
POSITIVE_LABEL = "<30"

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/296"
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
]

ID_COLUMNS = ["encounter_id", "patient_nbr"]

CATEGORICAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

DROP_COLUMNS = ID_COLUMNS


def ensure_directories() -> None:
    """Create project output directories if they do not exist."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
