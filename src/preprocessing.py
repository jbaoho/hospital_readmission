"""Leakage-safe preprocessing for sklearn, XGBoost, PyTorch, and TabNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    CATEGORICAL_ID_COLUMNS,
    DATASET_CSV,
    DROP_COLUMNS,
    POSITIVE_LABEL,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class SklearnPreparedData:
    preprocessor: ColumnTransformer
    X_train: object
    X_val: object
    X_test: object
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    categorical_cols: list[str]
    numeric_cols: list[str]
    feature_names: list[str]


@dataclass
class DeepPreparedData:
    X_cat_train: np.ndarray
    X_cat_val: np.ndarray
    X_cat_test: np.ndarray
    X_num_train: np.ndarray
    X_num_val: np.ndarray
    X_num_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cat_cols: list[str]
    num_cols: list[str]
    cat_dims: list[int]
    cat_idxs: list[int]
    cat_encoder: OrdinalEncoder
    cat_imputer: SimpleImputer | None
    num_pipeline: Pipeline | None


def load_raw_csv(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw UCI CSV."""
    path = Path(csv_path) if csv_path is not None else RAW_DATA_DIR / DATASET_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python -m src.data_download` "
            "or place diabetic_data.csv in data/raw/."
        )
    return pd.read_csv(path)


def _icd9_group(value: object) -> str:
    """Map raw ICD-9 diagnosis codes to broad clinical categories."""
    if pd.isna(value):
        return "Missing"

    text = str(value).strip()
    if not text:
        return "Missing"
    if text.startswith("V"):
        return "Supplementary"
    if text.startswith("E"):
        return "External Injury"

    try:
        code = float(text)
    except ValueError:
        return "Other"

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"
    if 460 <= code <= 519 or code == 786:
        return "Respiratory"
    if 520 <= code <= 579 or code == 787:
        return "Digestive"
    if int(code) == 250:
        return "Diabetes"
    if 800 <= code <= 999:
        return "Injury"
    if 710 <= code <= 739:
        return "Musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    if 140 <= code <= 239:
        return "Neoplasms"
    if 240 <= code <= 279 and int(code) != 250:
        return "Endocrine"
    if 680 <= code <= 709:
        return "Skin"
    if 290 <= code <= 319:
        return "Mental"
    if 280 <= code <= 289:
        return "Blood"
    return "Other"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level clinical/utilization features before train/test splitting."""
    engineered = df.copy()

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in engineered.columns:
            engineered[f"{col}_group"] = engineered[col].map(_icd9_group)

    utilization_cols = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"] if c in engineered]
    if utilization_cols:
        engineered["prior_visit_count"] = engineered[utilization_cols].fillna(0).sum(axis=1)
        engineered["prior_visit_count_log1p"] = np.log1p(engineered["prior_visit_count"])
        engineered["has_prior_inpatient"] = (
            engineered.get("number_inpatient", pd.Series(0, index=engineered.index)).fillna(0) > 0
        ).astype(int)
        engineered["has_emergency_history"] = (
            engineered.get("number_emergency", pd.Series(0, index=engineered.index)).fillna(0) > 0
        ).astype(int)

    med_cols = [c for c in MEDICATION_COLUMNS if c in engineered.columns]
    if med_cols:
        med_frame = engineered[med_cols].fillna("No")
        engineered["medication_active_count"] = med_frame.ne("No").sum(axis=1)
        engineered["medication_change_count"] = med_frame.isin(["Up", "Down"]).sum(axis=1)
        engineered["any_medication_change_engineered"] = (engineered["medication_change_count"] > 0).astype(int)

    if "insulin" in engineered.columns:
        engineered["insulin_changed_engineered"] = engineered["insulin"].isin(["Up", "Down"]).astype(int)
        engineered["insulin_used_engineered"] = engineered["insulin"].ne("No").astype(int)

    if {"num_medications", "time_in_hospital"}.issubset(engineered.columns):
        denominator = engineered["time_in_hospital"].replace(0, np.nan)
        engineered["medications_per_hospital_day"] = (
            engineered["num_medications"] / denominator
        ).replace([np.inf, -np.inf], np.nan)

    if {"num_lab_procedures", "time_in_hospital"}.issubset(engineered.columns):
        denominator = engineered["time_in_hospital"].replace(0, np.nan)
        engineered["labs_per_hospital_day"] = (
            engineered["num_lab_procedures"] / denominator
        ).replace([np.inf, -np.inf], np.nan)

    return engineered


def clean_dataframe(
    df: pd.DataFrame,
    missing_threshold: float = 0.4,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Replace UCI missing markers, create binary target, and drop sparse/id columns."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column `{TARGET_COLUMN}` in raw dataframe.")

    cleaned = df.replace("?", np.nan).copy()
    cleaned = add_engineered_features(cleaned)
    cleaned["readmitted_binary"] = (cleaned[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    cleaned = cleaned.drop(columns=[TARGET_COLUMN])

    high_missing = cleaned.columns[cleaned.isna().mean() > missing_threshold].tolist()
    requested_drop = drop_columns if drop_columns is not None else DROP_COLUMNS
    to_drop = sorted(set(high_missing + [c for c in requested_drop if c in cleaned.columns]))
    if to_drop:
        cleaned = cleaned.drop(columns=to_drop)

    return cleaned


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "readmitted_binary" not in df.columns:
        raise ValueError("Expected `readmitted_binary`; call clean_dataframe first.")
    X = df.drop(columns=["readmitted_binary"])
    y = df["readmitted_binary"].astype(int)
    return X, y


def create_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Create stratified train/validation/test splits."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=y_temp,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def create_patient_level_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Create splits where each `patient_nbr` appears in only one split."""
    if "patient_nbr" not in df.columns:
        raise ValueError("Patient-level split requires `patient_nbr` before id columns are dropped.")
    if "readmitted_binary" not in df.columns:
        raise ValueError("Patient-level split requires `readmitted_binary`.")
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    patient_labels = df.groupby("patient_nbr")["readmitted_binary"].max().astype(int)
    patients = patient_labels.index.to_series()

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=patient_labels,
    )
    temp_labels = patient_labels.loc[temp_patients]
    relative_test = test_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test,
        random_state=random_state,
        stratify=temp_labels,
    )

    train_df = df[df["patient_nbr"].isin(train_patients)].copy()
    val_df = df[df["patient_nbr"].isin(val_patients)].copy()
    test_df = df[df["patient_nbr"].isin(test_patients)].copy()

    drop_ids = [c for c in DROP_COLUMNS if c in df.columns]

    def split_xy(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y_part = part["readmitted_binary"].astype(int).reset_index(drop=True)
        X_part = part.drop(columns=["readmitted_binary"] + drop_ids).reset_index(drop=True)
        return X_part, y_part

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify categorical and numeric columns with known coded IDs treated as categorical."""
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in X.columns:
        if col in CATEGORICAL_ID_COLUMNS:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return categorical_cols, numeric_cols


def prepare_splits(
    csv_path: str | Path | None = None,
    sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
    missing_threshold: float = 0.4,
    split_strategy: str = "encounter",
) -> SplitData:
    """Load raw data, clean it, optionally sample rows, and split."""
    raw = load_raw_csv(csv_path)
    drop_columns = [] if split_strategy == "patient" else None
    cleaned = clean_dataframe(raw, missing_threshold=missing_threshold, drop_columns=drop_columns)

    if sample_size is not None and sample_size < len(cleaned):
        cleaned, _ = train_test_split(
            cleaned,
            train_size=sample_size,
            random_state=random_state,
            stratify=cleaned["readmitted_binary"],
        )
        cleaned = cleaned.reset_index(drop=True)

    if split_strategy == "patient":
        return create_patient_level_splits(cleaned, random_state=random_state)
    if split_strategy != "encounter":
        raise ValueError("split_strategy must be either 'encounter' or 'patient'.")

    X, y = separate_features_target(cleaned)
    return create_splits(X, y, random_state=random_state)


def build_sklearn_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(num_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def fit_transform_sklearn(split: SplitData, scale_numeric: bool = True) -> SklearnPreparedData:
    """Fit sklearn preprocessing on train only and transform all splits."""
    categorical_cols, numeric_cols = identify_feature_types(split.X_train)
    preprocessor = build_sklearn_preprocessor(categorical_cols, numeric_cols, scale_numeric)
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = numeric_cols + categorical_cols

    return SklearnPreparedData(
        preprocessor=preprocessor,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=split.y_train.to_numpy(dtype=np.int64),
        y_val=split.y_val.to_numpy(dtype=np.int64),
        y_test=split.y_test.to_numpy(dtype=np.int64),
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        feature_names=feature_names,
    )


def fit_transform_deep(split: SplitData) -> DeepPreparedData:
    """Prepare categorical-index and normalized numeric arrays for MLP/TabNet."""
    cat_cols, num_cols = identify_feature_types(split.X_train)

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        train_cat_raw = cat_imputer.fit_transform(split.X_train[cat_cols].astype("object"))
        val_cat_raw = cat_imputer.transform(split.X_val[cat_cols].astype("object"))
        test_cat_raw = cat_imputer.transform(split.X_test[cat_cols].astype("object"))

        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_train = cat_encoder.fit_transform(train_cat_raw).astype(np.int64) + 1
        X_cat_val = cat_encoder.transform(val_cat_raw).astype(np.int64) + 1
        X_cat_test = cat_encoder.transform(test_cat_raw).astype(np.int64) + 1
        cat_dims = [len(categories) + 1 for categories in cat_encoder.categories_]
    else:
        cat_imputer = None
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_train = np.empty((len(split.X_train), 0), dtype=np.int64)
        X_cat_val = np.empty((len(split.X_val), 0), dtype=np.int64)
        X_cat_test = np.empty((len(split.X_test), 0), dtype=np.int64)
        cat_dims = []

    if num_cols:
        num_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        X_num_train = num_pipeline.fit_transform(split.X_train[num_cols]).astype(np.float32)
        X_num_val = num_pipeline.transform(split.X_val[num_cols]).astype(np.float32)
        X_num_test = num_pipeline.transform(split.X_test[num_cols]).astype(np.float32)
    else:
        num_pipeline = None
        X_num_train = np.empty((len(split.X_train), 0), dtype=np.float32)
        X_num_val = np.empty((len(split.X_val), 0), dtype=np.float32)
        X_num_test = np.empty((len(split.X_test), 0), dtype=np.float32)

    return DeepPreparedData(
        X_cat_train=X_cat_train,
        X_cat_val=X_cat_val,
        X_cat_test=X_cat_test,
        X_num_train=X_num_train,
        X_num_val=X_num_val,
        X_num_test=X_num_test,
        y_train=split.y_train.to_numpy(dtype=np.float32),
        y_val=split.y_val.to_numpy(dtype=np.float32),
        y_test=split.y_test.to_numpy(dtype=np.float32),
        cat_cols=cat_cols,
        num_cols=num_cols,
        cat_dims=cat_dims,
        cat_idxs=list(range(len(cat_cols))),
        cat_encoder=cat_encoder,
        cat_imputer=cat_imputer,
        num_pipeline=num_pipeline,
    )


def combined_tabnet_matrix(X_cat: np.ndarray, X_num: np.ndarray) -> np.ndarray:
    """Return one float32 matrix with categorical columns first for TabNet."""
    return np.concatenate([X_cat.astype(np.float32), X_num.astype(np.float32)], axis=1)
