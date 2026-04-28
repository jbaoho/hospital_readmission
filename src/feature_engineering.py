"""Leakage-safe split-level feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringSpec:
    missing_indicator_cols: list[str]
    log_transform_cols: list[str]
    interaction_pairs: list[tuple[str, str]]


def _numeric_columns(X: pd.DataFrame) -> list[str]:
    return [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]


def fit_feature_engineering_spec(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    max_interaction_features: int = 6,
    max_log_features: int = 12,
) -> FeatureEngineeringSpec:
    """Fit feature engineering choices on training data only."""
    missing_indicator_cols = [col for col in X_train.columns if X_train[col].isna().any()]
    numeric_cols = _numeric_columns(X_train)

    log_transform_cols: list[str] = []
    for col in numeric_cols:
        values = pd.to_numeric(X_train[col], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty or finite.min() < 0:
            continue
        if abs(float(finite.skew())) >= 1.0 and finite.nunique() > 4:
            log_transform_cols.append(col)
    log_transform_cols = log_transform_cols[:max_log_features]

    y = pd.Series(y_train).reset_index(drop=True)
    candidate_scores: list[tuple[float, str]] = []
    for col in numeric_cols:
        values = pd.to_numeric(X_train[col], errors="coerce")
        if values.nunique(dropna=True) <= 2:
            continue
        corr = values.fillna(values.median()).reset_index(drop=True).corr(y)
        if pd.notna(corr):
            candidate_scores.append((abs(float(corr)), col))

    top_cols = [col for _, col in sorted(candidate_scores, reverse=True)[:max_interaction_features]]
    interaction_pairs = [(left, right) for idx, left in enumerate(top_cols) for right in top_cols[idx + 1 :]]

    return FeatureEngineeringSpec(
        missing_indicator_cols=missing_indicator_cols,
        log_transform_cols=log_transform_cols,
        interaction_pairs=interaction_pairs,
    )


def transform_with_feature_engineering_spec(X: pd.DataFrame, spec: FeatureEngineeringSpec) -> pd.DataFrame:
    """Apply a fitted feature-engineering spec to a dataframe."""
    transformed = X.copy()

    for col in spec.missing_indicator_cols:
        if col in transformed.columns:
            transformed[f"{col}_missing_indicator"] = transformed[col].isna().astype(np.int8)

    for col in spec.log_transform_cols:
        if col in transformed.columns:
            values = pd.to_numeric(transformed[col], errors="coerce")
            clipped = values.clip(lower=0)
            transformed[f"{col}_log1p_fe"] = np.log1p(clipped)

    for left, right in spec.interaction_pairs:
        if left in transformed.columns and right in transformed.columns:
            left_values = pd.to_numeric(transformed[left], errors="coerce")
            right_values = pd.to_numeric(transformed[right], errors="coerce")
            transformed[f"{left}_x_{right}_fe"] = left_values * right_values

    return transformed
