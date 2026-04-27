"""Evaluation metrics and threshold selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression

from .config import METRICS_PATH, ensure_directories


def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate binary probabilities at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    labels = np.unique(y_true)
    roc_auc = roc_auc_score(y_true, y_proba) if len(labels) == 2 else np.nan
    pr_auc = average_precision_score(y_true, y_proba) if len(labels) == 2 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_grid(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate a set of thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    rows = [evaluate_binary(y_true, y_proba, float(t)) for t in thresholds]
    return pd.DataFrame(rows)


def choose_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Choose the validation threshold that maximizes F1."""
    grid = threshold_grid(y_true, y_proba)
    return float(grid.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"])


def choose_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.8,
) -> float:
    """Choose the highest-F1 threshold among thresholds meeting a recall target."""
    grid = threshold_grid(y_true, y_proba)
    eligible = grid[grid["recall"] >= min_recall]
    if eligible.empty:
        return choose_threshold_max_f1(y_true, y_proba)
    return float(eligible.sort_values(["f1", "precision"], ascending=False).iloc[0]["threshold"])


def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "max_f1",
    min_recall: float = 0.6,
) -> tuple[float, str]:
    """Choose a threshold with a named validation strategy."""
    if strategy == "max_f1":
        return choose_threshold_max_f1(y_true, y_proba), "val_max_f1"
    if strategy == "recall_target":
        return choose_threshold_for_recall(y_true, y_proba, min_recall), f"val_recall_{min_recall:g}"
    if strategy == "fixed_0.5":
        return 0.5, "fixed_0.5"
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def calibrate_probabilities_isotonic(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    p_apply: np.ndarray,
) -> tuple[IsotonicRegression, np.ndarray]:
    """Fit isotonic calibration on validation probabilities and transform new probabilities."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(p_cal, dtype=float), np.asarray(y_cal, dtype=int))
    return calibrator, calibrator.transform(np.asarray(p_apply, dtype=float))


def tune_weighted_average(
    y_true: np.ndarray,
    val_probabilities: dict[str, np.ndarray],
    metric: str = "pr_auc",
    step: float = 0.02,
) -> tuple[dict[str, float], np.ndarray, float]:
    """Tune a two-model weighted average on validation probabilities."""
    if len(val_probabilities) != 2:
        raise ValueError("Weighted average tuning currently expects exactly two models.")
    names = list(val_probabilities)
    best_score = -np.inf
    best_weight = 0.5
    best_proba = None

    for weight in np.arange(0.0, 1.0 + step / 2, step):
        combined = weight * val_probabilities[names[0]] + (1.0 - weight) * val_probabilities[names[1]]
        if metric == "pr_auc":
            score = average_precision_score(y_true, combined)
        elif metric == "roc_auc":
            score = roc_auc_score(y_true, combined)
        elif metric == "f1":
            threshold = choose_threshold_max_f1(y_true, combined)
            score = f1_score(y_true, combined >= threshold, zero_division=0)
        else:
            raise ValueError(f"Unsupported ensemble tuning metric: {metric}")
        if score > best_score:
            best_score = float(score)
            best_weight = float(weight)
            best_proba = combined

    assert best_proba is not None
    return {names[0]: best_weight, names[1]: 1.0 - best_weight}, best_proba, best_score


def evaluate_subgroups(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    threshold: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate recall and PR-AUC by sensitive subgroup columns."""
    if columns is None:
        columns = ["age", "gender"]

    rows: list[dict] = []
    for column in columns:
        if column not in X.columns:
            continue
        values = X[column].fillna("Missing").astype(str)
        for group, idx in values.groupby(values).groups.items():
            idx_array = np.asarray(list(idx))
            if len(idx_array) < 20:
                continue
            y_group = np.asarray(y_true)[idx_array]
            p_group = np.asarray(y_proba)[idx_array]
            if len(np.unique(y_group)) < 2:
                pr_auc = np.nan
            else:
                pr_auc = float(average_precision_score(y_group, p_group))
            y_pred = p_group >= threshold
            rows.append(
                {
                    "model": model_name,
                    "attribute": column,
                    "group": group,
                    "n": int(len(idx_array)),
                    "positive_rate": float(np.mean(y_group)),
                    "pr_auc": pr_auc,
                    "recall": float(recall_score(y_group, y_pred, zero_division=0)),
                    "precision": float(precision_score(y_group, y_pred, zero_division=0)),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(rows)


def metrics_row(
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    threshold_source: str = "fixed_0.5",
) -> dict[str, float | int | str]:
    """Create a metrics row with model/split metadata."""
    row = evaluate_binary(y_true, y_proba, threshold)
    return {
        "model": model_name,
        "split": split,
        "threshold_source": threshold_source,
        **row,
    }


def save_metrics(rows: list[dict], path: Path = METRICS_PATH, append: bool = False) -> pd.DataFrame:
    """Save metrics rows to CSV."""
    ensure_directories()
    df_new = pd.DataFrame(rows)
    if append and path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    return df
