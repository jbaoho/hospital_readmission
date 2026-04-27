"""Plotting utilities for model diagnostics and final analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "hospital_readmission_matplotlib"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .config import FIGURES_DIR, ensure_directories


def _save(fig: plt.Figure, path: Path) -> Path:
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(y: pd.Series | np.ndarray, path: Path = FIGURES_DIR / "class_distribution.png") -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4c78a8")
    ax.set_xlabel("Readmitted within 30 days")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, path)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"ROC Curve: {model_name}")
    return _save(fig, path)


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.set_title(f"Precision-Recall Curve: {model_name}")
    return _save(fig, path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    threshold: float,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["NO/>30", "<30"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    return _save(fig, path)


def plot_xgboost_feature_importance(
    model,
    feature_names: list[str],
    path: Path = FIGURES_DIR / "xgboost_feature_importance.png",
    top_n: int = 25,
) -> Path:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("XGBoost model does not expose feature_importances_.")

    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.22)))
    sns.barplot(data=df, x="importance", y="feature", ax=ax, color="#59a14f")
    ax.set_title(f"Top {len(df)} XGBoost Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return _save(fig, path)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "pr_auc",
    split: str = "test",
    path: Path | None = None,
) -> Path:
    if path is None:
        path = FIGURES_DIR / f"{metric}_comparison.png"
    data = metrics_df[metrics_df["split"] == split].copy()
    if data.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    data = data.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x=metric, y="model", ax=ax, color="#4c78a8")
    ax.set_title(f"{metric.upper()} Comparison ({split})")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("")
    return _save(fig, path)


def plot_recall_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    path: Path,
    thresholds: np.ndarray | None = None,
) -> Path:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    recalls = []
    precisions = []
    for threshold in thresholds:
        y_pred = np.asarray(y_proba) >= threshold
        tp = np.sum((np.asarray(y_true) == 1) & y_pred)
        fp = np.sum((np.asarray(y_true) == 0) & y_pred)
        fn = np.sum((np.asarray(y_true) == 1) & ~y_pred)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, recalls, label="Recall", color="#4c78a8")
    ax.plot(thresholds, precisions, label="Precision", color="#f58518")
    ax.set_title(f"Recall/Precision vs Threshold: {model_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    return _save(fig, path)
