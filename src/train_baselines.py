"""Training utilities for logistic regression and XGBoost baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.linear_model import LogisticRegression

from .config import MODELS_DIR, ensure_directories
from .evaluate import choose_threshold_max_f1, metrics_row, save_metrics
from .preprocessing import fit_transform_sklearn, prepare_splits


def positive_class_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0:
        return 1.0
    return neg / pos


def train_logistic_regression(X_train, y_train: np.ndarray, model_dir: Path = MODELS_DIR) -> LogisticRegression:
    """Train class-balanced logistic regression."""
    ensure_directories()
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_dir / "logistic_regression.joblib")
    return model


def train_xgboost(
    X_train,
    y_train: np.ndarray,
    X_val=None,
    y_val: np.ndarray | None = None,
    model_dir: Path = MODELS_DIR,
    quick: bool = False,
    n_jobs: int = -1,
    params: dict | None = None,
):
    """Train XGBoost with scale_pos_weight for class imbalance."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run `pip install xgboost` or use the Colab setup cells."
        ) from exc

    ensure_directories()
    base_params = dict(
        n_estimators=50 if quick else 400,
        max_depth=3 if quick else 5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=positive_class_weight(y_train),
        random_state=42,
        n_jobs=n_jobs,
    )
    if params:
        base_params.update(params)

    model = XGBClassifier(**base_params)
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    joblib.dump(model, model_dir / "xgboost.joblib")
    return model


def tune_xgboost(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    model_dir: Path = MODELS_DIR,
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Manual validation-set random search optimized for PR-AUC."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run `pip install xgboost` or use the Colab setup cells."
        ) from exc

    ensure_directories()
    search_space = {
        "n_estimators": [250, 400, 650, 900],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.04, 0.06, 0.08, 0.12],
        "min_child_weight": [1, 3, 5, 8],
        "subsample": [0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.65, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.25, 0.5, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": [
            0.75 * positive_class_weight(y_train),
            positive_class_weight(y_train),
            1.25 * positive_class_weight(y_train),
            1.5 * positive_class_weight(y_train),
        ],
    }
    sampler = ParameterSampler(search_space, n_iter=n_iter, random_state=random_state)

    best_model = None
    best_score = -np.inf
    best_params = None
    for idx, params in enumerate(sampler, start=1):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        score = float(average_precision_score(y_val, val_proba))
        print(f"xgb_tune {idx}/{n_iter} val_pr_auc={score:.5f} params={params}")
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("XGBoost tuning failed to train any model.")

    joblib.dump(best_model, model_dir / "xgboost_tuned.joblib")
    joblib.dump({"val_pr_auc": best_score, "params": best_params}, model_dir / "xgboost_tuning_summary.joblib")
    print(f"best_xgb_val_pr_auc={best_score:.5f} params={best_params}")
    return best_model, {"val_pr_auc": best_score, "params": best_params}


def predict_proba(model, X) -> np.ndarray:
    """Return positive-class probabilities for sklearn-compatible classifiers."""
    return model.predict_proba(X)[:, 1]


def run_baselines(sample_size: int | None = None, quick: bool = False) -> list[dict]:
    """Train/evaluate logistic regression and XGBoost, returning test metrics rows."""
    split = prepare_splits(sample_size=sample_size)
    data = fit_transform_sklearn(split)
    joblib.dump(data.preprocessor, MODELS_DIR / "sklearn_preprocessor.joblib")

    rows: list[dict] = []

    logreg = train_logistic_regression(data.X_train, data.y_train)
    val_proba = predict_proba(logreg, data.X_val)
    threshold = choose_threshold_max_f1(data.y_val, val_proba)
    test_proba = predict_proba(logreg, data.X_test)
    rows.append(metrics_row("Logistic Regression", "test", data.y_test, test_proba, threshold, "val_max_f1"))

    xgb = train_xgboost(data.X_train, data.y_train, data.X_val, data.y_val, quick=quick)
    val_proba = predict_proba(xgb, data.X_val)
    threshold = choose_threshold_max_f1(data.y_val, val_proba)
    test_proba = predict_proba(xgb, data.X_test)
    rows.append(metrics_row("XGBoost", "test", data.y_test, test_proba, threshold, "val_max_f1"))

    save_metrics(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use faster XGBoost settings.")
    args = parser.parse_args()
    rows = run_baselines(sample_size=args.sample_size, quick=args.quick)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
