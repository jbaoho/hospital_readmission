"""End-to-end experiment runner for baselines, deep models, plots, and ensemble."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch

from .config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_RANKINGS_PATH,
    MODEL_ABLATION_PATH,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    THRESHOLDS_PATH,
    ensure_directories,
)
from .data_download import download_dataset
from .evaluate import (
    calibrate_probabilities_isotonic,
    choose_threshold,
    evaluate_binary,
    evaluate_subgroups,
    metrics_row,
    save_metrics,
    tune_weighted_average_any,
)
from .plots import (
    plot_class_distribution,
    plot_combined_precision_recall_curves,
    plot_combined_roc_curves,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curve,
    plot_recall_threshold_curve,
    plot_roc_curve,
    plot_xgboost_feature_importance,
)
from .preprocessing import fit_transform_deep, fit_transform_sklearn, prepare_splits
from .train_baselines import (
    predict_proba,
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
    tune_catboost,
    tune_lightgbm,
    tune_xgboost,
)
from .train_deep import predict_mlp, predict_tabnet, train_mlp, train_tabnet


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace("+", "_plus_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "_")
    )


def _add_threshold_rows(
    rows: list[dict],
    model_name: str,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
    threshold_strategy: str,
    min_recall: float,
    include_recall_target: bool,
    runtime_seconds: float | None = None,
) -> float:
    threshold, source = choose_threshold(y_val, val_proba, threshold_strategy, min_recall)
    rows.append(
        metrics_row(
            model_name,
            "test",
            y_test,
            test_proba,
            threshold,
            source,
            min_recall,
            is_recall_target=source.startswith("val_recall"),
            runtime_seconds=runtime_seconds,
        )
    )

    if include_recall_target and threshold_strategy != "recall_target":
        recall_threshold, recall_source = choose_threshold(y_val, val_proba, "recall_target", min_recall)
        rows.append(
            metrics_row(
                f"{model_name} (recall target)",
                "test",
                y_test,
                test_proba,
                recall_threshold,
                recall_source,
                min_recall,
                is_recall_target=True,
                runtime_seconds=runtime_seconds,
            )
        )
    return threshold


def _save_prediction_files(
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
) -> None:
    """Persist model probabilities for reproducible reporting and ensembling checks."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in PREDICTIONS_DIR.glob("*_predictions.csv"):
        old_file.unlink()
    for model_name, y_proba in val_probabilities.items():
        pd.DataFrame({"y_true": y_val, "y_proba": y_proba}).to_csv(
            PREDICTIONS_DIR / f"{_slug(model_name)}_val_predictions.csv",
            index=False,
        )
    for model_name, y_proba in test_probabilities.items():
        pd.DataFrame({"y_true": y_test, "y_proba": y_proba}).to_csv(
            PREDICTIONS_DIR / f"{_slug(model_name)}_test_predictions.csv",
            index=False,
        )


def _save_thresholds(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "split",
        "threshold_source",
        "recall_target",
        "min_recall",
        "runtime_seconds",
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    thresholds_df = metrics_df[[col for col in columns if col in metrics_df.columns]].copy()
    thresholds_df.to_csv(THRESHOLDS_PATH, index=False)
    return thresholds_df


def _save_model_rankings(metrics_df: pd.DataFrame, min_recall: float) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    if test_df.empty:
        rankings = pd.DataFrame()
        rankings.to_csv(MODEL_RANKINGS_PATH, index=False)
        return rankings

    by_pr_auc = test_df.sort_values(["test_pr_auc", "recall"], ascending=False).copy()
    by_pr_auc["ranking"] = "test_pr_auc"
    by_pr_auc["rank"] = np.arange(1, len(by_pr_auc) + 1)

    if "recall_target" in test_df.columns:
        test_df["recall_target"] = test_df["recall_target"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    recall_target_df = test_df[(test_df["recall_target"]) & (test_df["recall"] >= min_recall)].copy()
    recall_target_df = recall_target_df.sort_values(["precision", "test_pr_auc"], ascending=False)
    recall_target_df["ranking"] = f"precision_at_recall_{min_recall:g}"
    recall_target_df["rank"] = np.arange(1, len(recall_target_df) + 1)

    rankings = pd.concat([by_pr_auc, recall_target_df], ignore_index=True)
    rankings.to_csv(MODEL_RANKINGS_PATH, index=False)
    return rankings


def _probability_key_for_metrics_model(model_name: str, available: dict[str, np.ndarray]) -> str | None:
    cleaned = model_name.replace(" (recall target)", "")
    if cleaned in available:
        return cleaned
    prefixes = {
        "XGBoost Tuned Calibrated": "XGBoost Calibrated",
        "XGBoost Calibrated": "XGBoost Calibrated",
        "XGBoost Tuned": "XGBoost",
        "XGBoost": "XGBoost",
        "LightGBM Tuned": "LightGBM",
        "LightGBM": "LightGBM",
        "CatBoost Tuned": "CatBoost",
        "CatBoost": "CatBoost",
        "Residual MLP Focal": "MLP",
        "TabM-Style Ensemble MLP": "TabM",
        "TabTransformer": "TabTransformer",
        "TabNet Pretrained": "TabNet",
        "TabNet": "TabNet",
    }
    for prefix, key in prefixes.items():
        if cleaned.startswith(prefix) and key in available:
            return key
    if cleaned.startswith("Weighted "):
        key = cleaned.removeprefix("Weighted ").removesuffix(" Ensemble")
        if key in available:
            return key
    if cleaned.startswith("Stacking Logistic Regression C="):
        c_text = cleaned.split("C=", 1)[1].split(" ", 1)[0]
        key = f"Stacking LogReg C={c_text}"
        if key in available:
            return key
    if cleaned.startswith("Stacking LightGBM Meta") and "Stacking LightGBM Meta" in available:
        return "Stacking LightGBM Meta"
    return None


def _add_weighted_ensemble(
    rows: list[dict],
    val_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    key: str,
    model_names: list[str],
    y_val: np.ndarray,
    y_test: np.ndarray,
    threshold_strategy: str,
    min_recall: float,
    include_recall_target: bool,
    quick: bool = False,
) -> None:
    if key in val_probabilities or any(name not in val_probabilities or name not in test_probabilities for name in model_names):
        return
    started_at = time.perf_counter()
    weights, val_ensemble, ensemble_score = tune_weighted_average_any(
        y_val,
        {name: val_probabilities[name] for name in model_names},
        metric="pr_auc",
        step=0.02,
        n_random=100 if quick else 500,
    )
    test_ensemble = np.zeros_like(next(iter(test_probabilities.values())), dtype=float)
    for name, weight in weights.items():
        test_ensemble += weight * test_probabilities[name]
    val_probabilities[key] = val_ensemble
    test_probabilities[key] = test_ensemble
    print(f"{key}_val_pr_auc={ensemble_score:.5f} weights={weights}")
    thresholds[key] = _add_threshold_rows(
        rows,
        f"Weighted {key} Ensemble",
        y_val,
        val_ensemble,
        y_test,
        test_ensemble,
        threshold_strategy,
        min_recall,
        include_recall_target,
        time.perf_counter() - started_at,
    )


def _save_ablation_study(
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
    min_recall: float,
    quick: bool,
) -> pd.DataFrame:
    """Save validation-tuned ablations without fitting on test labels."""
    ablations = {
        "XGBoost only": ["XGBoost"],
        "XGBoost + TabTransformer": ["XGBoost", "TabTransformer"],
        "XGBoost + LightGBM + CatBoost": ["XGBoost", "LightGBM", "CatBoost"],
        "All models": [
            name
            for name in [
                "Logistic Regression",
                "XGBoost",
                "XGBoost Calibrated",
                "LightGBM",
                "CatBoost",
                "MLP",
                "TabNet",
                "TabTransformer",
                "TabM",
            ]
            if name in val_probabilities and name in test_probabilities
        ],
    }
    rows: list[dict] = []
    for ablation_name, model_names in ablations.items():
        available = [name for name in model_names if name in val_probabilities and name in test_probabilities]
        if not available:
            rows.append({"ablation": ablation_name, "models": "", "status": "skipped_missing_models"})
            continue
        if len(available) == 1:
            weights = {available[0]: 1.0}
            val_proba = val_probabilities[available[0]]
            test_proba = test_probabilities[available[0]]
            val_pr_auc = float(average_precision_score(y_val, val_proba))
        else:
            weights, val_proba, val_pr_auc = tune_weighted_average_any(
                y_val,
                {name: val_probabilities[name] for name in available},
                metric="pr_auc",
                n_random=100 if quick else 500,
            )
            test_proba = np.zeros_like(test_probabilities[available[0]], dtype=float)
            for name, weight in weights.items():
                test_proba += weight * test_probabilities[name]

        threshold, source = choose_threshold(y_val, val_proba, "recall_target", min_recall)
        test_metrics = evaluate_binary(y_test, test_proba, threshold)
        rows.append(
            {
                "ablation": ablation_name,
                "models": "+".join(available),
                "status": "ok",
                "weights": weights,
                "val_pr_auc": val_pr_auc,
                "threshold_source": source,
                "recall_target": True,
                "min_recall": float(min_recall),
                **test_metrics,
            }
        )

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(MODEL_ABLATION_PATH, index=False)
    return ablation_df


def run_device_benchmark(
    sample_size: int | None = None,
    quick: bool = True,
    skip_download: bool = False,
    skip_tabnet: bool = False,
    skip_modern_dl: bool = False,
    split_strategy: str = "encounter",
    advanced_feature_engineering: bool = True,
    deep_batch_size: int = 64,
    modern_dl_epochs: int | None = None,
    min_recall: float = 0.6,
) -> pd.DataFrame:
    """Benchmark deep models on CPU vs MPS/CUDA without running tree models."""
    ensure_directories()
    if not skip_download:
        download_dataset()

    split = prepare_splits(
        sample_size=sample_size,
        split_strategy=split_strategy,
        advanced_feature_engineering=advanced_feature_engineering,
    )
    deep_data = fit_transform_deep(split)
    joblib.dump(deep_data, MODELS_DIR / "deep_preprocessing.joblib")

    devices = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    elif torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("No GPU backend available for benchmarking; running CPU only.")

    rows: list[dict] = []
    model_specs = [("Residual MLP", "residual", "focal", "benchmark_residual")]
    if not skip_modern_dl:
        model_specs.extend(
            [
                ("TabTransformer", "tabtransformer", "focal", "benchmark_tabtransformer"),
                ("TabM", "tabm", "focal", "benchmark_tabm"),
            ]
        )

    epochs = modern_dl_epochs if modern_dl_epochs is not None else (1 if quick else 5)
    for device_name in devices:
        for display_name, architecture, loss_name, artifact_name in model_specs:
            try:
                started_at = time.perf_counter()
                model, _ = train_mlp(
                    deep_data,
                    epochs=epochs,
                    patience=1 if quick else 3,
                    batch_size=deep_batch_size,
                    device=device_name,
                    architecture=architecture,
                    loss_name=loss_name,
                    model_name=f"{artifact_name}_{device_name}",
                )
                val_proba = predict_mlp(
                    model,
                    deep_data.X_cat_val,
                    deep_data.X_num_val,
                    batch_size=deep_batch_size,
                    device=device_name,
                )
                test_proba = predict_mlp(
                    model,
                    deep_data.X_cat_test,
                    deep_data.X_num_test,
                    batch_size=deep_batch_size,
                    device=device_name,
                )
                runtime_seconds = time.perf_counter() - started_at
                actual_device = getattr(model, "hospital_device", device_name)
                threshold, source = choose_threshold(deep_data.y_val, val_proba, "recall_target", min_recall)
                row = metrics_row(
                    f"Benchmark {display_name} ({actual_device})",
                    "test",
                    deep_data.y_test,
                    test_proba,
                    threshold,
                    source,
                    min_recall,
                    is_recall_target=True,
                    runtime_seconds=runtime_seconds,
                )
                row["device"] = actual_device
                rows.append(row)
            except Exception as exc:
                print(f"Benchmark {display_name} on {device_name} skipped/failed: {exc}")

        if not skip_tabnet:
            try:
                started_at = time.perf_counter()
                tabnet = train_tabnet(
                    deep_data,
                    quick=quick,
                    batch_size=deep_batch_size,
                    device=device_name,
                )
                val_proba = predict_tabnet(tabnet, deep_data.X_cat_val, deep_data.X_num_val)
                test_proba = predict_tabnet(tabnet, deep_data.X_cat_test, deep_data.X_num_test)
                runtime_seconds = time.perf_counter() - started_at
                actual_device = getattr(tabnet, "hospital_device", device_name)
                threshold, source = choose_threshold(deep_data.y_val, val_proba, "recall_target", min_recall)
                row = metrics_row(
                    f"Benchmark TabNet ({actual_device})",
                    "test",
                    deep_data.y_test,
                    test_proba,
                    threshold,
                    source,
                    min_recall,
                    is_recall_target=True,
                    runtime_seconds=runtime_seconds,
                )
                row["device"] = actual_device
                rows.append(row)
            except Exception as exc:
                print(f"Benchmark TabNet on {device_name} skipped/failed: {exc}")

    metrics_df = save_metrics(rows, METRICS_PATH, append=False)
    print("Device benchmark results:")
    if not metrics_df.empty:
        print(metrics_df[["model", "device", "runtime_seconds", "test_pr_auc", "recall", "precision"]].to_string(index=False))
    return metrics_df


def run_experiment(
    sample_size: int | None = None,
    quick: bool = False,
    skip_download: bool = False,
    skip_tabnet: bool = False,
    skip_modern_dl: bool = False,
    tune_xgb: bool = False,
    xgb_tune_iter: int = 20,
    xgb_n_jobs: int = -1,
    include_lightgbm: bool = False,
    lgbm_tune_iter: int = 0,
    include_catboost: bool = False,
    catboost_tune_iter: int = 0,
    calibrate_xgb: bool = True,
    threshold_strategy: str = "max_f1",
    min_recall: float = 0.6,
    include_recall_target: bool = True,
    use_tabnet_pretraining: bool = False,
    split_strategy: str = "encounter",
    advanced_feature_engineering: bool = True,
    skip_stacking: bool = False,
    torch_device: str | None = None,
    modern_dl_epochs: int | None = None,
    deep_batch_size: int = 64,
) -> pd.DataFrame:
    """Run all available models and save metrics/figures."""
    ensure_directories()
    if not skip_download:
        download_dataset()

    split = prepare_splits(
        sample_size=sample_size,
        split_strategy=split_strategy,
        advanced_feature_engineering=advanced_feature_engineering,
    )
    plot_class_distribution(split.y_train, FIGURES_DIR / "class_distribution_train.png")

    rows: list[dict] = []
    test_probabilities: dict[str, np.ndarray] = {}
    val_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    sklearn_data = fit_transform_sklearn(split)
    joblib.dump(sklearn_data.preprocessor, MODELS_DIR / "sklearn_preprocessor.joblib")

    started_at = time.perf_counter()
    logreg = train_logistic_regression(sklearn_data.X_train, sklearn_data.y_train)
    val_probabilities["Logistic Regression"] = predict_proba(logreg, sklearn_data.X_val)
    test_probabilities["Logistic Regression"] = predict_proba(logreg, sklearn_data.X_test)
    runtime_seconds = time.perf_counter() - started_at
    thresholds["Logistic Regression"] = _add_threshold_rows(
        rows,
        "Logistic Regression",
        sklearn_data.y_val,
        val_probabilities["Logistic Regression"],
        sklearn_data.y_test,
        test_probabilities["Logistic Regression"],
        threshold_strategy,
        min_recall,
        include_recall_target,
        runtime_seconds,
    )

    try:
        started_at = time.perf_counter()
        if tune_xgb:
            xgb, _ = tune_xgboost(
                sklearn_data.X_train,
                sklearn_data.y_train,
                sklearn_data.X_val,
                sklearn_data.y_val,
                n_iter=4 if quick else xgb_tune_iter,
                n_jobs=xgb_n_jobs,
            )
            xgb_name = "XGBoost Tuned"
        else:
            xgb = train_xgboost(
                sklearn_data.X_train,
                sklearn_data.y_train,
                sklearn_data.X_val,
                sklearn_data.y_val,
                quick=quick,
                n_jobs=xgb_n_jobs,
            )
            xgb_name = "XGBoost"
        val_probabilities["XGBoost"] = predict_proba(xgb, sklearn_data.X_val)
        test_probabilities["XGBoost"] = predict_proba(xgb, sklearn_data.X_test)
        runtime_seconds = time.perf_counter() - started_at
        thresholds["XGBoost"] = _add_threshold_rows(
            rows,
            xgb_name,
            sklearn_data.y_val,
            val_probabilities["XGBoost"],
            sklearn_data.y_test,
            test_probabilities["XGBoost"],
            threshold_strategy,
            min_recall,
            include_recall_target,
            runtime_seconds,
        )

        if calibrate_xgb:
            started_at = time.perf_counter()
            _, xgb_test_calibrated = calibrate_probabilities_isotonic(
                sklearn_data.y_val,
                val_probabilities["XGBoost"],
                test_probabilities["XGBoost"],
            )
            _, xgb_val_calibrated = calibrate_probabilities_isotonic(
                sklearn_data.y_val,
                val_probabilities["XGBoost"],
                val_probabilities["XGBoost"],
            )
            val_probabilities["XGBoost Calibrated"] = xgb_val_calibrated
            test_probabilities["XGBoost Calibrated"] = xgb_test_calibrated
            runtime_seconds = time.perf_counter() - started_at
            thresholds["XGBoost Calibrated"] = _add_threshold_rows(
                rows,
                f"{xgb_name} Calibrated",
                sklearn_data.y_val,
                xgb_val_calibrated,
                sklearn_data.y_test,
                xgb_test_calibrated,
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        plot_xgboost_feature_importance(
            xgb,
            sklearn_data.feature_names,
            FIGURES_DIR / "xgboost_feature_importance.png",
            top_n=25,
        )
    except Exception as exc:
        print(f"XGBoost skipped/failed: {exc}")

    if include_lightgbm:
        try:
            started_at = time.perf_counter()
            if lgbm_tune_iter > 0:
                lgbm, _ = tune_lightgbm(
                    sklearn_data.X_train,
                    sklearn_data.y_train,
                    sklearn_data.X_val,
                    sklearn_data.y_val,
                    n_iter=2 if quick else lgbm_tune_iter,
                )
                lgbm_name = "LightGBM Tuned"
            else:
                lgbm = train_lightgbm(
                    sklearn_data.X_train,
                    sklearn_data.y_train,
                    sklearn_data.X_val,
                    sklearn_data.y_val,
                    quick=quick,
                )
                lgbm_name = "LightGBM"
            val_probabilities["LightGBM"] = predict_proba(lgbm, sklearn_data.X_val)
            test_probabilities["LightGBM"] = predict_proba(lgbm, sklearn_data.X_test)
            runtime_seconds = time.perf_counter() - started_at
            thresholds["LightGBM"] = _add_threshold_rows(
                rows,
                lgbm_name,
                sklearn_data.y_val,
                val_probabilities["LightGBM"],
                sklearn_data.y_test,
                test_probabilities["LightGBM"],
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        except Exception as exc:
            print(f"LightGBM skipped/failed: {exc}")

    if include_catboost:
        try:
            started_at = time.perf_counter()
            if catboost_tune_iter > 0:
                catboost, _ = tune_catboost(
                    sklearn_data.X_train,
                    sklearn_data.y_train,
                    sklearn_data.X_val,
                    sklearn_data.y_val,
                    n_iter=2 if quick else catboost_tune_iter,
                )
                catboost_name = "CatBoost Tuned"
            else:
                catboost = train_catboost(
                    sklearn_data.X_train,
                    sklearn_data.y_train,
                    sklearn_data.X_val,
                    sklearn_data.y_val,
                    quick=quick,
                )
                catboost_name = "CatBoost"
            val_probabilities["CatBoost"] = predict_proba(catboost, sklearn_data.X_val)
            test_probabilities["CatBoost"] = predict_proba(catboost, sklearn_data.X_test)
            runtime_seconds = time.perf_counter() - started_at
            thresholds["CatBoost"] = _add_threshold_rows(
                rows,
                catboost_name,
                sklearn_data.y_val,
                val_probabilities["CatBoost"],
                sklearn_data.y_test,
                test_probabilities["CatBoost"],
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        except Exception as exc:
            print(f"CatBoost skipped/failed: {exc}")

    deep_data = fit_transform_deep(split)
    joblib.dump(deep_data, MODELS_DIR / "deep_preprocessing.joblib")

    try:
        started_at = time.perf_counter()
        mlp, _ = train_mlp(
            deep_data,
            epochs=3 if quick else 30,
            patience=2 if quick else 5,
            batch_size=deep_batch_size,
            device=torch_device,
        )
        val_probabilities["MLP"] = predict_mlp(
            mlp, deep_data.X_cat_val, deep_data.X_num_val, batch_size=deep_batch_size, device=torch_device
        )
        test_probabilities["MLP"] = predict_mlp(
            mlp, deep_data.X_cat_test, deep_data.X_num_test, batch_size=deep_batch_size, device=torch_device
        )
        runtime_seconds = time.perf_counter() - started_at
        thresholds["MLP"] = _add_threshold_rows(
            rows,
            "Residual MLP Focal",
            deep_data.y_val,
            val_probabilities["MLP"],
            deep_data.y_test,
            test_probabilities["MLP"],
            threshold_strategy,
            min_recall,
            include_recall_target,
            runtime_seconds,
        )
    except Exception as exc:
        print(f"MLP skipped/failed: {exc}")

    if not skip_modern_dl:
        model_epochs = modern_dl_epochs if modern_dl_epochs is not None else (3 if quick else 8)
        try:
            started_at = time.perf_counter()
            tabtransformer, _ = train_mlp(
                deep_data,
                epochs=model_epochs,
                patience=2 if quick else 5,
                batch_size=deep_batch_size,
                device=torch_device,
                architecture="tabtransformer",
                loss_name="focal",
                model_name="tabtransformer",
            )
            val_probabilities["TabTransformer"] = predict_mlp(
                tabtransformer,
                deep_data.X_cat_val,
                deep_data.X_num_val,
                batch_size=deep_batch_size,
                device=torch_device,
            )
            test_probabilities["TabTransformer"] = predict_mlp(
                tabtransformer,
                deep_data.X_cat_test,
                deep_data.X_num_test,
                batch_size=deep_batch_size,
                device=torch_device,
            )
            runtime_seconds = time.perf_counter() - started_at
            thresholds["TabTransformer"] = _add_threshold_rows(
                rows,
                "TabTransformer",
                deep_data.y_val,
                val_probabilities["TabTransformer"],
                deep_data.y_test,
                test_probabilities["TabTransformer"],
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        except Exception as exc:
            print(f"TabTransformer skipped/failed: {exc}")

        try:
            started_at = time.perf_counter()
            tabm, _ = train_mlp(
                deep_data,
                epochs=model_epochs,
                patience=2 if quick else 5,
                batch_size=deep_batch_size,
                device=torch_device,
                architecture="tabm",
                loss_name="focal",
                model_name="tabm",
            )
            val_probabilities["TabM"] = predict_mlp(
                tabm, deep_data.X_cat_val, deep_data.X_num_val, batch_size=deep_batch_size, device=torch_device
            )
            test_probabilities["TabM"] = predict_mlp(
                tabm, deep_data.X_cat_test, deep_data.X_num_test, batch_size=deep_batch_size, device=torch_device
            )
            runtime_seconds = time.perf_counter() - started_at
            thresholds["TabM"] = _add_threshold_rows(
                rows,
                "TabM-Style Ensemble MLP",
                deep_data.y_val,
                val_probabilities["TabM"],
                deep_data.y_test,
                test_probabilities["TabM"],
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        except Exception as exc:
            print(f"TabM skipped/failed: {exc}")

    if not skip_tabnet:
        try:
            started_at = time.perf_counter()
            tabnet = train_tabnet(
                deep_data,
                quick=quick,
                use_pretraining=use_tabnet_pretraining and not quick,
                batch_size=deep_batch_size,
                device=torch_device,
            )
            val_probabilities["TabNet"] = predict_tabnet(tabnet, deep_data.X_cat_val, deep_data.X_num_val)
            test_probabilities["TabNet"] = predict_tabnet(tabnet, deep_data.X_cat_test, deep_data.X_num_test)
            runtime_seconds = time.perf_counter() - started_at
            tabnet_name = "TabNet Pretrained" if use_tabnet_pretraining and not quick else "TabNet"
            thresholds["TabNet"] = _add_threshold_rows(
                rows,
                tabnet_name,
                deep_data.y_val,
                val_probabilities["TabNet"],
                deep_data.y_test,
                test_probabilities["TabNet"],
                threshold_strategy,
                min_recall,
                include_recall_target,
                runtime_seconds,
            )
        except Exception as exc:
            print(f"TabNet skipped/failed: {exc}")

    _add_weighted_ensemble(
        rows,
        val_probabilities,
        test_probabilities,
        thresholds,
        "XGBoost+TabNet",
        ["XGBoost", "TabNet"],
        sklearn_data.y_val,
        sklearn_data.y_test,
        threshold_strategy,
        min_recall,
        include_recall_target,
        quick,
    )
    _add_weighted_ensemble(
        rows,
        val_probabilities,
        test_probabilities,
        thresholds,
        "XGBoost+LightGBM",
        ["XGBoost", "LightGBM"],
        sklearn_data.y_val,
        sklearn_data.y_test,
        threshold_strategy,
        min_recall,
        include_recall_target,
        quick,
    )
    _add_weighted_ensemble(
        rows,
        val_probabilities,
        test_probabilities,
        thresholds,
        "XGBoost+CatBoost",
        ["XGBoost", "CatBoost"],
        sklearn_data.y_val,
        sklearn_data.y_test,
        threshold_strategy,
        min_recall,
        include_recall_target,
        quick,
    )
    _add_weighted_ensemble(
        rows,
        val_probabilities,
        test_probabilities,
        thresholds,
        "XGBoost+LightGBM+CatBoost",
        ["XGBoost", "LightGBM", "CatBoost"],
        sklearn_data.y_val,
        sklearn_data.y_test,
        threshold_strategy,
        min_recall,
        include_recall_target,
        quick,
    )

    for deep_name in ["TabTransformer", "TabM"]:
        _add_weighted_ensemble(
            rows,
            val_probabilities,
            test_probabilities,
            thresholds,
            f"XGBoost+{deep_name}",
            ["XGBoost", deep_name],
            sklearn_data.y_val,
            sklearn_data.y_test,
            threshold_strategy,
            min_recall,
            include_recall_target,
            quick,
        )

    deep_candidates = [name for name in ["MLP", "TabNet", "TabTransformer", "TabM"] if name in val_probabilities]
    if "XGBoost" in val_probabilities and deep_candidates:
        best_deep = max(
            deep_candidates,
            key=lambda name: average_precision_score(sklearn_data.y_val, val_probabilities[name]),
        )
        _add_weighted_ensemble(
            rows,
            val_probabilities,
            test_probabilities,
            thresholds,
            f"XGBoost+BestDeep({best_deep})",
            ["XGBoost", best_deep],
            sklearn_data.y_val,
            sklearn_data.y_test,
            threshold_strategy,
            min_recall,
            include_recall_target,
            quick,
        )

    base_ensemble_candidates = [
        name
        for name in [
            "Logistic Regression",
            "XGBoost",
            "XGBoost Calibrated",
            "LightGBM",
            "CatBoost",
            "MLP",
            "TabNet",
            "TabTransformer",
            "TabM",
        ]
        if name in val_probabilities and name in test_probabilities
    ]
    if len(base_ensemble_candidates) >= 2:
        _add_weighted_ensemble(
            rows,
            val_probabilities,
            test_probabilities,
            thresholds,
            "AllAvailableModels",
            base_ensemble_candidates,
            sklearn_data.y_val,
            sklearn_data.y_test,
            threshold_strategy,
            min_recall,
            include_recall_target,
            quick,
        )

    if not skip_stacking and "XGBoost" in val_probabilities:
        stack_names = base_ensemble_candidates
        if len(stack_names) >= 2:
            X_stack_val = np.column_stack([val_probabilities[name] for name in stack_names])
            X_stack_test = np.column_stack([test_probabilities[name] for name in stack_names])
            for c_value in [0.1, 1.0, 10.0]:
                started_at = time.perf_counter()
                stacker = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                        solver="liblinear",
                    ),
                )
                stacker.fit(X_stack_val, sklearn_data.y_val)
                key = f"Stacking LogReg C={c_value:g}"
                joblib.dump(
                    {"model": stacker, "base_models": stack_names},
                    MODELS_DIR / f"stacking_logreg_c_{str(c_value).replace('.', '_')}.joblib",
                )
                val_probabilities[key] = stacker.predict_proba(X_stack_val)[:, 1]
                test_probabilities[key] = stacker.predict_proba(X_stack_test)[:, 1]
                thresholds[key] = _add_threshold_rows(
                    rows,
                    f"Stacking Logistic Regression C={c_value:g} ({'+'.join(stack_names)})",
                    sklearn_data.y_val,
                    val_probabilities[key],
                    sklearn_data.y_test,
                    test_probabilities[key],
                    threshold_strategy,
                    min_recall,
                    include_recall_target,
                    time.perf_counter() - started_at,
                )

            try:
                from lightgbm import LGBMClassifier

                started_at = time.perf_counter()
                lgbm_stacker = LGBMClassifier(
                    objective="binary",
                    n_estimators=80 if quick else 180,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=20,
                    scale_pos_weight=float(np.sum(sklearn_data.y_val == 0) / max(np.sum(sklearn_data.y_val == 1), 1)),
                    random_state=42,
                    n_jobs=1,
                    verbose=-1,
                )
                lgbm_stacker.fit(X_stack_val, sklearn_data.y_val, eval_set=[(X_stack_val, sklearn_data.y_val)])
                key = "Stacking LightGBM Meta"
                joblib.dump(
                    {"model": lgbm_stacker, "base_models": stack_names},
                    MODELS_DIR / "stacking_lightgbm_meta.joblib",
                )
                val_probabilities[key] = lgbm_stacker.predict_proba(X_stack_val)[:, 1]
                test_probabilities[key] = lgbm_stacker.predict_proba(X_stack_test)[:, 1]
                thresholds[key] = _add_threshold_rows(
                    rows,
                    f"Stacking LightGBM Meta ({'+'.join(stack_names)})",
                    sklearn_data.y_val,
                    val_probabilities[key],
                    sklearn_data.y_test,
                    test_probabilities[key],
                    threshold_strategy,
                    min_recall,
                    include_recall_target,
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                print(f"LightGBM meta-stacker skipped/failed: {exc}")

    ablation_df = _save_ablation_study(
        sklearn_data.y_val,
        sklearn_data.y_test,
        val_probabilities,
        test_probabilities,
        min_recall,
        quick,
    )
    metrics_df = save_metrics(rows, METRICS_PATH, append=False)
    _save_prediction_files(sklearn_data.y_val, sklearn_data.y_test, val_probabilities, test_probabilities)
    _save_thresholds(metrics_df)
    rankings_df = _save_model_rankings(metrics_df, min_recall)

    fairness_frames = []
    for model_name, y_proba in test_probabilities.items():
        if model_name in thresholds:
            fairness_frames.append(
                evaluate_subgroups(
                    split.X_test,
                    sklearn_data.y_test,
                    y_proba,
                    model_name,
                    thresholds[model_name],
                    columns=["age", "gender"],
                )
            )
    if fairness_frames:
        fairness_df = pd.concat(fairness_frames, ignore_index=True)
        fairness_df.to_csv(RESULTS_DIR / f"fairness_{split_strategy}.csv", index=False)

    for model_name, y_proba in test_probabilities.items():
        stem = _slug(model_name)
        plot_roc_curve(sklearn_data.y_test, y_proba, model_name, FIGURES_DIR / f"{stem}_roc.png")
        plot_precision_recall_curve(sklearn_data.y_test, y_proba, model_name, FIGURES_DIR / f"{stem}_pr.png")
        if model_name in thresholds:
            plot_confusion_matrix(
                sklearn_data.y_test,
                y_proba,
                model_name,
                thresholds[model_name],
                FIGURES_DIR / f"{stem}_confusion_matrix.png",
            )
        plot_recall_threshold_curve(
            sklearn_data.y_test,
            y_proba,
            model_name,
            FIGURES_DIR / f"{stem}_threshold_curve.png",
        )

    if not metrics_df.empty:
        plot_metric_comparison(metrics_df, "pr_auc", "test", FIGURES_DIR / "pr_auc_comparison.png")
        plot_metric_comparison(metrics_df, "test_pr_auc", "test", FIGURES_DIR / "test_pr_auc_comparison.png")
        plot_metric_comparison(metrics_df, "recall", "test", FIGURES_DIR / "recall_comparison.png")
        plot_metric_comparison(metrics_df, "roc_auc", "test", FIGURES_DIR / "roc_auc_comparison.png")
        plot_combined_precision_recall_curves(
            sklearn_data.y_test,
            test_probabilities,
            FIGURES_DIR / "pr_curves_all_models.png",
        )
        plot_combined_roc_curves(
            sklearn_data.y_test,
            test_probabilities,
            FIGURES_DIR / "roc_curves_all_models.png",
        )

    recall_target_mask = metrics_df["recall_target"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    recall_rows = metrics_df[
        (metrics_df["split"] == "test")
        & recall_target_mask
    ].sort_values("test_pr_auc", ascending=False)
    for _, row in recall_rows.head(3).iterrows():
        key = _probability_key_for_metrics_model(str(row["model"]), test_probabilities)
        if key is None:
            continue
        plot_confusion_matrix(
            sklearn_data.y_test,
            test_probabilities[key],
            str(row["model"]),
            float(row["threshold"]),
            FIGURES_DIR / f"top_recall_target_{_slug(str(row['model']))}_confusion_matrix.png",
        )

    if not rankings_df.empty:
        print("Top models by test PR-AUC:")
        print(
            rankings_df[rankings_df["ranking"] == "test_pr_auc"]
            .head(10)[["rank", "model", "test_pr_auc", "test_roc_auc", "precision", "recall", "f1"]]
            .to_string(index=False)
        )
    if not ablation_df.empty:
        print("Ablation study:")
        print(
            ablation_df[["ablation", "status", "test_pr_auc", "precision", "recall", "threshold"]]
            .to_string(index=False)
        )

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full readmission experiment.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size.")
    parser.add_argument("--quick", action="store_true", help="Use shorter training for smoke tests.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing data/raw/diabetic_data.csv.")
    parser.add_argument("--skip-tabnet", action="store_true", help="Skip TabNet even if installed.")
    parser.add_argument("--skip-modern-dl", action="store_true", help="Skip TabTransformer and TabM-style MLP.")
    parser.add_argument("--benchmark-device", action="store_true", help="Benchmark deep models on CPU vs available GPU.")
    parser.add_argument("--tune-xgb", action="store_true", help="Tune XGBoost on validation PR-AUC.")
    parser.add_argument("--xgb-tune-iter", type=int, default=20)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--include-lightgbm", action="store_true", help="Train optional LightGBM baseline.")
    parser.add_argument("--lgbm-tune-iter", type=int, default=0, help="LightGBM validation random-search iterations.")
    parser.add_argument("--include-catboost", action="store_true", help="Train optional CatBoost baseline.")
    parser.add_argument("--catboost-tune-iter", type=int, default=0, help="CatBoost validation random-search iterations.")
    parser.add_argument("--no-calibrate-xgb", action="store_true")
    parser.add_argument("--threshold-strategy", choices=["max_f1", "recall_target", "fixed_0.5"], default="max_f1")
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--no-recall-target-rows", action="store_true")
    parser.add_argument("--tabnet-pretraining", action="store_true")
    parser.add_argument("--split-strategy", choices=["encounter", "patient"], default="encounter")
    parser.add_argument("--no-advanced-features", action="store_true", help="Disable split-level FE ablation.")
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--modern-dl-epochs", type=int, default=None)
    parser.add_argument("--deep-batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.benchmark_device:
        metrics_df = run_device_benchmark(
            sample_size=args.sample_size,
            quick=args.quick,
            skip_download=args.skip_download,
            skip_tabnet=args.skip_tabnet,
            skip_modern_dl=args.skip_modern_dl,
            split_strategy=args.split_strategy,
            advanced_feature_engineering=not args.no_advanced_features,
            deep_batch_size=args.deep_batch_size,
            modern_dl_epochs=args.modern_dl_epochs,
            min_recall=args.min_recall,
        )
        print(metrics_df)
        return

    metrics_df = run_experiment(
        sample_size=args.sample_size,
        quick=args.quick,
        skip_download=args.skip_download,
        skip_tabnet=args.skip_tabnet,
        skip_modern_dl=args.skip_modern_dl,
        tune_xgb=args.tune_xgb,
        xgb_tune_iter=args.xgb_tune_iter,
        xgb_n_jobs=args.xgb_n_jobs,
        include_lightgbm=args.include_lightgbm,
        lgbm_tune_iter=args.lgbm_tune_iter,
        include_catboost=args.include_catboost,
        catboost_tune_iter=args.catboost_tune_iter,
        calibrate_xgb=not args.no_calibrate_xgb,
        threshold_strategy=args.threshold_strategy,
        min_recall=args.min_recall,
        include_recall_target=not args.no_recall_target_rows,
        use_tabnet_pretraining=args.tabnet_pretraining,
        split_strategy=args.split_strategy,
        advanced_feature_engineering=not args.no_advanced_features,
        skip_stacking=args.skip_stacking,
        torch_device=None if args.device == "auto" else args.device,
        modern_dl_epochs=args.modern_dl_epochs,
        deep_batch_size=args.deep_batch_size,
    )
    print(metrics_df)


if __name__ == "__main__":
    main()
