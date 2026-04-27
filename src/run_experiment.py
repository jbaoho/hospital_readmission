"""End-to-end experiment runner for baselines, deep models, plots, and ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, RESULTS_DIR, ensure_directories
from .data_download import download_dataset
from .evaluate import (
    calibrate_probabilities_isotonic,
    choose_threshold,
    evaluate_subgroups,
    metrics_row,
    save_metrics,
    tune_weighted_average,
)
from .plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curve,
    plot_recall_threshold_curve,
    plot_roc_curve,
    plot_xgboost_feature_importance,
)
from .preprocessing import fit_transform_deep, fit_transform_sklearn, prepare_splits
from .train_baselines import predict_proba, train_logistic_regression, train_xgboost, tune_xgboost
from .train_deep import predict_mlp, predict_tabnet, train_mlp, train_tabnet


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
) -> float:
    threshold, source = choose_threshold(y_val, val_proba, threshold_strategy, min_recall)
    rows.append(metrics_row(model_name, "test", y_test, test_proba, threshold, source))

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
            )
        )
    return threshold


def run_experiment(
    sample_size: int | None = None,
    quick: bool = False,
    skip_download: bool = False,
    skip_tabnet: bool = False,
    skip_modern_dl: bool = False,
    tune_xgb: bool = False,
    xgb_tune_iter: int = 20,
    xgb_n_jobs: int = -1,
    calibrate_xgb: bool = True,
    threshold_strategy: str = "max_f1",
    min_recall: float = 0.6,
    include_recall_target: bool = True,
    use_tabnet_pretraining: bool = False,
    split_strategy: str = "encounter",
    skip_stacking: bool = False,
    torch_device: str | None = None,
    modern_dl_epochs: int | None = None,
    deep_batch_size: int = 1024,
) -> pd.DataFrame:
    """Run all available models and save metrics/figures."""
    ensure_directories()
    if not skip_download:
        download_dataset()

    split = prepare_splits(sample_size=sample_size, split_strategy=split_strategy)
    plot_class_distribution(split.y_train, FIGURES_DIR / "class_distribution_train.png")

    rows: list[dict] = []
    test_probabilities: dict[str, np.ndarray] = {}
    val_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    sklearn_data = fit_transform_sklearn(split)
    joblib.dump(sklearn_data.preprocessor, MODELS_DIR / "sklearn_preprocessor.joblib")

    logreg = train_logistic_regression(sklearn_data.X_train, sklearn_data.y_train)
    val_probabilities["Logistic Regression"] = predict_proba(logreg, sklearn_data.X_val)
    test_probabilities["Logistic Regression"] = predict_proba(logreg, sklearn_data.X_test)
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
    )

    try:
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
        )

        if calibrate_xgb:
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
            )
        plot_xgboost_feature_importance(
            xgb,
            sklearn_data.feature_names,
            FIGURES_DIR / "xgboost_feature_importance.png",
            top_n=25,
        )
    except Exception as exc:
        print(f"XGBoost skipped/failed: {exc}")

    deep_data = fit_transform_deep(split)
    joblib.dump(deep_data, MODELS_DIR / "deep_preprocessing.joblib")

    mlp, _ = train_mlp(
        deep_data,
        epochs=3 if quick else 30,
        patience=2 if quick else 5,
        batch_size=deep_batch_size,
        device=torch_device,
    )
    val_probabilities["MLP"] = predict_mlp(mlp, deep_data.X_cat_val, deep_data.X_num_val)
    test_probabilities["MLP"] = predict_mlp(mlp, deep_data.X_cat_test, deep_data.X_num_test)
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
    )

    if not skip_modern_dl:
        model_epochs = modern_dl_epochs if modern_dl_epochs is not None else (3 if quick else 8)
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
            tabtransformer, deep_data.X_cat_val, deep_data.X_num_val
        )
        test_probabilities["TabTransformer"] = predict_mlp(
            tabtransformer, deep_data.X_cat_test, deep_data.X_num_test
        )
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
        )

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
        val_probabilities["TabM"] = predict_mlp(tabm, deep_data.X_cat_val, deep_data.X_num_val)
        test_probabilities["TabM"] = predict_mlp(tabm, deep_data.X_cat_test, deep_data.X_num_test)
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
        )

    if not skip_tabnet:
        try:
            tabnet = train_tabnet(deep_data, quick=quick, use_pretraining=use_tabnet_pretraining and not quick)
            val_probabilities["TabNet"] = predict_tabnet(tabnet, deep_data.X_cat_val, deep_data.X_num_val)
            test_probabilities["TabNet"] = predict_tabnet(tabnet, deep_data.X_cat_test, deep_data.X_num_test)
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
            )
        except Exception as exc:
            print(f"TabNet skipped/failed: {exc}")

    if "XGBoost" in test_probabilities and "TabNet" in test_probabilities:
        weights, val_ensemble, ensemble_score = tune_weighted_average(
            sklearn_data.y_val,
            {"XGBoost": val_probabilities["XGBoost"], "TabNet": val_probabilities["TabNet"]},
            metric="pr_auc",
            step=0.02,
        )
        test_ensemble = weights["XGBoost"] * test_probabilities["XGBoost"] + weights["TabNet"] * test_probabilities[
            "TabNet"
        ]
        test_probabilities["Weighted Ensemble"] = test_ensemble
        ensemble_name = f"Weighted XGBoost+TabNet Ensemble (xgb={weights['XGBoost']:.2f})"
        print(f"weighted_ensemble_val_pr_auc={ensemble_score:.5f} weights={weights}")
        thresholds["Weighted Ensemble"] = _add_threshold_rows(
            rows,
            ensemble_name,
            sklearn_data.y_val,
            val_ensemble,
            sklearn_data.y_test,
            test_ensemble,
            threshold_strategy,
            min_recall,
            include_recall_target,
        )

    for deep_name in ["TabTransformer", "TabM"]:
        if "XGBoost" in test_probabilities and deep_name in test_probabilities:
            weights, val_ensemble, ensemble_score = tune_weighted_average(
                sklearn_data.y_val,
                {"XGBoost": val_probabilities["XGBoost"], deep_name: val_probabilities[deep_name]},
                metric="pr_auc",
                step=0.02,
            )
            test_ensemble = (
                weights["XGBoost"] * test_probabilities["XGBoost"]
                + weights[deep_name] * test_probabilities[deep_name]
            )
            key = f"Weighted XGBoost+{deep_name}"
            test_probabilities[key] = test_ensemble
            print(f"{key}_val_pr_auc={ensemble_score:.5f} weights={weights}")
            thresholds[key] = _add_threshold_rows(
                rows,
                f"Weighted XGBoost+{deep_name} Ensemble (xgb={weights['XGBoost']:.2f})",
                sklearn_data.y_val,
                val_ensemble,
                sklearn_data.y_test,
                test_ensemble,
                threshold_strategy,
                min_recall,
                include_recall_target,
            )

    if not skip_stacking and "XGBoost" in val_probabilities:
        stack_names = [
            name
            for name in ["XGBoost", "TabNet", "TabTransformer", "TabM", "MLP"]
            if name in val_probabilities and name in test_probabilities
        ]
        if len(stack_names) >= 2:
            X_stack_val = np.column_stack([val_probabilities[name] for name in stack_names])
            X_stack_test = np.column_stack([test_probabilities[name] for name in stack_names])
            stacker = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            stacker.fit(X_stack_val, sklearn_data.y_val)
            joblib.dump({"model": stacker, "base_models": stack_names}, MODELS_DIR / "stacking_logreg.joblib")
            val_probabilities["Stacking"] = stacker.predict_proba(X_stack_val)[:, 1]
            test_probabilities["Stacking"] = stacker.predict_proba(X_stack_test)[:, 1]
            thresholds["Stacking"] = _add_threshold_rows(
                rows,
                f"Stacking Logistic Regression ({'+'.join(stack_names)})",
                sklearn_data.y_val,
                val_probabilities["Stacking"],
                sklearn_data.y_test,
                test_probabilities["Stacking"],
                threshold_strategy,
                min_recall,
                include_recall_target,
            )

    metrics_df = save_metrics(rows, METRICS_PATH, append=False)

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
        stem = model_name.lower().replace(" ", "_")
        plot_roc_curve(sklearn_data.y_test, y_proba, model_name, FIGURES_DIR / f"{stem}_roc.png")
        plot_precision_recall_curve(sklearn_data.y_test, y_proba, model_name, FIGURES_DIR / f"{stem}_pr.png")
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
        plot_metric_comparison(metrics_df, "recall", "test", FIGURES_DIR / "recall_comparison.png")
        plot_metric_comparison(metrics_df, "roc_auc", "test", FIGURES_DIR / "roc_auc_comparison.png")

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full readmission experiment.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size.")
    parser.add_argument("--quick", action="store_true", help="Use shorter training for smoke tests.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing data/raw/diabetic_data.csv.")
    parser.add_argument("--skip-tabnet", action="store_true", help="Skip TabNet even if installed.")
    parser.add_argument("--skip-modern-dl", action="store_true", help="Skip TabTransformer and TabM-style MLP.")
    parser.add_argument("--tune-xgb", action="store_true", help="Tune XGBoost on validation PR-AUC.")
    parser.add_argument("--xgb-tune-iter", type=int, default=20)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--no-calibrate-xgb", action="store_true")
    parser.add_argument("--threshold-strategy", choices=["max_f1", "recall_target", "fixed_0.5"], default="max_f1")
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--no-recall-target-rows", action="store_true")
    parser.add_argument("--tabnet-pretraining", action="store_true")
    parser.add_argument("--split-strategy", choices=["encounter", "patient"], default="encounter")
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--modern-dl-epochs", type=int, default=None)
    parser.add_argument("--deep-batch-size", type=int, default=1024)
    args = parser.parse_args()

    metrics_df = run_experiment(
        sample_size=args.sample_size,
        quick=args.quick,
        skip_download=args.skip_download,
        skip_tabnet=args.skip_tabnet,
        skip_modern_dl=args.skip_modern_dl,
        tune_xgb=args.tune_xgb,
        xgb_tune_iter=args.xgb_tune_iter,
        xgb_n_jobs=args.xgb_n_jobs,
        calibrate_xgb=not args.no_calibrate_xgb,
        threshold_strategy=args.threshold_strategy,
        min_recall=args.min_recall,
        include_recall_target=not args.no_recall_target_rows,
        use_tabnet_pretraining=args.tabnet_pretraining,
        split_strategy=args.split_strategy,
        skip_stacking=args.skip_stacking,
        torch_device=None if args.device == "auto" else args.device,
        modern_dl_epochs=args.modern_dl_epochs,
        deep_batch_size=args.deep_batch_size,
    )
    print(metrics_df)


if __name__ == "__main__":
    main()
