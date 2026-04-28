# Improving 30-Day Hospital Readmission Prediction Using TabNet and Ensemble Learning

This project predicts whether a diabetes-related hospital encounter will result in readmission within 30 days using the UCI Diabetes 130-US Hospitals dataset.

The binary target is created from `readmitted`:

- `<30` -> `1`
- `NO` and `>30` -> `0`

The main research question is whether TabNet can outperform strong tabular baselines such as XGBoost for clinically important 30-day readmission prediction. Evaluation emphasizes recall and PR-AUC because the positive class is important and imbalanced.

## Dataset

Dataset page: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>

The downloader tries the UCI archive URL and extracts `diabetic_data.csv` into `data/raw/`. If the UCI direct download fails, download the dataset manually from the page above and place `diabetic_data.csv` in `data/raw/`.

## Setup

Local setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.data_download
```

LightGBM and CatBoost are included in `requirements.txt`. If an environment was created before they were added, install them explicitly:

```bash
python -m pip install lightgbm catboost
python -c "import lightgbm, catboost; print('boosting extras ok')"
```

Python 3.10-3.12 is recommended. On macOS, keep native thread pools small for stability:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Large PyTorch batches such as `--deep-batch-size 2048` caused local native runtime/resource instability on this Mac. The runner now defaults to CPU and `--deep-batch-size 64`; use 64 or 128 locally unless you have verified that larger batches are stable.

Google Colab setup:

```python
from google.colab import drive
drive.mount("/content/drive")  # optional

!git clone <your-repo-url>
%cd hospital-readmission
!pip install -r requirements.txt
```

If cloning is not used, upload the project folder to Drive and `%cd` into it. If the automatic dataset download fails in Colab, upload `diabetic_data.csv` manually:

```python
from src.data_download import colab_upload_dataset
colab_upload_dataset()
```

Check GPU availability:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Run the Project

Quick smoke test on a stratified subset:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -X faulthandler -m src.run_experiment \
  --sample-size 2500 \
  --quick \
  --device cpu \
  --deep-batch-size 64 \
  --modern-dl-epochs 1
```

Stable local CPU run:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -X faulthandler -m src.run_experiment \
  --skip-download \
  --min-recall 0.6 \
  --device cpu \
  --deep-batch-size 64 \
  --modern-dl-epochs 1
```

Strong local experiment with tuned boosted baselines:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -X faulthandler -m src.run_experiment \
  --skip-download \
  --tune-xgb \
  --xgb-tune-iter 20 \
  --include-lightgbm \
  --lgbm-tune-iter 20 \
  --include-catboost \
  --catboost-tune-iter 20 \
  --min-recall 0.6 \
  --device cpu \
  --deep-batch-size 64 \
  --modern-dl-epochs 20
```

The runner logs and skips LightGBM or CatBoost if either optional native package is unavailable. Use the explicit install command above before running `--include-lightgbm` or `--include-catboost`.

Optional Apple MPS run. MPS is only used for PyTorch models, not XGBoost/LightGBM/CatBoost. If PyTorch reports MPS as available, enable fallback for unsupported ops:

```bash
source .venv312/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -X faulthandler -m src.run_experiment \
  --skip-download \
  --tune-xgb \
  --xgb-tune-iter 6 \
  --min-recall 0.6 \
  --device mps \
  --deep-batch-size 64
```

Check MPS availability:

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
PY
```

Benchmark CPU vs the available GPU backend for deep models only:

```bash
source .venv312/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -X faulthandler -m src.run_experiment \
  --skip-download \
  --benchmark-device \
  --sample-size 2000 \
  --quick \
  --device mps \
  --deep-batch-size 128 \
  --modern-dl-epochs 1
```

The benchmark writes CPU/GPU rows to `results/metrics.csv` with `runtime_seconds`, `test_pr_auc`, and `device`.

Patient-level split sensitivity run:

```bash
python -X faulthandler -m src.run_experiment \
  --skip-download \
  --split-strategy patient \
  --tune-xgb \
  --xgb-tune-iter 6 \
  --min-recall 0.6 \
  --device cpu \
  --deep-batch-size 64
```

Optional TabNet unsupervised pretraining is implemented, but it adds runtime:

```bash
python -m src.run_experiment --skip-download --tune-xgb --xgb-tune-iter 6 --tabnet-pretraining
```

If `pytorch-tabnet` is unavailable, keep the rest of the experiment running:

```bash
python -m src.run_experiment --skip-tabnet
```

The notebooks provide the course-report workflow:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_baselines.ipynb`
3. `notebooks/03_mlp_tabnet.ipynb`
4. `notebooks/04_results_analysis.ipynb`

## Models

- Logistic Regression with `class_weight="balanced"`
- XGBoost with `scale_pos_weight`
- Tuned XGBoost optimized on validation PR-AUC
- Calibrated XGBoost using validation-set isotonic calibration
- Optional LightGBM with `scale_pos_weight`, validation PR-AUC tuning, and `--include-lightgbm`
- Optional CatBoost with `Logloss`, `PRAUC`, class weights, quiet training, and `--include-catboost`
- PyTorch residual MLP with categorical embeddings, focal loss, and validation PR-AUC early stopping
- TabNet using `pytorch-tabnet`, with optional unsupervised pretraining
- TabTransformer implemented in PyTorch
- TabM-style shared-trunk, multi-head ensemble MLP implemented in PyTorch
- Optional TabPFN wrapper in `src/train_tabpfn.py`; not enabled by default because the official package is optional and full one-hot UCI data can exceed practical TabPFN limits
- TabR is documented as skipped because a faithful implementation would require a dedicated retrieval/indexing pipeline beyond this project scope
- Validation-weighted XGBoost + TabNet ensemble
- Validation-weighted XGBoost + TabTransformer and XGBoost + TabM ensembles
- Validation-weighted XGBoost + LightGBM, XGBoost + CatBoost, XGBoost + LightGBM + CatBoost, XGBoost + best deep model, and all-available-model ensembles
- Standardized logistic-regression stacking over validation-only model probabilities with multiple regularization strengths
- Optional LightGBM meta-stacking over validation-only model probabilities

## Feature Engineering

The preprocessing pipeline adds leakage-safe row-level features before splitting:

- broad ICD-9 diagnosis groups for `diag_1`, `diag_2`, and `diag_3`
- prior utilization counts and binary inpatient/emergency history flags
- medication active/change counts
- insulin usage/change flags
- medications and lab procedures per hospital day
- split-level missing-value indicators fitted on the training split
- split-level log transforms for skewed nonnegative numeric features
- split-level pairwise interactions among the top training-correlated numeric features

## Metrics

Each model is evaluated using predicted probabilities and a validation-selected threshold. PR-AUC is the primary ranking metric because the positive class, readmitted within 30 days, is clinically important and imbalanced.

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall / sensitivity
- F1 score
- Confusion matrix

The default helper supports threshold tuning to maximize validation F1 or meet a recall target. For recall-target rows, the threshold is selected on the validation set as the highest-precision threshold satisfying `recall >= --min-recall`. The test set is used only after the threshold is fixed. `results/metrics.csv` and `results/thresholds.csv` use `recall_target` as a boolean row marker and `min_recall` for the numeric target.

## Outputs

- `results/metrics.csv`: model comparison table
- `results/thresholds.csv`: selected thresholds and threshold-specific confusion counts
- `results/model_rankings.csv`: models sorted by test PR-AUC and precision at the recall target
- `results/model_ablation.csv`: XGBoost-only and ensemble ablations using validation-tuned weights
- `results/predictions/`: validation and test probabilities for each model
- `results/figures/`: class distribution, ROC curves, PR curves, confusion matrices, feature importance, metric comparison plots
- `results/models/`: trained model artifacts and preprocessors

## Known Limitations

- The default split is encounter-level, so the same patient may appear multiple times. Use `--split-strategy patient` for a stricter patient-disjoint sensitivity analysis.
- Coded diagnosis fields are treated as categorical strings rather than mapped to clinical ontologies.
- TabNet may require compatible PyTorch and CUDA versions; Colab is the recommended environment if local dependency resolution fails.
- LightGBM and CatBoost are optional native dependencies. If either package is unavailable or fails to load on macOS, the runner logs the failure and continues with the remaining models.
- Thresholds are selected on the validation split and should be rechecked before any clinical deployment.
