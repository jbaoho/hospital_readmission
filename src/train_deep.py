"""PyTorch MLP and TabNet training utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import MODELS_DIR, RANDOM_STATE, ensure_directories, get_device
from .evaluate import choose_threshold_max_f1, metrics_row, save_metrics
from .preprocessing import combined_tabnet_matrix, fit_transform_deep, prepare_splits
from .train_baselines import positive_class_weight


def set_torch_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device through the project-wide device helper."""
    if requested is None or str(requested).lower() == "auto":
        return get_device(prefer_gpu=True)

    requested_name = str(requested).lower()
    if requested_name == "cpu":
        return get_device(prefer_gpu=False)
    if requested_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if requested_name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda, mps")


def verify_device(model: nn.Module, x: torch.Tensor) -> None:
    """Print model and input device placement for the first train/inference batch."""
    print("Model device:", next(model.parameters()).device)
    print("Input device:", x.device)


def _prepare_batch(
    X_cat: torch.Tensor,
    X_num: torch.Tensor,
    y: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    X_cat = X_cat.to(device=device, dtype=torch.long).contiguous()
    X_num = X_num.to(device=device, dtype=torch.float32).contiguous()
    y_out = None if y is None else y.to(device=device, dtype=torch.float32).contiguous()
    return X_cat, X_num, y_out


class TabularDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = torch.as_tensor(np.ascontiguousarray(X_cat), dtype=torch.long)
        self.X_num = torch.as_tensor(np.ascontiguousarray(X_num, dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """Embedding-based MLP for mixed tabular features."""

    def __init__(
        self,
        cat_dims: list[int],
        num_features: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(50, max(2, (dim + 1) // 2))) for dim in cat_dims]
        )
        embedding_dim = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = embedding_dim + num_features

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embedded = [emb(X_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            x = torch.cat(embedded + [X_num], dim=1)
        else:
            x = X_num
        return self.network(x).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLPClassifier(nn.Module):
    """Residual tabular MLP with categorical embeddings and numeric projection."""

    def __init__(
        self,
        cat_dims: list[int],
        num_features: int,
        width: int = 256,
        depth: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(64, max(4, int(round(dim**0.5)) + 1))) for dim in cat_dims]
        )
        embedding_dim = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = embedding_dim + num_features
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(width, width * 2, dropout) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embedded = [emb(X_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            x = torch.cat(embedded + [X_num], dim=1)
        else:
            x = X_num
        return self.output(self.blocks(self.input(x))).squeeze(1)


class TabTransformerClassifier(nn.Module):
    """TabTransformer-style encoder for categorical tokens plus numeric features."""

    def __init__(
        self,
        cat_dims: list[int],
        num_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in cat_dims])
        self.has_cats = len(cat_dims) > 0
        if self.has_cats:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.transformer = None

        self.num_projection = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        classifier_input = d_model * (2 if self.has_cats else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        num_repr = self.num_projection(X_num)
        if self.has_cats:
            tokens = torch.stack([emb(X_cat[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1)
            cat_repr = self.transformer(tokens).mean(dim=1)
            x = torch.cat([cat_repr, num_repr], dim=1)
        else:
            x = num_repr
        return self.classifier(x).squeeze(1)


class TabMStyleClassifier(nn.Module):
    """Approximate TabM with a shared trunk and multiple prediction heads."""

    def __init__(
        self,
        cat_dims: list[int],
        num_features: int,
        width: int = 256,
        depth: int = 3,
        dropout: float = 0.25,
        n_heads: int = 8,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(64, max(4, int(round(dim**0.5)) + 1))) for dim in cat_dims]
        )
        embedding_dim = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = embedding_dim + num_features
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[ResidualBlock(width, width * 2, dropout) for _ in range(depth)],
            nn.LayerNorm(width),
        )
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_heads)])

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embedded = [emb(X_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            x = torch.cat(embedded + [X_num], dim=1)
        else:
            x = X_num
        features = self.trunk(x)
        logits = torch.stack([head(features).squeeze(1) for head in self.heads], dim=1)
        return logits.mean(dim=1)


class FocalLossWithLogits(nn.Module):
    """Binary focal loss for imbalanced classification."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


def train_mlp(
    prepared,
    model_dir: Path = MODELS_DIR,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
    architecture: str = "residual",
    loss_name: str = "focal",
    model_name: str | None = None,
    allow_device_fallback: bool = True,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train an MLP with validation PR-AUC early stopping."""
    ensure_directories()
    set_torch_seed()
    device = get_torch_device(device)
    print(f"MLP device: {device}")

    train_loader = DataLoader(
        TabularDataset(prepared.X_cat_train, prepared.X_num_train, prepared.y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        TabularDataset(prepared.X_cat_val, prepared.X_num_val, prepared.y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    if architecture == "residual":
        model = ResidualMLPClassifier(prepared.cat_dims, prepared.X_num_train.shape[1]).to(device)
    elif architecture == "plain":
        model = MLPClassifier(prepared.cat_dims, prepared.X_num_train.shape[1]).to(device)
    elif architecture == "tabtransformer":
        model = TabTransformerClassifier(prepared.cat_dims, prepared.X_num_train.shape[1]).to(device)
    elif architecture == "tabm":
        model = TabMStyleClassifier(prepared.cat_dims, prepared.X_num_train.shape[1]).to(device)
    else:
        raise ValueError(f"Unknown MLP architecture: {architecture}")
    print("Model device:", next(model.parameters()).device)

    if loss_name == "focal":
        criterion = FocalLossWithLogits(alpha=0.75, gamma=2.0)
    elif loss_name == "weighted_bce":
        pos_weight = torch.tensor([positive_class_weight(prepared.y_train)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown MLP loss: {loss_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_score = -np.inf
    artifact_name = model_name or architecture
    best_path = model_dir / f"{artifact_name}.pt"
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}

    try:
        train_device_logged = False
        val_device_logged = False
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for X_cat, X_num, y in tqdm(train_loader, desc=f"MLP epoch {epoch + 1}/{epochs}", leave=False):
                X_cat, X_num, y = _prepare_batch(X_cat, X_num, y, device)
                if not train_device_logged:
                    verify_device(model, X_num)
                    print("Batch device:", X_num.device)
                    train_device_logged = True
                optimizer.zero_grad()
                logits = model(X_cat, X_num)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            val_losses = []
            val_probs = []
            val_targets = []
            with torch.no_grad():
                for X_cat, X_num, y in val_loader:
                    X_cat, X_num, y = _prepare_batch(X_cat, X_num, y, device)
                    if not val_device_logged:
                        verify_device(model, X_num)
                        print("Batch device:", X_num.device)
                        val_device_logged = True
                    logits = model(X_cat, X_num)
                    val_losses.append(float(criterion(logits, y).cpu()))
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
                    val_targets.append(y.cpu().numpy())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            val_pr_auc = float(average_precision_score(np.concatenate(val_targets), np.concatenate(val_probs)))
            scheduler.step(val_pr_auc)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_pr_auc"].append(val_pr_auc)
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_pr_auc={val_pr_auc:.4f}"
            )

            if val_pr_auc > best_score:
                best_score = val_pr_auc
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cat_dims": prepared.cat_dims,
                        "architecture": architecture,
                        "loss_name": loss_name,
                    },
                    best_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    except RuntimeError as exc:
        if device.type == "mps" and allow_device_fallback:
            print(f"Falling back to CPU due to MPS incompatibility: {exc}")
            return train_mlp(
                prepared,
                model_dir=model_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                device="cpu",
                architecture=architecture,
                loss_name=loss_name,
                model_name=model_name,
                allow_device_fallback=False,
            )
        raise

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.hospital_device = str(device)
    return model, history


def predict_mlp(
    model: MLPClassifier,
    X_cat: np.ndarray,
    X_num: np.ndarray,
    batch_size: int = 64,
    device: str | None = None,
    allow_device_fallback: bool = True,
) -> np.ndarray:
    """Return MLP positive-class probabilities."""
    requested_device = device
    device = get_torch_device(device)
    if str(requested_device).lower() == "mps" and getattr(model, "hospital_device", None) == "cpu":
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    dummy_y = np.zeros(len(X_cat), dtype=np.float32)
    loader = DataLoader(
        TabularDataset(X_cat, X_num, dummy_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    probabilities: list[np.ndarray] = []
    try:
        inference_device_logged = False
        with torch.no_grad():
            for batch_cat, batch_num, _ in loader:
                batch_cat, batch_num, _ = _prepare_batch(batch_cat, batch_num, None, device)
                if not inference_device_logged:
                    verify_device(model, batch_num)
                    print("Batch device:", batch_num.device)
                    inference_device_logged = True
                logits = model(batch_cat, batch_num)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
    except RuntimeError as exc:
        if device.type == "mps" and allow_device_fallback:
            print(f"Falling back to CPU due to MPS incompatibility: {exc}")
            return predict_mlp(model, X_cat, X_num, batch_size=batch_size, device="cpu", allow_device_fallback=False)
        raise
    return np.concatenate(probabilities)


def train_tabnet(
    prepared,
    model_dir: Path = MODELS_DIR,
    quick: bool = False,
    use_pretraining: bool = False,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    allow_device_fallback: bool = True,
):
    """Train pytorch-tabnet if installed."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        from pytorch_tabnet.pretraining import TabNetPretrainer
    except ImportError as exc:
        raise ImportError(
            "pytorch-tabnet is not installed. Run `pip install pytorch-tabnet` in Colab/local env."
        ) from exc

    ensure_directories()
    resolved_device = get_torch_device(device)
    print(f"TabNet device: {resolved_device}")
    X_train = np.ascontiguousarray(combined_tabnet_matrix(prepared.X_cat_train, prepared.X_num_train), dtype=np.float32)
    X_val = np.ascontiguousarray(combined_tabnet_matrix(prepared.X_cat_val, prepared.X_num_val), dtype=np.float32)

    common_params = dict(
        cat_idxs=prepared.cat_idxs,
        cat_dims=prepared.cat_dims,
        cat_emb_dim=2 if quick else 4,
        n_d=8 if quick else 24,
        n_a=8 if quick else 24,
        n_steps=3 if quick else 4,
        gamma=1.3 if quick else 1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-2 if quick else 1e-2},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=RANDOM_STATE,
        verbose=1,
        device_name=resolved_device.type,
    )

    pretrainer = None
    try:
        if use_pretraining:
            try:
                pretrainer = TabNetPretrainer(**common_params)
                pretrainer.fit(
                    X_train=X_train,
                    eval_set=[X_val],
                    max_epochs=10 if quick else 40,
                    patience=3 if quick else 8,
                    batch_size=batch_size,
                    virtual_batch_size=min(32, batch_size),
                    num_workers=0,
                    drop_last=False,
                    pretraining_ratio=0.8,
                )
                pretrainer.save_model(str(model_dir / "tabnet_pretrainer"))
            except Exception as exc:
                print(f"TabNet pretraining failed; continuing supervised-only: {exc}")
                pretrainer = None

        model = TabNetClassifier(**common_params)
        print("Model device:", resolved_device)
        sample_weights = compute_sample_weight(class_weight="balanced", y=prepared.y_train.astype(int)).astype(np.float32)
        fit_kwargs = {}
        if pretrainer is not None:
            fit_kwargs["from_unsupervised"] = pretrainer

        model.fit(
            X_train=X_train,
            y_train=prepared.y_train.astype(int),
            eval_set=[(X_val, prepared.y_val.astype(int))],
            eval_name=["valid"],
            eval_metric=["auc"],
            weights=sample_weights,
            max_epochs=5 if quick else 50,
            patience=3 if quick else 10,
            batch_size=batch_size,
            virtual_batch_size=min(32, batch_size),
            num_workers=0,
            drop_last=False,
            **fit_kwargs,
        )
    except RuntimeError as exc:
        if resolved_device.type == "mps" and allow_device_fallback:
            print(f"Falling back to CPU due to MPS incompatibility: {exc}")
            return train_tabnet(
                prepared,
                model_dir=model_dir,
                quick=quick,
                use_pretraining=use_pretraining,
                batch_size=batch_size,
                device="cpu",
                allow_device_fallback=False,
            )
        raise
    model.hospital_device = resolved_device.type
    model.save_model(str(model_dir / "tabnet"))
    return model


def predict_tabnet(model, X_cat: np.ndarray, X_num: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(combined_tabnet_matrix(X_cat, X_num), dtype=np.float32)
    return model.predict_proba(X)[:, 1]


def run_deep(
    sample_size: int | None = None,
    quick: bool = False,
    skip_tabnet: bool = False,
    device: str | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """Train/evaluate MLP and TabNet, returning metrics rows."""
    split = prepare_splits(sample_size=sample_size)
    prepared = fit_transform_deep(split)
    joblib.dump(prepared, MODELS_DIR / "deep_preprocessing.joblib")
    rows: list[dict] = []

    mlp_epochs = 3 if quick else 30
    mlp, _ = train_mlp(prepared, epochs=mlp_epochs, patience=2 if quick else 5, device=device, batch_size=batch_size)
    val_proba = predict_mlp(mlp, prepared.X_cat_val, prepared.X_num_val, batch_size=batch_size, device=device)
    threshold = choose_threshold_max_f1(prepared.y_val, val_proba)
    test_proba = predict_mlp(mlp, prepared.X_cat_test, prepared.X_num_test, batch_size=batch_size, device=device)
    rows.append(metrics_row("MLP", "test", prepared.y_test, test_proba, threshold, "val_max_f1"))

    if not skip_tabnet:
        try:
            tabnet = train_tabnet(prepared, quick=quick, batch_size=batch_size, device=device)
            val_proba = predict_tabnet(tabnet, prepared.X_cat_val, prepared.X_num_val)
            threshold = choose_threshold_max_f1(prepared.y_val, val_proba)
            test_proba = predict_tabnet(tabnet, prepared.X_cat_test, prepared.X_num_test)
            rows.append(metrics_row("TabNet", "test", prepared.y_test, test_proba, threshold, "val_max_f1"))
        except Exception as exc:
            print(f"TabNet training skipped/failed: {exc}")

    save_metrics(rows, append=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep tabular models.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-tabnet", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--deep-batch-size", type=int, default=64)
    args = parser.parse_args()
    rows = run_deep(
        sample_size=args.sample_size,
        quick=args.quick,
        skip_tabnet=args.skip_tabnet,
        device=None if args.device == "auto" else args.device,
        batch_size=args.deep_batch_size,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
