"""Optional TabPFN integration.

TabPFN is not part of the default runner because the official package is an
optional dependency and the full one-hot UCI design matrix can be too large for
the usual TabPFN use case. This wrapper makes the limitation explicit while
keeping a clear path for Colab experiments on a reduced feature set.
"""

from __future__ import annotations

import numpy as np


def train_tabpfn(X_train, y_train):
    """Train TabPFN if the optional `tabpfn` package is installed."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "TabPFN is optional and not installed. In Colab, try `pip install tabpfn`, "
            "then use a reduced feature set or sample because full one-hot UCI data can be large."
        ) from exc

    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    return model


def predict_tabpfn(model, X) -> np.ndarray:
    """Return positive-class probabilities from a fitted TabPFN model."""
    return model.predict_proba(X)[:, 1]
