"""Shared global, per-client, and round-wise congestion evaluation helpers."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import nn


def _numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().reshape(-1)
    return np.asarray(values).reshape(-1)


def evaluate_predictions(y_true: np.ndarray | torch.Tensor,
                         predictions: np.ndarray | torch.Tensor) -> dict[str, float]:
    """Compute regression and rounded ordinal metrics for one evaluation set."""
    truth = _numpy(y_true)
    raw = _numpy(predictions)
    if len(truth) == 0 or len(truth) != len(raw):
        raise ValueError("truth and predictions must have the same non-zero length")
    if not np.isfinite(truth).all() or not np.isfinite(raw).all():
        raise ValueError("truth and predictions must contain only finite values")
    truth_rounded = np.clip(np.rint(truth), 0, 3).astype(int)
    predicted_rounded = np.clip(np.rint(raw), 0, 3).astype(int)
    return {
        "mae": float(mean_absolute_error(truth, raw)),
        "rmse": float(np.sqrt(mean_squared_error(truth, raw))),
        "accuracy_rounded": float(accuracy_score(truth_rounded, predicted_rounded)),
        "macro_f1_rounded": float(f1_score(
            truth_rounded, predicted_rounded, labels=[0, 1, 2, 3],
            average="macro", zero_division=0,
        )),
    }


def evaluate_global(model: nn.Module, X: torch.Tensor,
                    y: torch.Tensor) -> dict[str, float]:
    """Evaluate a model on the shared global test set."""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    return evaluate_predictions(y, predictions)


def evaluate_per_client(
    model: nn.Module,
    client_test_data: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict]:
    """Evaluate a global model on each client's held-out local test data.

    The caller supplies client test splits separately; training data should not
    be passed here as a substitute for a held-out client test set.
    """
    output = {}
    for client_id, (X_client, y_client) in sorted(client_test_data.items()):
        if len(X_client) == 0:
            output[client_id] = {"sample_count": 0, "metrics": None}
            continue
        output[client_id] = {
            "sample_count": len(X_client),
            "metrics": evaluate_global(model, X_client, y_client),
        }
    return output


# Short alias used by the existing experiment scripts and tests.
metrics = evaluate_predictions
