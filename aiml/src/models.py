"""Small FedAvg-compatible MLP for ordinal congestion regression."""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import nn


class CongestionMLP(nn.Module):
    """Tabular MLP with identical, stateless parameter shapes on every client."""
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int] = (32, 16), dropout: float = 0.0):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        dims = [input_dim, *list(hidden_sizes), 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend((nn.Linear(dims[i], dims[i + 1]), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small tabular MLPs often run faster on Windows CPU with one BLAS/OpenMP
    # worker; large default thread pools add overhead to each full-batch step.
    torch.set_num_threads(1)


def train_local(model: CongestionMLP, X: torch.Tensor, y: torch.Tensor, *,
                epochs: int = 10, learning_rate: float = 1e-3, loss_name: str = "huber",
                batch_size: int | None = None) -> list[float]:
    """Train a local client model and return sample-weighted epoch losses.

    ``batch_size=None`` preserves full-batch training. A positive batch size
    enables shuffled mini-batch updates for larger client datasets.
    """
    if loss_name not in {"huber", "mse"}:
        raise ValueError("loss_name must be 'huber' or 'mse'")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must have equal, non-zero lengths")
    criterion = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(epochs):
        if batch_size is None or batch_size >= len(X):
            batches = (torch.arange(len(X), device=X.device),)
        else:
            order = torch.randperm(len(X), device=X.device)
            batches = order.split(batch_size)
        epoch_loss = 0.0
        for indices in batches:
            X_batch, y_batch = X[indices], y[indices]
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices) / len(X)
        losses.append(epoch_loss)
    return losses


@torch.no_grad()
def predict(model: CongestionMLP, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(X)
