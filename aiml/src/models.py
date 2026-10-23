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


def train_local(model: CongestionMLP, X: torch.Tensor, y: torch.Tensor, *,
                epochs: int = 10, learning_rate: float = 1e-3, loss_name: str = "huber") -> list[float]:
    """Train a local client model and return epoch mean losses."""
    if loss_name not in {"huber", "mse"}:
        raise ValueError("loss_name must be 'huber' or 'mse'")
    criterion = nn.HuberLoss() if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def predict(model: CongestionMLP, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(X)
