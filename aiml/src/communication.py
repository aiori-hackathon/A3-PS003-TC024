"""Estimated model-payload communication for synchronous FedAvg."""
from __future__ import annotations

from torch import nn


def model_size_bytes(model: nn.Module) -> int:
    """Bytes in one full parameter upload; assumes the model's tensor dtypes."""
    return sum(parameter.numel() * parameter.element_size()
               for parameter in model.parameters() if parameter.requires_grad)


def communication_cost_per_round(model: nn.Module, n_clients: int) -> dict[str, int]:
    """Estimate uplink, downlink, and total bytes for one FedAvg round.

    Assumes every client uploads one full model and receives one full global
    model. Transport headers, retries, compression, and secure aggregation are
    excluded. Broadcast is counted once per receiving client.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be positive")
    one_model = model_size_bytes(model)
    uplink = one_model * n_clients
    downlink = one_model * n_clients
    return {
        "model_bytes": one_model,
        "client_to_server_bytes": uplink,
        "server_to_client_bytes": downlink,
        "total_bytes": uplink + downlink,
    }


def communication_cost_total(model: nn.Module, n_clients: int,
                             rounds: int) -> dict[str, int]:
    """Estimate directional and total payload bytes over multiple rounds."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    per_round = communication_cost_per_round(model, n_clients)
    return {key: value * rounds for key, value in per_round.items()}
