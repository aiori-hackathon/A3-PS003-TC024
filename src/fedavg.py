"""Backward-compatible imports for the FedAvg engine in :mod:`src.federated`."""
from src.communication import communication_cost_per_round, model_size_bytes
from src.federated import aggregate_state_dicts, run_fedavg


def communication_bytes_per_round(model, n_clients: int) -> int:
    """Backward-compatible total-byte helper."""
    return communication_cost_per_round(model, n_clients)["total_bytes"]


__all__ = [
    "aggregate_state_dicts",
    "run_fedavg",
    "communication_bytes_per_round",
    "communication_cost_per_round",
    "model_size_bytes",
]
