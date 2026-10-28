"""FedAvg training and communication accounting for the congestion MLP."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

import torch
from torch import nn

from aiml.src.communication import communication_cost_per_round
from aiml.src.evaluation import evaluate_global
from aiml.src.models import CongestionMLP, train_local


def aggregate_state_dicts(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return the sample-weighted FedAvg state for identically shaped models."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Provide one positive sample count for each client state")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("FedAvg sample counts must all be positive")
    reference_keys = tuple(client_states[0])
    for state in client_states[1:]:
        if tuple(state) != reference_keys:
            raise ValueError("Client model state keys do not match")
        for key in reference_keys:
            if state[key].shape != client_states[0][key].shape:
                raise ValueError(f"Client model tensor shape differs for {key}")
    total = sum(sample_counts)
    result: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        reference = client_states[0][key]
        if not torch.is_floating_point(reference):
            raise TypeError(f"FedAvg expects floating model state; got {key}={reference.dtype}")
        average = torch.zeros_like(reference)
        for state, count in zip(client_states, sample_counts):
            average.add_(state[key], alpha=count / total)
        result[key] = average
    return result


def run_fedavg(
    clients: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    input_dim: int,
    seed: int,
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int | None = None,
    learning_rate: float = 0.01,
    loss_name: str = "huber",
) -> tuple[CongestionMLP, list[dict[str, float]]]:
    """Train clients from one global model and evaluate after each aggregation."""
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    active = [(name, pair) for name, pair in sorted(clients.items()) if len(pair[0]) > 0]
    if len(active) < 2:
        raise ValueError("FedAvg requires at least two non-empty clients")

    # Model initialization and local training use the run's reproducible torch RNG.
    torch.manual_seed(seed)
    global_model = CongestionMLP(input_dim, hidden_sizes=(32, 16), dropout=0.0)
    history: list[dict[str, float]] = []
    for round_number in range(1, rounds + 1):
        client_states = []
        sample_counts = []
        final_local_losses = []
        for _, (X_client, y_client) in active:
            local_model = deepcopy(global_model)
            losses = train_local(
                local_model,
                X_client,
                y_client,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                loss_name=loss_name,
            )
            count = len(X_client)
            client_states.append({key: value.detach().clone() for key, value in local_model.state_dict().items()})
            sample_counts.append(count)
            final_local_losses.append((losses[-1], count))

        global_model.load_state_dict(aggregate_state_dicts(client_states, sample_counts))
        round_metrics = evaluate_global(global_model, X_test, y_test)
        history.append({
            "round": round_number,
            "local_loss": sum(loss * count for loss, count in final_local_losses) / sum(sample_counts),
            **round_metrics,
        })
    return global_model, history


def communication_bytes_per_round(model: nn.Module, n_clients: int) -> int:
    """Raw float32 payload estimate: one full upload and one broadcast per client."""
    return communication_cost_per_round(model, n_clients)["total_bytes"]
