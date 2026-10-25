"""Focused unit tests for Chanchal's federated engine and accounting."""
import pytest
import torch
from torch import nn

from src.communication import communication_cost_per_round, communication_cost_total
from src.evaluation import evaluate_global, evaluate_per_client
from src.federated import aggregate_state_dicts, run_fedavg
from src.models import CongestionMLP, train_local


def test_fedavg_aggregation_is_weighted_by_client_samples():
    states = [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([4.0, 8.0])},
    ]
    combined = aggregate_state_dicts(states, [1, 3])
    assert torch.allclose(combined["weight"], torch.tensor([3.25, 6.5]))


def test_round_runner_trains_clients_and_returns_global_metrics():
    torch.manual_seed(10)
    clients = {
        "client_1": (torch.randn(8, 3), torch.randn(8, 1)),
        "client_2": (torch.randn(12, 3), torch.randn(12, 1)),
    }
    X_test, y_test = torch.randn(6, 3), torch.randn(6, 1)
    model, history = run_fedavg(
        clients, X_test, y_test, input_dim=3, seed=5,
        rounds=2, local_epochs=1, batch_size=4,
    )
    assert isinstance(model, CongestionMLP)
    assert [entry["round"] for entry in history] == [1, 2]
    for entry in history:
        assert {"mae", "rmse", "accuracy_rounded", "macro_f1_rounded"} <= set(entry)
        assert all(torch.isfinite(torch.tensor(entry[name])) for name in
                   ("mae", "rmse", "accuracy_rounded", "macro_f1_rounded"))


def test_configurable_mini_batch_training_rejects_invalid_batch_size():
    model = CongestionMLP(2, hidden_sizes=(4,))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        train_local(model, torch.ones(4, 2), torch.ones(4, 1), batch_size=0)


def test_communication_reports_uplink_downlink_and_total():
    model = nn.Linear(2, 1)  # 3 parameters, float32 -> 12 bytes per model.
    one_round = communication_cost_per_round(model, n_clients=4)
    assert one_round == {
        "model_bytes": 12,
        "client_to_server_bytes": 48,
        "server_to_client_bytes": 48,
        "total_bytes": 96,
    }
    total = communication_cost_total(model, n_clients=4, rounds=3)
    assert total["client_to_server_bytes"] == 144
    assert total["server_to_client_bytes"] == 144
    assert total["total_bytes"] == 288


def test_global_and_per_client_evaluation_use_held_out_data():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    test_sets = {
        "client_1": (torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]])),
        "client_2": (torch.tensor([[3.0]]), torch.tensor([[4.0]])),
    }
    global_metrics = evaluate_global(model, torch.tensor([[1.0], [2.0]]),
                                     torch.tensor([[1.0], [2.0]]))
    local_metrics = evaluate_per_client(model, test_sets)
    assert global_metrics["mae"] == 0.0
    assert local_metrics["client_1"]["metrics"]["mae"] == 0.0
    assert local_metrics["client_2"]["sample_count"] == 1
    assert local_metrics["client_2"]["metrics"]["mae"] == 1.0
