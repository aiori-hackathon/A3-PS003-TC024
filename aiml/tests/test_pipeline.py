"""Unit tests use tiny explicitly labelled fixtures; real data is integration-only."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aiml.src.data_utils import LABEL_MAP, load_clean_data, partition_clients, prepare_data
from aiml.src.experiment import metrics
from aiml.src.federated import aggregate_state_dicts
from aiml.src.models import CongestionMLP, predict, train_local


def test_timestamp_parsing_missing_handling_and_repeated_pairs(tmp_path: Path):
    fixture = pd.DataFrame({
        "stop_id_from": [1, 1, 2, 3], "arrival_time": ["08:00:00", "08:00:00", "bad", "09:00:00"],
        "speed": [10.0, 12.0, 4.0, np.inf], "Degree_of_congestion": ["Smooth", "Smooth", "Very smooth", "Heavy congestion"]
    })
    path = tmp_path / "fixture.csv"
    fixture.to_csv(path, index=False)
    clean, audit = load_clean_data(path)
    assert len(clean) == 2
    assert audit["unparseable_arrival_time_dropped"] == 1
    assert audit["missing_or_nonfinite_speed_dropped"] == 1
    assert audit["repeated_origin_time_pairs_retained"] == 1


def test_invalid_schema_rejected(tmp_path: Path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_clean_data(p)


def test_empty_input_rejected(tmp_path: Path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["stop_id_from", "arrival_time", "speed", "Degree_of_congestion"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_clean_data(p)


def test_all_invalid_rows_rejected(tmp_path: Path):
    p = tmp_path / "invalid.csv"
    pd.DataFrame({"stop_id_from": [1], "arrival_time": ["bad"], "speed": [np.inf],
                  "Degree_of_congestion": ["unknown"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="No usable rows"):
        load_clean_data(p)


def test_feature_pipeline_prevents_target_and_speed_leakage():
    prepared = prepare_data()
    assert "Degree_of_congestion" not in prepared.feature_names
    assert "_target" not in prepared.feature_names
    assert "speed" not in prepared.feature_names and "_speed" not in prepared.feature_names
    assert set(prepared.location_vocab) == set(prepared.train_rows["stop_id_from"])
    assert set(prepared.location_vocab).isdisjoint(
        set(prepared.test_rows["stop_id_from"]) - set(prepared.train_rows["stop_id_from"])
    )
    assert torch.allclose((prepared.X_train[:, 0] ** 2 + prepared.X_train[:, 1] ** 2),
                          torch.ones(len(prepared.X_train)), atol=1e-5)


def test_model_shapes_training_and_state_dict_compatibility():
    torch.manual_seed(7)
    X = torch.randn(32, 5)
    y = X[:, :1] * 0.4
    model = CongestionMLP(5, (8,))
    before = float(torch.nn.functional.mse_loss(model(X), y).detach())
    train_local(model, X, y, epochs=30, learning_rate=0.02, loss_name="mse")
    after = float(torch.nn.functional.mse_loss(model(X), y).detach())
    assert after < before
    assert predict(model, X).shape == (32, 1)
    assert all(torch.isfinite(p).all() for p in model.state_dict().values())
    other = CongestionMLP(5, (8,))
    other.load_state_dict(model.state_dict())
    assert list(other.state_dict()) == list(model.state_dict())


def test_fedavg_state_dict_average_loads_and_predicts():
    """FedAvg aggregation produces a compatible state that can predict."""
    torch.manual_seed(12)
    client_a = CongestionMLP(4, (6,))
    torch.manual_seed(34)
    client_b = CongestionMLP(4, (6,))
    states = [client_a.state_dict(), client_b.state_dict()]
    averaged = aggregate_state_dicts(states, sample_counts=[1, 1])
    global_model = CongestionMLP(4, (6,))
    global_model.load_state_dict(averaged)
    result = predict(global_model, torch.ones(3, 4))
    assert result.shape == (3, 1)
    assert torch.isfinite(result).all()
    assert set(global_model.state_dict()) == set(averaged)


def test_partition_determinism_coverage_and_no_overlap():
    X = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    y = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    rows = pd.DataFrame({"stop_id_from": [1, 1, 2, 2, 3, 3] * 2})
    for mode in ("iid", "location"):
        clients = partition_clients(X, y, rows, n_clients=3, seed=19, mode=mode)
        clients2 = partition_clients(X, y, rows, n_clients=3, seed=19, mode=mode)
        if mode == "iid":
            iid_clients = clients
        merged = torch.cat([v[0] for v in clients.values()])
        assert len(merged) == len(X)
        assert sorted(merged[:, 0].tolist()) == sorted(X[:, 0].tolist())
        assert all(torch.equal(clients[k][0], clients2[k][0]) for k in clients)
    iid_other_seed = partition_clients(X, y, rows, n_clients=3, seed=20, mode="iid")
    assert any(not torch.equal(iid_clients[k][0], iid_other_seed[k][0]) for k in iid_clients)
    assert max(len(v[0]) for v in iid_clients.values()) - min(len(v[0]) for v in iid_clients.values()) <= 1
    grouped = partition_clients(X, y, rows, n_clients=3, mode="location")
    client_locations = [set(rows.iloc[torch.isin(X[:, 0], xx[:, 0]).numpy()].stop_id_from) for xx, _ in grouped.values()]
    assert all(not (client_locations[i] & client_locations[j]) for i in range(3) for j in range(i))
    with pytest.raises(ValueError, match="between 3 and 5"):
        partition_clients(X, y, rows, n_clients=2)


def test_reference_baseline_metrics_are_finite():
    """Tiny labelled fixture checks metric execution; not an experiment result."""
    result = metrics(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.2, 0.9, 1.8]))
    assert set(result) == {"mae", "rmse", "accuracy_rounded", "macro_f1_rounded"}
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.integration
def test_real_dataset_chronological_time_of_day_holdout():
    path = Path(__file__).resolve().parents[1] / "dataset" / "pune_traffic.csv"
    if not path.exists():
        pytest.skip("user-provided dataset is absent")
    prepared = prepare_data(path)
    train_groups = set(prepared.train_rows["_group"])
    test_groups = set(prepared.test_rows["_group"])
    assert train_groups.isdisjoint(test_groups)
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1] == len(prepared.feature_names)
    assert torch.isfinite(prepared.X_train).all() and torch.isfinite(prepared.y_test).all()
    split_counts = prepared.stop_split_counts
    splittable = split_counts[split_counts["split_method"] == "per_stop"]
    assert len(splittable) == prepared.audit["stops_split_per_stop"]
    assert ((splittable.train_rows > 0) & (splittable.test_rows > 0)).all()
    train_end = prepared.train_rows.groupby("stop_id_from")["_time"].max()
    test_start = prepared.test_rows.groupby("stop_id_from")["_time"].min()
    assert (train_end.loc[splittable.stop_id_from].to_numpy() <
            test_start.loc[splittable.stop_id_from].to_numpy()).all()
    assert prepared.audit["stops_appearing_in_both"] == len(splittable)
    assert prepared.audit["train_hour_range"] and prepared.audit["test_hour_range"]
    assert "speed" not in prepared.feature_names and "_speed" not in prepared.feature_names
    assert prepared.audit["row_order_stops_nondecreasing"] < prepared.audit["row_order_stops_checked"]
    assert prepared.audit["split_kind"].startswith("chronological time-of-day")
    assert prepared.audit["speed_target_spearman"] < 0
