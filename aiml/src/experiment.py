"""Reproducible real-dataset baseline and partition audit.

Run from the repository root with ``python3 -m aiml.src.experiment``. No
network or data other than aiml/dataset/pune_traffic.csv is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from aiml.src.data_utils import DEFAULT_DATA_PATH, partition_clients, prepare_data
from aiml.src.models import CongestionMLP, predict, seed_everything, train_local


def metrics(y_true: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    truth = np.rint(y_true).astype(int).reshape(-1)
    prediction = np.clip(np.rint(raw), 0, 3).astype(int).reshape(-1)
    return {"mae": float(mean_absolute_error(y_true, raw)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, raw))),
            "accuracy_rounded": float(accuracy_score(truth, prediction)),
            "macro_f1_rounded": float(f1_score(truth, prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0))}


def client_distribution(clients: dict) -> dict:
    labels = ("Very smooth", "Smooth", "Mild congestion", "Heavy congestion")
    output = {}
    for client_id, (_, y) in clients.items():
        counts = {label: int((y.reshape(-1) == class_id).sum())
                  for class_id, label in enumerate(labels)}
        output[client_id] = {"sample_count": len(y), "class_counts": counts}
    return output


def run(seed: int, epochs: int = 60) -> dict:
    data = prepare_data(DEFAULT_DATA_PATH, seed=seed)
    seed_everything(seed)
    model = CongestionMLP(data.X_train.shape[1])
    losses = train_local(model, data.X_train, data.y_train, epochs=epochs, learning_rate=0.01, loss_name="huber")
    prediction = predict(model, data.X_test).numpy().reshape(-1)
    truth = data.y_test.numpy().reshape(-1)
    train_mean = float(data.y_train.mean())
    mean_prediction = np.full(len(truth), train_mean)
    # Location mean baseline uses training labels only; unseen test locations use global training mean.
    means = data.train_rows.groupby("stop_id_from")["_target"].mean()
    by_location = data.test_rows["stop_id_from"].map(means).fillna(train_mean).to_numpy()
    iid = partition_clients(data.X_train, data.y_train, data.train_rows, seed=seed, mode="iid")
    noniid = partition_clients(data.X_train, data.y_train, data.train_rows, seed=seed, mode="location")
    assert sum(len(pair[0]) for pair in iid.values()) == len(data.X_train)
    assert sum(len(pair[0]) for pair in noniid.values()) == len(data.X_train)
    parameter_count = sum(p.numel() for p in model.parameters())
    local_client_metrics = {}
    for partition_name, clients in (("iid", iid), ("location", noniid)):
        per_client = {}
        for index, (client_id, (client_X, client_y)) in enumerate(clients.items()):
            if len(client_X) == 0:
                per_client[client_id] = {"sample_count": 0, "metrics": None}
                continue
            seed_everything(seed + 1000 + index)
            local_model = CongestionMLP(data.X_train.shape[1])
            train_local(local_model, client_X, client_y, epochs=epochs,
                        learning_rate=0.01, loss_name="huber")
            local_raw = predict(local_model, data.X_test).numpy().reshape(-1)
            per_client[client_id] = {"sample_count": len(client_X),
                                     "metrics": metrics(truth, local_raw)}
        local_client_metrics[partition_name] = per_client
    local_average = {}
    for partition_name, per_client in local_client_metrics.items():
        eligible = [entry["metrics"] for entry in per_client.values() if entry["metrics"] is not None]
        local_average[partition_name] = {
            name: float(np.mean([entry[name] for entry in eligible]))
            for name in eligible[0]
        }
    return {"seed": seed, "audit": data.audit, "centralized": metrics(truth, prediction),
            "train_mean_baseline": metrics(truth, mean_prediction),
            "location_mean_baseline": metrics(truth, by_location),
            "local_client_metrics": local_client_metrics,
            "local_client_average": local_average,
            "client_distributions": {"iid": client_distribution(iid),
                                     "location": client_distribution(noniid)},
            "stop_split_counts": data.stop_split_counts,
            "final_train_huber": losses[-1],
            "iid_client_counts": {k: len(v[0]) for k, v in iid.items()},
            "location_client_counts": {k: len(v[0]) for k, v in noniid.items()},
            "client_parameter_counts": {k: parameter_count for k in iid},
            "feature_count": data.X_train.shape[1],
            "parameter_count": parameter_count}


if __name__ == "__main__":
    results = [run(seed) for seed in (42, 123, 456)]
    rows = []
    local_rows = []
    distribution_rows = []
    for result in results:
        seed = result["seed"]
        for name in ("centralized", "train_mean_baseline", "location_mean_baseline"):
            rows.append({"seed": seed, "scenario": name, **result[name]})
        for partition, clients in result["local_client_metrics"].items():
            rows.append({"seed": seed, "scenario": f"local_{partition}_client_average",
                         **result["local_client_average"][partition]})
            for client_id, value in clients.items():
                local_rows.append({"seed": seed, "partition": partition, "client_id": client_id,
                                   "sample_count": value["sample_count"],
                                   **(value["metrics"] or {})})
        for partition, client_values in result["client_distributions"].items():
            for client_id, value in client_values.items():
                distribution_rows.append({"seed": seed, "partition": partition,
                                          "client_id": client_id,
                                          "sample_count": value["sample_count"],
                                          **value["class_counts"]})
    output = DEFAULT_DATA_PATH.parents[1] / "results"
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "seed_metrics.csv", index=False)
    results[0]["stop_split_counts"].to_csv(output / "per_stop_split_counts.csv", index=False)
    distribution_frame = pd.DataFrame(distribution_rows)
    distribution_frame.to_csv(output / "client_distributions.csv", index=False)
    local_frame = pd.DataFrame(local_rows)
    local_frame.to_csv(output / "local_client_metrics.csv", index=False)
    local_summary = local_frame.groupby(["partition", "client_id"]).agg(
        sample_count_mean=("sample_count", "mean"), sample_count_std=("sample_count", "std"),
        **{f"{metric}_{stat}": (metric, stat)
           for metric in ("mae", "rmse", "accuracy_rounded", "macro_f1_rounded")
           for stat in ("mean", "std")}
    ).reset_index()
    local_summary.to_csv(output / "local_client_summary.csv", index=False)
    summary_rows = []
    for scenario in sorted({row["scenario"] for row in rows}):
        values = [row for row in rows if row["scenario"] == scenario]
        summary = {"scenario": scenario}
        for metric in values[0]:
            if metric in {"seed", "scenario"}:
                continue
            arr = np.asarray([row[metric] for row in values], dtype=float)
            summary[f"{metric}_mean"] = float(arr.mean())
            summary[f"{metric}_std"] = float(arr.std(ddof=1))
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(output / "summary_metrics.csv", index=False)
    print("Per-seed centralized and baseline results:")
    print(pd.DataFrame(rows).to_string(index=False))
    split_audit = results[0]["audit"]
    print("\nPer-stop split evidence:")
    print({key: split_audit[key] for key in (
        "split_method", "global_fallback_cutoff", "train_hour_range", "test_hour_range",
        "train_rows", "test_rows", "stops_total", "stops_split_per_stop",
        "stops_single_time_fallback", "stops_appearing_in_both")})
    print("Per-stop train/test row counts written to aiml/results/per_stop_split_counts.csv")
    print("\nPer-client training sample counts and class distributions by seed:")
    print(distribution_frame.to_string(index=False))
    print("\nMean and sample standard deviation across seeds:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nPer-client results written to aiml/results/local_client_metrics.csv")
    print("Per-client mean/std across seeds written to aiml/results/local_client_summary.csv")
    print("Summary written to aiml/results/summary_metrics.csv")
