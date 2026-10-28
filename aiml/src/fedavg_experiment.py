"""Run three-seed FedAvg comparisons on the shared congestion data split.

Run from the project root with ``python -m src.fedavg_experiment``.
This reuses the project's data preparation, four-client partitions, MLP,
ordinal metrics, and training-mean/location-mean reference baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aiml.src.data_utils import DEFAULT_DATA_PATH, partition_clients, prepare_data
from aiml.src.communication import communication_cost_per_round, communication_cost_total
from aiml.src.evaluation import evaluate_per_client, metrics
from aiml.src.federated import run_fedavg
from aiml.src.models import predict, seed_everything


SEEDS = (42, 123, 456)
N_CLIENTS = 4


def _reference_rows(data, seed: int) -> list[dict]:
    truth = data.y_test.numpy().reshape(-1)
    train_mean = float(data.y_train.mean())
    train_mean_prediction = np.full(len(truth), train_mean)
    location_means = data.train_rows.groupby("stop_id_from")["_target"].mean()
    location_prediction = (
        data.test_rows["stop_id_from"].map(location_means)
        .fillna(train_mean)
        .to_numpy()
    )
    return [
        {"seed": seed, "scenario": "train_mean_baseline", **metrics(truth, train_mean_prediction)},
        {"seed": seed, "scenario": "location_mean_baseline", **metrics(truth, location_prediction)},
    ]


def _client_distribution(seed: int, mode: str, clients: dict) -> list[dict]:
    label_names = ("very_smooth", "smooth", "mild_congestion", "heavy_congestion")
    rows = []
    for client_id, (_, y_client) in clients.items():
        labels = y_client.reshape(-1).to(torch.int64)
        record = {
            "seed": seed,
            "partition": mode,
            "client_id": client_id,
            "sample_count": len(labels),
        }
        for class_id, label_name in enumerate(label_names):
            record[label_name] = int((labels == class_id).sum())
        rows.append(record)
    return rows


def run(seed: int, *, rounds: int, local_epochs: int,
        batch_size: int | None = None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    data = prepare_data(DEFAULT_DATA_PATH, seed=seed, test_fraction=0.2)
    seed_everything(seed)
    seed_rows = _reference_rows(data, seed)
    round_rows: list[dict] = []
    distribution_rows: list[dict] = []
    client_metric_rows: list[dict] = []
    truth = data.y_test.numpy().reshape(-1)

    partitions = {
        "iid": partition_clients(
            data.X_train, data.y_train, data.train_rows,
            n_clients=N_CLIENTS, seed=seed, mode="iid",
        ),
        "location": partition_clients(
            data.X_train, data.y_train, data.train_rows,
            n_clients=N_CLIENTS, seed=seed, mode="location",
        ),
    }
    for mode, clients in partitions.items():
        if sum(len(X_client) for X_client, _ in clients.values()) != len(data.X_train):
            raise RuntimeError(f"{mode} partition does not contain every training row exactly once")
        distribution_rows.extend(_client_distribution(seed, mode, clients))
        seed_everything(seed)
        global_model, history = run_fedavg(
            clients,
            data.X_test,
            data.y_test,
            input_dim=data.X_train.shape[1],
            seed=seed,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=0.01,
            loss_name="huber",
        )
        # Use the same sklearn metric implementation as centralized/local baselines.
        predictions = predict(global_model, data.X_test).numpy().reshape(-1)
        final_metrics = metrics(truth, predictions)
        per_round_comm = communication_cost_per_round(global_model, len(clients))
        total_comm = communication_cost_total(global_model, len(clients), rounds)
        client_test_sets = partition_clients(
            data.X_test, data.y_test, data.test_rows,
            n_clients=N_CLIENTS, seed=seed, mode=mode,
        )
        for client_id, evaluated in evaluate_per_client(global_model, client_test_sets).items():
            client_metric_rows.append({
                "seed": seed,
                "partition": mode,
                "client_id": client_id,
                "sample_count": evaluated["sample_count"],
                **(evaluated["metrics"] or {}),
            })
        seed_rows.append({
            "seed": seed,
            "scenario": f"fedavg_{mode}",
            **final_metrics,
            "n_clients": len(clients),
            "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "model_bytes": per_round_comm["model_bytes"],
            "client_to_server_bytes_per_round": per_round_comm["client_to_server_bytes"],
            "server_to_client_bytes_per_round": per_round_comm["server_to_client_bytes"],
            "communication_bytes_per_round": per_round_comm["total_bytes"],
            "client_to_server_bytes_total": total_comm["client_to_server_bytes"],
            "server_to_client_bytes_total": total_comm["server_to_client_bytes"],
            "communication_bytes_total": total_comm["total_bytes"],
            "final_local_loss": history[-1]["local_loss"],
        })
        for item in history:
            round_rows.append({
                "seed": seed,
                "partition": mode,
                **item,
                "model_bytes": per_round_comm["model_bytes"],
                "client_to_server_bytes_per_round": per_round_comm["client_to_server_bytes"],
                "server_to_client_bytes_per_round": per_round_comm["server_to_client_bytes"],
                "communication_bytes_per_round": per_round_comm["total_bytes"],
                "client_to_server_bytes_cumulative": total_comm["client_to_server_bytes"] // rounds * int(item["round"]),
                "server_to_client_bytes_cumulative": total_comm["server_to_client_bytes"] // rounds * int(item["round"]),
                "communication_bytes_cumulative": total_comm["total_bytes"] // rounds * int(item["round"]),
            })
    return seed_rows, round_rows, distribution_rows, client_metric_rows


def _summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    excluded = {"seed", "scenario"}
    for scenario, group in seed_metrics.groupby("scenario", sort=True):
        row = {"scenario": scenario}
        for column in seed_metrics.columns:
            if column in excluded:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                row[f"{column}_mean"] = float(np.mean(values))
                row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="positive local mini-batch size; default None keeps full-batch training",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("rounds/local-epochs must be positive and batch-size must be positive when set")

    seed_rows: list[dict] = []
    round_rows: list[dict] = []
    distribution_rows: list[dict] = []
    client_metric_rows: list[dict] = []
    for seed in SEEDS:
        print(f"FedAvg seed {seed}...", flush=True)
        one_seed, one_rounds, one_distributions, one_client_metrics = run(
            seed, rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size,
        )
        seed_rows.extend(one_seed)
        round_rows.extend(one_rounds)
        distribution_rows.extend(one_distributions)
        client_metric_rows.extend(one_client_metrics)

    output = DEFAULT_DATA_PATH.parents[1] / "results"
    output.mkdir(exist_ok=True)
    current_seed = pd.DataFrame(seed_rows)
    current_seed.to_csv(output / "fedavg_seed_metrics.csv", index=False)
    round_frame = pd.DataFrame(round_rows)
    distribution_frame = pd.DataFrame(distribution_rows)
    client_metrics_frame = pd.DataFrame(client_metric_rows)
    round_frame.to_csv(output / "fedavg_round_metrics.csv", index=False)
    distribution_frame.to_csv(output / "fedavg_client_distributions.csv", index=False)
    client_metrics_frame.to_csv(output / "fedavg_client_metrics.csv", index=False)

    # Merge into the project's one summary table while retaining centralized and
    # local-only rows previously generated by ``python -m src.experiment``.
    summary_path = output / "summary_metrics.csv"
    current_summary = _summary(current_seed)
    replace = {"train_mean_baseline", "location_mean_baseline", "fedavg_iid", "fedavg_location"}
    if summary_path.is_file():
        prior = pd.read_csv(summary_path)
        prior = prior.loc[~prior["scenario"].isin(replace)]
        combined_summary = pd.concat([prior, current_summary], ignore_index=True, sort=False)
    else:
        combined_summary = current_summary
    combined_summary.to_csv(summary_path, index=False)

    # Keep the integrated root results used by the backend while also exposing
    # separate IID and Non-IID experiment folders for review and presentation.
    experiments_dir = output.parent / "experiments"
    for mode, folder in (("iid", "iid"), ("location", "non_iid")):
        mode_dir = experiments_dir / folder
        mode_dir.mkdir(parents=True, exist_ok=True)
        current_seed.loc[current_seed["scenario"] == f"fedavg_{mode}"].to_csv(
            mode_dir / "seed_metrics.csv", index=False
        )
        round_frame.loc[round_frame["partition"] == mode].to_csv(
            mode_dir / "round_metrics.csv", index=False
        )
        client_metrics_frame.loc[client_metrics_frame["partition"] == mode].to_csv(
            mode_dir / "client_metrics.csv", index=False
        )
        distribution_frame.loc[distribution_frame["partition"] == mode].to_csv(
            mode_dir / "client_distributions.csv", index=False
        )
    experiments_results = experiments_dir / "results"
    experiments_results.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(experiments_results / "summary_metrics.csv", index=False)

    print("\nFedAvg final results by seed and partition:")
    print(current_seed.loc[current_seed["scenario"].str.startswith("fedavg_")].to_string(index=False))
    print(f"\nCombined baseline and FedAvg summary: {summary_path}")
    print(f"Per-round convergence: {output / 'fedavg_round_metrics.csv'}")


if __name__ == "__main__":
    main()
