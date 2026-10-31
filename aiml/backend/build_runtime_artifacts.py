"""Build compact API artifacts offline from the canonical data and results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aiml.src.data_utils import DEFAULT_DATA_PATH, prepare_data
from aiml.src.models import CongestionMLP, seed_everything, train_local


ARTIFACTS = Path(__file__).with_name("artifacts")


def build() -> None:
    """Prepare the dashboard payload and seed-42 predictor for deployment."""
    data = prepare_data(DEFAULT_DATA_PATH, seed=42)
    results = DEFAULT_DATA_PATH.parents[1] / "results"

    def records(name: str) -> list[dict]:
        frame = pd.read_csv(results / name)
        output = frame.to_dict(orient="records")
        for record in output:
            for key, value in record.items():
                if value is None or pd.isna(value):
                    record[key] = None
                elif isinstance(value, np.generic):
                    record[key] = value.item()
                if isinstance(record[key], float) and not np.isfinite(record[key]):
                    record[key] = None
        return output

    summary = records("summary_metrics.csv")
    central = next((row for row in summary if row.get("scenario") == "centralized"), None)
    recent = records("seed_metrics.csv") + records("fedavg_seed_metrics.csv")
    rounds = records("fedavg_round_metrics.csv")
    fedavg_rows = records("fedavg_seed_metrics.csv")
    trend_counts = (
        data.train_rows["_time"].dt.floor("15min").value_counts().sort_index()
    )
    overview = {
        "project": "AIORI Traffic Intelligence",
        "target": "Ordinal congestion category",
        "labels": ["Very smooth", "Smooth", "Mild congestion", "Heavy congestion"],
        "locations": len(data.location_vocab),
        "train_rows": len(data.train_rows),
        "test_rows": len(data.test_rows),
        "feature_count": len(data.feature_names),
        "split": data.audit["split_kind"],
        "observed_time_range": {
            "start": data.train_rows["_time"].min().strftime("%H:%M:%S"),
            "end": data.train_rows["_time"].max().strftime("%H:%M:%S"),
        },
        "best_verified": central,
        "experiments": recent,
        "observation_trend": [
            {"time": timestamp.strftime("%H:%M"), "observations": int(count)}
            for timestamp, count in trend_counts.items()
        ],
        "federated_available": bool(rounds),
        "federated_rounds": max(
            (int(row["round"]) for row in rounds if row.get("round") is not None), default=0
        ),
        "federated_partitions": sorted(
            {row.get("partition") for row in rounds if row.get("partition")}
        ),
        "fedavg_summary": [
            row for row in summary
            if str(row.get("scenario", "")).startswith("fedavg_")
        ],
        "saved_fedavg_runs": sum(
            str(row.get("scenario", "")).startswith("fedavg_") for row in fedavg_rows
        ),
        "communication_measured": False,
    }
    location_vocab = {str(location): index for location, index in data.location_vocab.items()}
    runtime_metadata = {
        "overview": overview,
        "location_ids": sorted(location_vocab),
        "location_vocab": location_vocab,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "overview.json").write_text(
        json.dumps(runtime_metadata, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    seed_everything(42)
    model = CongestionMLP(data.X_train.shape[1])
    train_local(model, data.X_train, data.y_train, epochs=60, learning_rate=0.01, loss_name="huber")
    model.eval()
    torch.save(model.state_dict(), ARTIFACTS / "predictor.pt")


if __name__ == "__main__":
    build()
