"""FastAPI adapter for the existing Pune traffic ML pipeline."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aiml.src.data_utils import DEFAULT_DATA_PATH, prepare_data
from aiml.src.models import CongestionMLP, predict, seed_everything, train_local

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LABELS = ["Very smooth", "Smooth", "Mild congestion", "Heavy congestion"]

app = FastAPI(title="AIORI Traffic Intelligence API", version="1.0.0")
configured_origins = os.getenv(
    "AIORI_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
origins = [x.strip() for x in configured_origins if x.strip()]
if "https://synroute.vercel.app" not in origins:
    origins.append("https://synroute.vercel.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=(
        r"^https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.vercel\.app$|"
        r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(:\d+)?$"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class PredictionRequest(BaseModel):
    location: str
    time: str = Field(pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    model: Literal["centralized"] = "centralized"


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise HTTPException(404, f"Result artifact is unavailable: {name}")
    return pd.read_csv(path)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if value is None or pd.isna(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            if isinstance(record[key], float) and not np.isfinite(record[key]):
                record[key] = None
    return records


def _records(name: str) -> list[dict]:
    """Read a saved CSV as JSON-safe records, treating absent files as empty."""
    path = RESULTS / name
    if not path.is_file():
        return []
    return _json_records(pd.read_csv(path))


@lru_cache(maxsize=1)
def _data():
    if not DEFAULT_DATA_PATH.exists():
        raise HTTPException(503, "Traffic dataset is unavailable")
    return prepare_data(DEFAULT_DATA_PATH, seed=42)


@lru_cache(maxsize=1)
def _trained_predictor():
    data = _data()
    seed_everything(42)
    model = CongestionMLP(data.X_train.shape[1])
    train_local(model, data.X_train, data.y_train, epochs=60, learning_rate=0.01, loss_name="huber")
    model.eval()
    return model


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_available": DEFAULT_DATA_PATH.exists(),
            "results_available": (RESULTS / "summary_metrics.csv").exists()}


@app.get("/api/overview")
def overview():
    data = _data()
    summary = _read_csv("summary_metrics.csv")
    central = summary.loc[summary.scenario == "centralized"]
    best = _json_records(central)[0] if len(central) else None
    recent = _records("seed_metrics.csv") + _records("fedavg_seed_metrics.csv")
    rounds = _records("fedavg_round_metrics.csv")
    fedavg_rows = _records("fedavg_seed_metrics.csv")
    buckets = data.train_rows["_time"].dt.floor("15min")
    counts = buckets.value_counts().sort_index()
    trend = [{"time": t.strftime("%H:%M"), "observations": int(v)} for t, v in counts.items()]
    observed_range = {
        "start": data.train_rows["_time"].min().strftime("%H:%M:%S"),
        "end": data.train_rows["_time"].max().strftime("%H:%M:%S"),
    }
    return {"project": "AIORI Traffic Intelligence", "target": "Ordinal congestion category",
            "labels": LABELS, "locations": len(data.location_vocab), "train_rows": len(data.train_rows),
            "test_rows": len(data.test_rows), "feature_count": len(data.feature_names),
            "split": data.audit["split_kind"], "observed_time_range": observed_range, "best_verified": best,
            "experiments": recent, "observation_trend": trend,
            "federated_available": bool(rounds),
            "federated_rounds": max((int(row["round"]) for row in rounds if row.get("round") is not None), default=0),
            "federated_partitions": sorted({row.get("partition") for row in rounds if row.get("partition")}),
            "fedavg_summary": [row for row in _json_records(summary) if str(row.get("scenario", "")).startswith("fedavg_")],
            "saved_fedavg_runs": sum(str(row.get("scenario", "")).startswith("fedavg_") for row in fedavg_rows),
            "communication_measured": False}


@app.get("/api/locations")
def locations():
    data = _data()
    return {"locations": sorted(str(x) for x in data.train_rows["stop_id_from"].unique())}


@app.get("/api/experiments")
def experiments():
    return {"experiments": _records("seed_metrics.csv") + _records("fedavg_seed_metrics.csv"),
            "summaries": _records("summary_metrics.csv")}


@app.get("/api/experiments/{experiment_id}")
def experiment_detail(experiment_id: str):
    parts = experiment_id.split(":")
    if len(parts) != 2 or not parts[0].isdigit():
        raise HTTPException(400, "Use experiment IDs in the format seed:scenario")
    frames = [_read_csv(name) for name in ("seed_metrics.csv", "fedavg_seed_metrics.csv") if (RESULTS / name).is_file()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    found = df[(df.seed == int(parts[0])) & (df.scenario == parts[1])]
    if found.empty:
        raise HTTPException(404, "Experiment not found")
    record = _json_records(found)[0]
    return {"experiment": record, "id": experiment_id,
            "evaluation": "Per-stop time-of-day holdout; dataset has no date.",
            "saved_artifact": True, "session_run": False}


@app.get("/api/experiments/{experiment_id}/communication")
def communication(experiment_id: str):
    if not experiment_id.startswith("42:fedavg_"):
        return {"available": False, "measured": False,
                "reason": "Communication estimates are available for saved FedAvg runs only.",
                "experiment_id": experiment_id}
    mode = "iid" if experiment_id.endswith("fedavg_iid") else "location" if experiment_id.endswith("fedavg_location") else None
    if mode is None:
        raise HTTPException(400, "Use seed:fedavg_iid or seed:fedavg_location")
    rows = _records("fedavg_seed_metrics.csv")
    row = next((item for item in rows if int(item["seed"]) == 42 and item["scenario"] == f"fedavg_{mode}"), None)
    if row is None:
        raise HTTPException(404, "Saved FedAvg communication estimate is unavailable")
    return {"available": True, "measured": False, "basis": "model-size estimate; excludes protocol overhead",
            "experiment_id": experiment_id, "rounds": int(row["rounds"]), "clients": int(row["n_clients"]),
            "model_bytes": int(row["model_bytes"]),
            "client_to_server_bytes_per_round": int(row["client_to_server_bytes_per_round"]),
            "server_to_client_bytes_per_round": int(row["server_to_client_bytes_per_round"]),
            "total_bytes_per_round": int(row["communication_bytes_per_round"]),
            "total_bytes": int(row["communication_bytes_total"])}


@app.get("/api/federated/status")
def federated_status():
    rows = _records("fedavg_round_metrics.csv")
    seeds = _records("fedavg_seed_metrics.csv")
    partitions = sorted({row.get("partition") for row in rows if row.get("partition")})
    client_counts = [int(row["n_clients"]) for row in seeds if row.get("n_clients") is not None]
    return {"available": bool(rows), "rounds": max((int(row["round"]) for row in rows if row.get("round") is not None), default=0),
            "clients": client_counts[0] if client_counts else 4,
            "partitions": partitions,
            "saved_runs": sum(str(row.get("scenario", "")).startswith("fedavg_") for row in seeds),
            "session_run": False,
            "reason": "Loaded saved FedAvg artifacts; no new experiment was run in this API session." if rows else "No saved FedAvg round artifacts are available."}


@app.post("/api/predict")
def prediction(request: PredictionRequest):
    data = _data()
    try:
        location = int(request.location)
    except ValueError:
        raise HTTPException(422, "Select a valid traffic location ID")
    if location not in data.location_vocab:
        raise HTTPException(422, "This location is not present in the training vocabulary")
    time_value = request.time if len(request.time) == 8 else request.time + ":00"
    try:
        parsed = pd.to_datetime(time_value, format="%H:%M:%S", errors="raise")
    except (ValueError, TypeError):
        raise HTTPException(422, "Time must be a valid 24-hour time")
    seconds = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    angle = seconds * (2 * np.pi / 86400)
    row = np.zeros((1, len(data.feature_names)), dtype=np.float32)
    row[0, :2] = [np.sin(angle), np.cos(angle)]
    row[0, 2 + data.location_vocab[location]] = 1
    raw = float(predict(_trained_predictor(), torch.tensor(row)).item())
    category_index = int(np.clip(np.rint(raw), 0, 3))
    return {"location": str(location), "time": parsed.strftime("%H:%M:%S"),
            "model": "centralized CongestionMLP (seed 42)", "raw_ordinal_score": raw,
            "category": LABELS[category_index], "category_index": category_index,
            "basis": "Model trained on the repository's training split; prediction is for a supplied location and time, not live traffic."}
