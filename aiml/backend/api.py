"""FastAPI adapter for the existing Pune traffic ML pipeline."""
from __future__ import annotations

import os
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = ROOT / "dataset" / "pune_traffic.csv"
ARTIFACTS = Path(__file__).with_name("artifacts")
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


def _read_csv(name: str):
    import pandas as pd

    path = RESULTS / name
    if not path.exists():
        raise HTTPException(404, f"Result artifact is unavailable: {name}")
    return pd.read_csv(path)


def _json_records(frame) -> list[dict]:
    import numpy as np
    import pandas as pd

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
def _runtime_metadata() -> dict:
    try:
        return json.loads((ARTIFACTS / "overview.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "Runtime metadata artifact is unavailable") from exc


@lru_cache(maxsize=1)
def _trained_predictor():
    import torch

    from aiml.src.models import CongestionMLP

    metadata = _runtime_metadata()
    model = CongestionMLP(len(metadata["location_vocab"]) + 2)
    state = torch.load(ARTIFACTS / "predictor.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_available": DATASET.exists(),
            "results_available": (RESULTS / "summary_metrics.csv").exists()}


@app.get("/api/overview")
def overview():
    return _runtime_metadata()["overview"]


@app.get("/api/locations")
def locations():
    return {"locations": _runtime_metadata()["location_ids"]}


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
    try:
        location = int(request.location)
    except ValueError:
        raise HTTPException(422, "Select a valid traffic location ID")
    metadata = _runtime_metadata()
    location_vocab = metadata["location_vocab"]
    if str(location) not in location_vocab:
        raise HTTPException(422, "This location is not present in the training vocabulary")
    time_value = request.time if len(request.time) == 8 else request.time + ":00"
    try:
        parsed = datetime.strptime(time_value, "%H:%M:%S")
    except (ValueError, TypeError):
        raise HTTPException(422, "Time must be a valid 24-hour time")
    seconds = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    angle = seconds * (2 * math.pi / 86400)
    import torch

    row = torch.zeros((1, len(location_vocab) + 2), dtype=torch.float32)
    row[0, 0] = math.sin(angle)
    row[0, 1] = math.cos(angle)
    row[0, 2 + location_vocab[str(location)]] = 1
    with torch.inference_mode():
        raw = float(_trained_predictor()(row).item())
    category_index = int(min(3, max(0, round(raw))))
    return {"location": str(location), "time": parsed.strftime("%H:%M:%S"),
            "model": "centralized CongestionMLP (seed 42)", "raw_ordinal_score": raw,
            "category": LABELS[category_index], "category_index": category_index,
            "basis": "Model trained on the repository's training split; prediction is for a supplied location and time, not live traffic."}
