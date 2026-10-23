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

from src.data_utils import DEFAULT_DATA_PATH, prepare_data
from src.models import CongestionMLP, predict, seed_everything, train_local

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LABELS = ["Very smooth", "Smooth", "Mild congestion", "Heavy congestion"]

app = FastAPI(title="AIORI Traffic Intelligence API", version="1.0.0")
origins = os.getenv("AIORI_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in origins if x.strip()],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    location: str
    time: str = Field(pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    model: Literal["centralized"] = "centralized"


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise HTTPException(404, f"Result artifact is unavailable: {name}")
    return pd.read_csv(path)


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
    seeds = _read_csv("seed_metrics.csv")
    central = summary.loc[summary.scenario == "centralized"]
    best = central.iloc[0].to_dict() if len(central) else None
    recent = seeds.to_dict(orient="records")
    buckets = data.train_rows["_time"].dt.floor("15min")
    counts = buckets.value_counts().sort_index()
    trend = [{"time": t.strftime("%H:%M"), "observations": int(v)} for t, v in counts.items()]
    return {"project": "AIORI Traffic Intelligence", "target": "Ordinal congestion category",
            "labels": LABELS, "locations": len(data.location_vocab), "train_rows": len(data.train_rows),
            "test_rows": len(data.test_rows), "feature_count": len(data.feature_names),
            "split": data.audit["split_kind"], "best_verified": best,
            "experiments": recent, "observation_trend": trend,
            "federated_available": False, "communication_measured": False}


@app.get("/api/locations")
def locations():
    data = _data()
    return {"locations": sorted(str(x) for x in data.train_rows["stop_id_from"].unique())}


@app.get("/api/experiments")
def experiments():
    df = _read_csv("seed_metrics.csv")
    return {"experiments": df.to_dict(orient="records")}


@app.get("/api/experiments/{experiment_id}")
def experiment_detail(experiment_id: str):
    parts = experiment_id.split(":")
    if len(parts) != 2 or not parts[0].isdigit():
        raise HTTPException(400, "Use experiment IDs in the format seed:scenario")
    df = _read_csv("seed_metrics.csv")
    found = df[(df.seed == int(parts[0])) & (df.scenario == parts[1])]
    if found.empty:
        raise HTTPException(404, "Experiment not found")
    return {"experiment": found.iloc[0].to_dict(), "id": experiment_id,
            "evaluation": "Per-stop time-of-day holdout; dataset has no date."}


@app.get("/api/experiments/{experiment_id}/communication")
def communication(experiment_id: str):
    return {"available": False, "reason": "The current runner does not implement FedAvg or measure serialized/network communication bytes.",
            "experiment_id": experiment_id}


@app.get("/api/federated/status")
def federated_status():
    return {"available": False, "rounds": None, "clients": 4,
            "reason": "Client partitions and local-only models exist, but a federated aggregation runner and round metrics have not been implemented."}


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
