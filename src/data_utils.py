"""Data preparation for Pune congestion prediction.

This dataset has time-of-day but no date, so evaluation is a later-clock-time
holdout within each origin, not a verified future-date forecast. Repeated
origin/time groups are kept intact. Feature vocabularies are learned from
training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Tuple

import numpy as np
import pandas as pd
import torch

LABEL_MAP = {"Very smooth": 0.0, "Smooth": 1.0,
             "Mild congestion": 2.0, "Heavy congestion": 3.0}
REQUIRED_COLUMNS = ("stop_id_from", "arrival_time", "speed", "Degree_of_congestion")
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "pune_traffic.csv"


@dataclass
class PreparedData:
    """Encoded split and metadata; X tensors have identical column order."""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    feature_names: Tuple[str, ...]
    location_vocab: Dict[Hashable, int]
    audit: dict
    stop_split_counts: pd.DataFrame


def load_clean_data(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Load only the supplied CSV and apply the agreed, counted row rules."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pune traffic dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    parsed = pd.to_datetime(frame["arrival_time"], format="%H:%M:%S", errors="coerce")
    speed = pd.to_numeric(frame["speed"], errors="coerce")
    invalid_time = parsed.isna()
    invalid_speed = speed.isna() | ~np.isfinite(speed)
    unknown_label = ~frame["Degree_of_congestion"].isin(LABEL_MAP)
    # Audit whether original row order is chronological by stop using the parsed
    # time-of-day values. Missing timestamps are excluded from this check.
    row_order = pd.DataFrame({"stop_id_from": frame["stop_id_from"], "clock": parsed}).loc[
        ~(invalid_time | invalid_speed | unknown_label)
    ]
    checked_stops = monotonic_stops = transitions = reversals = 0
    for _, stop_rows in row_order.groupby("stop_id_from", sort=False):
        if len(stop_rows) > 1:
            checked_stops += 1
            differences = stop_rows["clock"].diff().dropna()
            transitions += len(differences)
            reversals += int((differences < pd.Timedelta(0)).sum())
            monotonic_stops += int(stop_rows["clock"].is_monotonic_increasing)
    audit = {"input_rows": len(frame), "unparseable_arrival_time_dropped": int(invalid_time.sum()),
             "row_order_stops_checked": checked_stops,
             "row_order_stops_nondecreasing": monotonic_stops,
             "row_order_within_stop_transitions": transitions,
             "row_order_within_stop_time_reversals": reversals,
             "missing_speed_rows": int(speed.isna().sum()),
             "nonfinite_nonmissing_speed_rows": int((speed.notna() & ~np.isfinite(speed)).sum()),
             "missing_or_nonfinite_speed_dropped": int(invalid_speed.sum()),
             "total_rows_dropped_by_cleaning": int((invalid_time | invalid_speed | unknown_label).sum()),
             "unknown_target_dropped": int(unknown_label.sum()),
             "exact_duplicates": int(frame.duplicated().sum()),
             "repeated_origin_time_pairs_retained": int(frame.duplicated(["stop_id_from", "arrival_time"]).sum())}
    keep = ~(invalid_time | invalid_speed | unknown_label)
    clean = frame.loc[keep].copy()
    clean["_time"] = parsed.loc[keep]
    clean["_speed"] = speed.loc[keep].astype(float)
    clean["_target"] = clean["Degree_of_congestion"].map(LABEL_MAP).astype(float)
    # Group key normalizes the parsed clock value and never deduplicates rows.
    clean["_group"] = list(zip(clean["stop_id_from"], clean["_time"].dt.strftime("%H:%M:%S")))
    if clean.empty:
        raise ValueError(
            "No usable rows remain after dropping invalid timestamps, speeds, and target labels "
            f"(input_rows={len(frame)}, invalid_time={int(invalid_time.sum())}, "
            f"invalid_speed={int(invalid_speed.sum())}, unknown_target={int(unknown_label.sum())})"
        )
    if clean.empty:
        raise ValueError(
            "No usable rows remain after dropping invalid timestamps, speeds, and target labels "
            f"(input_rows={len(frame)}, invalid_time={int(invalid_time.sum())}, "
            f"invalid_speed={int(invalid_speed.sum())}, unknown_target={int(unknown_label.sum())})"
        )
    audit["usable_rows"] = len(clean)
    speed_corr = (clean["_speed"].corr(clean["_target"], method="spearman")
                  if clean["_speed"].nunique() > 1 and clean["_target"].nunique() > 1 else np.nan)
    audit["speed_target_spearman"] = float(speed_corr) if np.isfinite(speed_corr) else None
    audit["unique_speeds_with_multiple_targets"] = int(
        clean.groupby("_speed")["_target"].nunique().gt(1).sum()
    )
    audit["unique_speed_values"] = int(clean["_speed"].nunique())
    audit["speed_by_congestion_class"] = {
        str(label): {"count": int(group["_speed"].count()), "min": float(group["_speed"].min()),
                    "median": float(group["_speed"].median()), "max": float(group["_speed"].max())}
        for label, group in clean.groupby("Degree_of_congestion")
    }
    return clean.reset_index(drop=True), audit


def _features(frame: pd.DataFrame, locations: Dict[Hashable, int]) -> tuple[np.ndarray, tuple[str, ...]]:
    seconds = (frame["_time"].dt.hour * 3600 + frame["_time"].dt.minute * 60 + frame["_time"].dt.second).to_numpy()
    angle = seconds * (2.0 * np.pi / 86400.0)
    base = np.column_stack((np.sin(angle), np.cos(angle)))
    loc = np.zeros((len(frame), len(locations)), dtype=np.float32)
    for row, value in enumerate(frame["stop_id_from"].tolist()):
        col = locations.get(value)
        if col is not None:
            loc[row, col] = 1.0
    names = ("time_sin", "time_cos") + tuple(f"origin_{x}" for x in locations)
    return np.column_stack((base, loc)).astype(np.float32), names


def prepare_data(path: str | Path = DEFAULT_DATA_PATH, *, seed: int = 42,
                 test_fraction: float = 0.2) -> PreparedData:
    """Split chronologically by time of day within each origin where possible.

    The CSV has no date and row order is not chronological. For each origin
    with at least two distinct timestamps, choose the time-of-day cutoff that
    best approximates the requested train fraction while keeping tied times
    intact. Origins with only one distinct timestamp use a global time cutoff
    fallback because a per-origin chronological split is impossible.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    frame, audit = load_clean_data(path)
    del seed  # The chronological split is independent of RNG seeds.
    seconds = (frame["_time"].dt.hour * 3600 + frame["_time"].dt.minute * 60 + frame["_time"].dt.second)
    all_times = np.sort(seconds.unique())
    if len(all_times) < 2:
        raise ValueError("Cannot make a time-of-day holdout: fewer than two unique timestamps")
    global_candidates = []
    for cutoff in all_times[1:]:
        n_train = int((seconds < cutoff).sum())
        global_candidates.append((abs(n_train / len(frame) - (1.0 - test_fraction)), cutoff))
    _, global_cutoff = min(global_candidates, key=lambda item: item[0])
    train_mask = pd.Series(False, index=frame.index)
    split_rows = []
    target_train_fraction = 1.0 - test_fraction
    for stop_id, group in frame.groupby("stop_id_from", sort=True):
        group_seconds = seconds.loc[group.index]
        distinct_times = np.sort(group_seconds.unique())
        if len(distinct_times) >= 2:
            choices = []
            for cutoff in distinct_times[1:]:
                train_count = int((group_seconds < cutoff).sum())
                choices.append((abs(train_count / len(group) - target_train_fraction), cutoff))
            _, cutoff = min(choices, key=lambda item: item[0])
            local_train = group_seconds < cutoff
            split_method = "per_stop"
        else:
            cutoff = global_cutoff
            local_train = group_seconds < global_cutoff
            split_method = "global_fallback_single_time"
        train_mask.loc[group.index] = local_train.to_numpy()
        train_rows_for_stop = group.loc[local_train]
        test_rows_for_stop = group.loc[~local_train]
        split_rows.append({
            "stop_id_from": stop_id,
            "split_method": split_method,
            "cutoff_time": f"{int(cutoff)//3600:02d}:{int(cutoff)%3600//60:02d}:{int(cutoff)%60:02d}",
            "train_rows": len(train_rows_for_stop),
            "test_rows": len(test_rows_for_stop),
            "train_first_time": train_rows_for_stop["_time"].min().strftime("%H:%M:%S") if len(train_rows_for_stop) else "",
            "train_last_time": train_rows_for_stop["_time"].max().strftime("%H:%M:%S") if len(train_rows_for_stop) else "",
            "test_first_time": test_rows_for_stop["_time"].min().strftime("%H:%M:%S") if len(test_rows_for_stop) else "",
            "test_last_time": test_rows_for_stop["_time"].max().strftime("%H:%M:%S") if len(test_rows_for_stop) else "",
        })
    train, test = frame.loc[train_mask].copy(), frame.loc[~train_mask].copy()
    stop_split_counts = pd.DataFrame(split_rows)
    train_min, train_max = train["_time"].min(), train["_time"].max()
    test_min, test_max = test["_time"].min(), test["_time"].max()
    vocab_values = sorted(train["stop_id_from"].unique().tolist())
    vocab = {value: i for i, value in enumerate(vocab_values)}
    X_train, names = _features(train, vocab)
    X_test, _ = _features(test, vocab)
    audit.update({"train_rows": len(train), "test_rows": len(test), "feature_count": X_train.shape[1],
                  "train_locations": len(vocab), "test_locations_unseen": int((~test.stop_id_from.isin(vocab)).sum()),
                  "split_kind": "chronological time-of-day holdout; no date available",
                  "split_method": "per-stop cutoff; global cutoff fallback for one-time stops",
                  "global_fallback_cutoff": f"{int(global_cutoff)//3600:02d}:{int(global_cutoff)%3600//60:02d}:{int(global_cutoff)%60:02d}",
                  "split_train_fraction": len(train) / len(frame),
                  "train_hour_range": f"{train_min:%H:%M:%S}-{train_max:%H:%M:%S}",
                  "test_hour_range": f"{test_min:%H:%M:%S}-{test_max:%H:%M:%S}",
                  "stops_total": len(stop_split_counts),
                  "stops_split_per_stop": int((stop_split_counts["split_method"] == "per_stop").sum()),
                  "stops_single_time_fallback": int((stop_split_counts["split_method"] != "per_stop").sum()),
                  "stops_appearing_in_both": int(((stop_split_counts.train_rows > 0) & (stop_split_counts.test_rows > 0)).sum()),
                  "split_target_class_counts_train": train["_target"].value_counts().sort_index().to_dict(),
                  "split_target_class_counts_test": test["_target"].value_counts().sort_index().to_dict()})
    return PreparedData(torch.tensor(X_train), torch.tensor(train["_target"].to_numpy(dtype=np.float32)).reshape(-1, 1),
                        torch.tensor(X_test), torch.tensor(test["_target"].to_numpy(dtype=np.float32)).reshape(-1, 1),
                        train, test, names, vocab, audit, stop_split_counts)


def partition_clients(X: torch.Tensor, y: torch.Tensor, rows: pd.DataFrame, *,
                      n_clients: int = 4, seed: int = 42, mode: str = "iid") -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Partition training tensors into deterministic IID or location-grouped clients."""
    if n_clients < 3 or n_clients > 5 or len(X) != len(y) or len(rows) != len(X):
        raise ValueError("n_clients must be between 3 and 5 and X, y, rows must have equal length")
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(X), dtype=np.int64)
    if mode == "iid":
        order = rng.permutation(len(X))
        assignments[order] = np.arange(len(X)) % n_clients
    elif mode == "location":
        locs = sorted(rows["stop_id_from"].unique().tolist())
        loc_client = {loc: i % n_clients for i, loc in enumerate(locs)}
        assignments[:] = rows["stop_id_from"].map(loc_client).to_numpy()
    else:
        raise ValueError("mode must be 'iid' or 'location'")
    return {f"client_{i}": (X[torch.as_tensor(np.flatnonzero(assignments == i))],
                             y[torch.as_tensor(np.flatnonzero(assignments == i))])
            for i in range(n_clients)}

