# AIORI Traffic Intelligence — AI/ML Module

This directory contains the Python AI/ML contribution: data preparation,
congestion prediction, reproducible centralized and local-only experiments,
client partitions, result artifacts, and a FastAPI adapter. The repository's
shared Vite/React frontend stays at the root in `src/web/`; it calls the API
at `http://localhost:8000/api` by default.

## Dataset and preprocessing

The dataset is `aiml/dataset/pune_traffic.csv`. Its columns include origin and
destination stop IDs, trip identifiers, time of day, speed, trip count, SRI,
and the `Degree_of_congestion` target. The dataset has no calendar date, so
evaluation uses a per-stop time-of-day holdout and is not a verified
future-date forecast.

`aiml/src/data_utils.py` drops rows with unparseable arrival times,
missing/non-finite speed, or unknown congestion labels. Labels are ordinal:
`Very smooth=0`, `Smooth=1`, `Mild congestion=2`, `Heavy congestion=3`.
Features are cyclic time-of-day and a training-derived origin one-hot
vocabulary. Speed is excluded from inputs because it is strongly associated
with congestion. Per-stop splits target an 80/20 train/test ratio while
keeping tied times together; single-time stops use a documented global
cutoff fallback.

## Model and experiments

`aiml/src/models.py` implements an ordinal regression MLP, local training,
deterministic seeding, and prediction helpers. `aiml/src/experiment.py` runs
the centralized MLP, independent local-only client models, and training-mean
and location-mean reference baselines for seeds 42, 123, and 456.

Client partitions support two settings:

- **IID:** seeded random, balanced assignment of training rows.
- **Location-based Non-IID:** whole origins are assigned to clients.

Actual experiment CSVs are stored in `aiml/results/`: per-seed and summary
metrics, local client metrics, client class distributions, and per-stop split
counts. Running the experiment explicitly regenerates these files.

## Install, test, and run experiments

Run these commands from the repository root:

```sh
python3 -m pip install -r aiml/requirements.txt
python3 -m pytest -c aiml/pytest.ini aiml/tests -q
python3 -m aiml.src.experiment
```

## Start the API

From the repository root, run:

```sh
uvicorn aiml.backend.api:app --reload --host 127.0.0.1 --port 8000
```

The API loads data and saved metrics from this module. Its `/api/...` routes
and default port remain unchanged, so the root frontend's API calls continue
to work without frontend code changes. `VITE_API_URL` can set another API
origin; `AIORI_CORS_ORIGINS` configures allowed frontend origins.

## Incomplete capabilities

IID and location-based partitions and independent local-only training are
implemented. FedAvg aggregation, federated rounds, convergence tracking, and
measured communication-cost evaluation are not implemented. The float32
model update size in the integration notes is a parameter-size estimate, not
a measured network transfer.
