# SynRoute — Federated AI for Traffic Intelligence

This directory contains SynRoute's Python data pipeline, congestion model,
reproducible centralized and federated experiments, saved result artifacts,
and FastAPI adapter. The Vite/React frontend is in `src/web/` and calls the API
at `http://localhost:8000/api` by default. The project studies historical Pune
traffic records; it is not a live traffic service.

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

For API development and tests, create a virtual environment and install the
development requirements from the repository root. This includes all runtime
dependencies and the `httpx` test client dependency:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r aiml/requirements-dev.txt
.venv/bin/python -m pytest -c aiml/pytest.ini aiml/tests -q
.venv/bin/python -m aiml.src.experiment
```

For production/runtime-only installs, use `aiml/requirements.txt` instead. To
run the tests from an already activated virtual environment, use:

```sh
python3 -m pytest -c aiml/pytest.ini aiml/tests -q
```

## Start the API

From the repository root, run:

```sh
uvicorn aiml.backend.api:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the React dashboard from the same repository root:

```sh
npm install
npm run dev
```

Open `http://localhost:5173`; the API defaults to `http://localhost:8000`.
The dashboard reports API reachability separately from successful overview
and experiment data requests, and includes the failing endpoint when a request
does not load.

The API loads data and saved metrics from this module. It serves centralized,
local-only, and FedAvg seed metrics, round histories, and model-size
communication estimates from `results/` without launching experiments.
`/api/federated/status` reports saved artifacts and explicitly indicates that
they were not created during the current server session. Communication costs
are parameter-size estimates, not measured transfers. `VITE_API_URL` can set
another API origin; `AIORI_CORS_ORIGINS` configures allowed frontend origins.

## Experiment execution

FedAvg aggregation, round evaluation, and the CLI experiment runner are
implemented in `src/federated.py` and `src/fedavg_experiment.py`. To create a
new three-seed run, explicitly execute `python3 -m aiml.src.fedavg_experiment`
from the repository root. This takes substantially longer than serving the
dashboard. The API never starts training automatically. Communication values
remain model-size estimates and are not measured network transfers.
