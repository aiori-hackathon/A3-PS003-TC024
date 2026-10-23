# Pune congestion prediction prototype

The sole experiment dataset is `dataset/pune_traffic.csv`. It has time-of-day
but no date; within-stop file order is not chronological by time-of-day. The
experiment uses per-stop earlier-time-of-day train / later-time-of-day splits
where possible, not a verified future-date forecast.

## Run

```sh
python3 -m pip install torch pytest pandas numpy scikit-learn
python3 -m pytest -q
python3 -m src.experiment
```

## Dashboard

The dashboard reads the existing experiment artifacts and reuses the current
data preparation and model modules. It does not implement FedAvg; federated
rounds and communication measurements are shown as unavailable.

In one terminal, start the API from the project root:

```sh
python3 -m pip install -r requirements.txt
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, install and start the Vite app:

```sh
npm install
npm run dev
```

Open `http://localhost:5173`. Set `VITE_API_URL` at frontend build/dev time
to use a different API origin; `AIORI_CORS_ORIGINS` configures the comma-
separated allowed frontend origins for the API. The first prediction trains
and caches the existing centralized model in-process (seed 42, 60 epochs).
Prediction inputs are a training-known origin and a time of day, not live
traffic measurements.

The target is ordinal: `Very smooth=0`, `Smooth=1`, `Mild congestion=2`,
`Heavy congestion=3`. For seeds 42, 123, and 456, the runner reports the
centralized MLP, IID and location-based local-only models, and training-mean
and training-location-mean reference baselines. Metrics include regression
MAE/RMSE and accuracy/macro-F1 after rounding and clipping predictions.

## Data and partitions

`load_clean_data(path)` drops unparseable `arrival_time`, missing/non-finite
`speed`, and unknown target labels while retaining repeated location/time
pairs. Each stop gets the closest attainable 80/20 clock-time cutoff, with
ties intact. The 613 stops with one distinct usable time cannot be split
chronologically, so they remain intact and use the global `14:58:26` cutoff.
Features are cyclic time and a training-derived origin one-hot
vocabulary. Speed is excluded from inputs because it is strongly associated
with congestion and may be a target proxy.

`prepare_data(path, seed=42)` returns `PreparedData` with train/test tensors,
ordered feature names, training-only location vocabulary, audit values, source
rows and per-stop split counts. `partition_clients(X, y, rows, n_clients=4,
seed=42, mode="iid")` returns `dict[str, tuple[torch.Tensor, torch.Tensor]]`.
`mode="location"` assigns whole stops to clients. Use 3–5 clients; both
partition modes cover the same training examples once.

Full per-stop train/test counts and hour ranges are in
`results/per_stop_split_counts.csv`.

## Model integration

`CongestionMLP(input_dim, hidden_sizes=(32, 16), dropout=0.0)` outputs one
ordinal regression value per row. `train_local(model, X, y, ...)` supports
Huber or MSE loss; `predict(model, X)` returns an `(n, 1)` tensor. The current
input dimension is 4,097; the default model has 131,681 parameters and 526,724
raw float32 bytes per update. See [docs/INTEGRATION.md](docs/INTEGRATION.md)
for FedAvg state-dict keys, communication sizing, split evidence, and
partition reproduction. Federated aggregation and round accounting are not
implemented here.
