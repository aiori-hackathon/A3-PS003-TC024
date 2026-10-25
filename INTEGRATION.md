# Integration guide for Chanchal

## Dataset, split, and constraints

The only experiment source is `dataset/pune_traffic.csv`. The CSV contains
time-of-day but no date. Original row order is not reliably chronological
within stops: after cleaning, 3,606 stops have multiple rows, only 904 are
nondecreasing by time-of-day, and there are 24,109 reversals among 62,232
within-stop transitions.

The split is therefore by time of day **per stop**, choosing the attainable
cutoff closest to 80/20 while keeping tied timestamps together. Of 4,208
usable stops, 3,595 have at least two distinct usable times and occur on both
sides of their own cutoff. The other 613 have only one distinct time (602
have one row); they cannot be split chronologically without duplicating a row
or separating same-time observations. Those stops are kept intact and assigned
using a global 14:58:26 time-of-day cutoff. The resulting split has 52,735
training rows and 13,705 test rows (79.37% training).

Because dates are unavailable, even the per-stop split is a later-clock-time
holdout, not a verified future-date forecast. Overall ranges overlap across
stops: train spans 09:00:00-18:16:55; test spans 09:05:27-18:19:35. Each
per-stop cutoff and its row counts/ranges are recorded in
`results/per_stop_split_counts.csv`.

## Speed and target relationship

On the cleaned rows, speed has Spearman correlation -0.570 with the ordinal
congestion score (higher means heavier congestion). Median speed decreases
from 40.31 for `Very smooth` to 14.36 for `Heavy congestion`. Class ranges
overlap substantially; the same exact speed has multiple labels for 23 of
33,573 distinct speed values. The target is not a strict deterministic
function of this stored speed column, but speed is a strong contemporaneous
proxy, so it is excluded from model inputs. If the intended objective is to
predict traffic flow itself, speed regression is a reasonable alternative;
these runs retain the requested congestion target.

Cleaning drops rows with unparseable `arrival_time`, missing/non-finite
`speed`, or unknown target. Repeated `(stop_id_from, arrival_time)` pairs are
retained and remain on one side of the split. The target mapping is
`Very smooth=0`, `Smooth=1`, `Mild congestion=2`, `Heavy congestion=3`.
Features are cyclic time-of-day sin/cos and origin one-hot columns. The origin
vocabulary is fit from training data only; test-only origins map to zeros.

## Data and client interfaces

```python
from src.data_utils import prepare_data, partition_clients

prepared = prepare_data(path, seed=42, test_fraction=0.2)
# X_train: float32 Tensor [n_train, input_dim]
# y_train: float32 Tensor [n_train, 1], ordinal values 0..3
# X_test:  float32 Tensor [n_test, input_dim]
# y_test:  float32 Tensor [n_test, 1]

iid = partition_clients(prepared.X_train, prepared.y_train,
                        prepared.train_rows, n_clients=4,
                        seed=42, mode="iid")
noniid = partition_clients(prepared.X_train, prepared.y_train,
                           prepared.train_rows, n_clients=4,
                           seed=42, mode="location")
# dict[str, tuple[Tensor, Tensor]]: client_id -> (X_client, y_client)
```

`n_clients` must be 3, 4, or 5; default is 4. IID uses a seeded random
permutation and round-robin assignment. Location-based Non-IID sorts the
training origins and assigns each whole origin round-robin to one client.
Both scenarios contain exactly the same training examples once. The runner
prints per-client sample counts and ordinal class distributions and saves all
seed/client counts in `results/client_distributions.csv`.

The seed changes IID assignments and model initialization. The split is
seed-independent; location-based assignments are deterministic from sorted
origin IDs and client count, so they do not vary with the seed. To reproduce a
partition, use the same prepared train tensors, `n_clients`, `mode`, and, for
IID, the same seed.

## Model and FedAvg update size

```python
from src.models import CongestionMLP, train_local, predict

model = CongestionMLP(input_dim, hidden_sizes=(32, 16), dropout=0.0)
train_local(model, X_client, y_client, epochs=60,
            learning_rate=0.01, loss_name="huber")
predictions = predict(model, X_test)  # Tensor [n_test, 1]
```

Default state-dict keys and shapes:

| Key | Shape |
|---|---|
| `network.0.weight` | `[32, input_dim]` |
| `network.0.bias` | `[32]` |
| `network.2.weight` | `[16, 32]` |
| `network.2.bias` | `[16]` |
| `network.4.weight` | `[1, 16]` |
| `network.4.bias` | `[1]` |

There are no BatchNorm buffers. Parameter count is `32 * input_dim + 577`.
For this split, `input_dim=4097`, so each model has **131,681 parameters** and
one float32 update is **526,724 bytes** (`131,681 * 4`). With four clients,
the runner estimates **2,106,896 bytes client-to-server** and **2,106,896
bytes server-to-client** per round, or **4,213,792 total bytes per round**.
This counts one full upload and one full-model delivery per client and excludes
transport overhead, compression, retries, and encryption. It is an estimate,
not a measured network transfer. Clients must use identical model constructor
settings for state-dict averaging.

## Config and baseline conventions

This workspace has no `config.py`. Suggested federated config mapping:

| Key | Value |
|---|---|
| `data_path` | `dataset/pune_traffic.csv` |
| `seed` | `42`, `123`, or `456` |
| `test_fraction` | `0.2` target per stop; nearest attainable cutoff |
| `n_clients` | `4` default; allowed 3-5 |
| `partition_mode` | `iid` or `location` |
| `input_dim` | `prepared.X_train.shape[1]` (4,097 here) |
| `hidden_sizes` | `(32, 16)` |
| `dropout` | `0.0` |
| `epochs` | `60` |
| `learning_rate` | `0.01` |
| `loss_name` | `huber` |
| `batch_size` | `None` (full batch, preserves reported run); set a positive integer for mini-batches |

Centralized and client-local MLPs use the same architecture, loss, epochs,
and learning rate. Each local model is evaluated on the shared full test set.
Per-seed/per-client metrics are in `results/local_client_metrics.csv`; the
centralized, local-only, and reference summary is in
`results/summary_metrics.csv`. Per-client split counts and hour ranges are in
`results/per_stop_split_counts.csv`. The local partition score is an
unweighted mean of client metrics per seed; cross-seed standard deviation is
the sample standard deviation. These are not federated results. FedAvg,
convergence tracking, and communication accounting remain for the federated
runner.

## Federated engine, evaluation, and communication

`src/federated.py` implements client initialization from the current global
model, local training, sample-weighted FedAvg aggregation, and global model
updates over multiple rounds. `src/fedavg.py` keeps backward-compatible import
names. `train_local` accepts `batch_size=None` for full-batch training or a
positive integer for shuffled mini-batches; the experiment CLI exposes this as
`--batch-size`. The default remains full-batch so the reported results retain
their original training configuration.

`src/evaluation.py` provides shared prediction, global, and per-client metrics
(MAE, RMSE, rounded accuracy, and rounded macro-F1). The FedAvg runner logs
global metrics each round and final global-model metrics for each client's
held-out test partition. `src/communication.py` reports model size and
separate client-to-server and server-to-client byte estimates per round and
over the full run.

`src/fedavg_experiment.py` runs IID and location-based Non-IID partitions for
seeds 42/123/456. Defaults are four clients, 60 rounds, one local epoch per
round, full-batch training, learning rate 0.01, and Huber loss. One local epoch
per round gives 60 client data passes overall, matching the centralized and
client-local baselines while aggregating after each pass.

Run `python -m src.experiment` first to create centralized and client-local
results, then run `python -m src.fedavg_experiment`. Add `--batch-size 512` to
the latter command for 512-row mini-batches. FedAvg refreshes its two reference
baselines using the same split and merges IID and location-based rows into
`results/summary_metrics.csv`, preserving centralized and local-only rows.
Outputs include `fedavg_seed_metrics.csv`, `fedavg_round_metrics.csv`,
`fedavg_client_metrics.csv`, and `fedavg_client_distributions.csv` under
`results/`. The runner also writes partition-specific copies to
`experiments/iid/` and `experiments/non_iid/`, with a combined summary in
`experiments/results/`. Focused engine tests are in
`tests/test_federated.py` and the full suite runs with `python -m pytest -q`.
