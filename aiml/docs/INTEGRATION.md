# Integration guide for Chanchal

## Dataset, split, and constraints

The only experiment source is `aiml/dataset/pune_traffic.csv`. The CSV contains
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
`aiml/results/per_stop_split_counts.csv`.

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
from aiml.src.data_utils import prepare_data, partition_clients

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
seed/client counts in `aiml/results/client_distributions.csv`.

The seed changes IID assignments and model initialization. The split is
seed-independent; location-based assignments are deterministic from sorted
origin IDs and client count, so they do not vary with the seed. To reproduce a
partition, use the same prepared train tensors, `n_clients`, `mode`, and, for
IID, the same seed.

## Model and FedAvg update size

```python
from aiml.src.models import CongestionMLP, train_local, predict

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
one upload per client plus one full-model broadcast per client would be
`8 * 526,724 = 4,213,792` raw bytes per round, excluding protocol overhead.
This is a communication-size estimate, not a measured network transfer.
Clients must use the same model constructor settings for state-dict averaging.
`tests/test_pipeline.py` has a test-only average/load/predict check, not an
aggregation implementation.

## Config and baseline conventions

This workspace has no `config.py`. Suggested federated config mapping:

| Key | Value |
|---|---|
| `data_path` | `aiml/dataset/pune_traffic.csv` |
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

Centralized and client-local MLPs use the same architecture, loss, epochs,
and learning rate. Each local model is evaluated on the shared full test set.
Per-seed/per-client metrics are in `aiml/results/local_client_metrics.csv`; the
centralized, local-only, and reference summary is in
`aiml/results/summary_metrics.csv`. Per-client split counts and hour ranges are in
`aiml/results/per_stop_split_counts.csv`. The local partition score is an
unweighted mean of client metrics per seed; cross-seed standard deviation is
the sample standard deviation. These are not federated results. FedAvg,
convergence tracking, and communication accounting remain for the federated
runner.
