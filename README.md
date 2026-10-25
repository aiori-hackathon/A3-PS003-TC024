# AIORI-3-HACKATHON-PROJECTS
The youtube video link is : https://youtu.be/MnNg1J3wpDo

## Traffic congestion experiments

The model predicts `Degree_of_congestion` (Very smooth=0, Smooth=1, Mild congestion=2, Heavy congestion=3) using time-of-day and origin features. Four IID or location-based clients use a per-stop time-of-day holdout. The dataset has no dates, so this is not a future-date forecast.

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m src.experiment
python -m src.fedavg_experiment
```

The default FedAvg run uses full-batch local training. To use mini-batches,
pass a positive batch size, for example `python -m src.fedavg_experiment
--batch-size 512`.

Scores below are mean +/- sample standard deviation across seeds 42, 123, and 456.

| Scenario | MAE | RMSE | Rounded accuracy | Rounded macro-F1 |
|---|---:|---:|---:|---:|
| Training mean baseline | 0.8513 +/- 0.0000 | 1.0385 +/- 0.0000 | 0.3118 +/- 0.0000 | 0.1188 +/- 0.0000 |
| Per-location mean baseline | 0.6242 +/- 0.0000 | 0.8248 +/- 0.0000 | 0.4931 +/- 0.0000 | 0.4362 +/- 0.0000 |
| Centralized MLP | 0.6300 +/- 0.0060 | 0.8381 +/- 0.0094 | 0.5056 +/- 0.0050 | 0.4731 +/- 0.0090 |
| Local-only IID average | 0.6877 +/- 0.0077 | 0.9018 +/- 0.0141 | 0.4577 +/- 0.0038 | 0.4267 +/- 0.0037 |
| Local-only location average | 0.8235 +/- 0.0153 | 1.0390 +/- 0.0154 | 0.3610 +/- 0.0017 | 0.2503 +/- 0.0031 |
| FedAvg IID | 0.6773 +/- 0.0136 | 0.8819 +/- 0.0091 | 0.4632 +/- 0.0035 | 0.3685 +/- 0.0072 |
| FedAvg location-based | 0.7084 +/- 0.0444 | 0.8978 +/- 0.0485 | 0.4506 +/- 0.0272 | 0.3383 +/- 0.0484 |

The location-mean baseline had the lowest MAE and RMSE; centralized training had the highest rounded accuracy and macro-F1. FedAvg did not exceed centralized training on these metrics. These are preliminary time-of-day holdout results.

FedAvg used four clients, 60 rounds, and one local epoch per round. Estimated
communication was 2,106,896 bytes client-to-server and 2,106,896 bytes
server-to-client per round. The combined estimate is 4,213,792 bytes per round
(252,827,520 bytes over 60 rounds per partition and seed), excluding protocol
overhead. Detailed outputs are in `results/`.

The implementation is organized into `src/federated.py` (local client updates and FedAvg), `src/evaluation.py` (global, per-client, and round metrics), and `src/communication.py` (model size and directional byte estimates). `src/fedavg.py` remains as a compatibility import. Local training supports full-batch mode by default and shuffled mini-batches with `--batch-size 512`.

For the default 131,681-parameter float32 model, communication is estimated at
2,106,896 bytes client-to-server and 2,106,896 bytes server-to-client per
round with four clients. The combined estimate is 4,213,792 bytes per round,
or 252,827,520 bytes over 60 rounds; protocol overhead is excluded.

Per-round global metrics, final per-client test metrics, and directional
communication estimates are saved under `results/`. Partition-specific runs
are saved in `experiments/iid/` and `experiments/non_iid/`; their combined
summary is saved in `experiments/results/`. Run the tests with
`python -m pytest -q`.
