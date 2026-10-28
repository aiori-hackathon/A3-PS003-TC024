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

Scores below are mean ± sample standard deviation across seeds 42, 123, and 456.

| Scenario | MAE | RMSE | Rounded accuracy | Rounded macro-F1 |
|---|---:|---:|---:|---:|
| Training mean baseline | 0.8513 ± 0.0000 | 1.0385 ± 0.0000 | 0.3118 ± 0.0000 | 0.1188 ± 0.0000 |
| Per-location mean baseline | 0.6242 ± 0.0000 | 0.8248 ± 0.0000 | 0.4931 ± 0.0000 | 0.4362 ± 0.0000 |
| Centralized MLP | 0.6300 ± 0.0060 | 0.8381 ± 0.0094 | 0.5056 ± 0.0050 | 0.4731 ± 0.0090 |
| Local-only IID average | 0.6877 ± 0.0077 | 0.9018 ± 0.0141 | 0.4577 ± 0.0038 | 0.4267 ± 0.0037 |
| Local-only location average | 0.8235 ± 0.0153 | 1.0390 ± 0.0154 | 0.3610 ± 0.0017 | 0.2503 ± 0.0031 |
| FedAvg IID | 0.6773 ± 0.0136 | 0.8819 ± 0.0091 | 0.4632 ± 0.0035 | 0.3685 ± 0.0072 |
| FedAvg location-based | 0.7084 ± 0.0444 | 0.8978 ± 0.0485 | 0.4506 ± 0.0272 | 0.3383 ± 0.0484 |

The location-mean baseline had the lowest MAE and RMSE; centralized training had the highest rounded accuracy and macro-F1. FedAvg did not exceed centralized training on these metrics. These are preliminary time-of-day holdout results.

FedAvg used four clients, 60 rounds, and one local epoch per round. Estimated communication was 4,213,792 bytes per round (252,827,520 bytes over 60 rounds per partition and seed), excluding protocol overhead. Detailed outputs are in `results/`.
