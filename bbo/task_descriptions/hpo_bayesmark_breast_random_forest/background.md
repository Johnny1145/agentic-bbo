# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `breast` × `RandomForest`.

## Data Card

| Field | Value |
|---|---|
| Dataset | Breast Cancer Wisconsin (`breast`) |
| Task type | classification |
| Total samples | 569 |
| Published training split | 455 |
| Published held-out split | 114 |
| Numerical features | 30 |
| Categorical features | 0 |
| Training class distribution | class 0: 165, class 1: 290 |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
