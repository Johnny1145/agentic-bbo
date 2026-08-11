# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `wine` × `RandomForest`.

## Data Card

| Field | Value |
|---|---|
| Dataset | Wine Recognition (`wine`) |
| Task type | classification |
| Total samples | 178 |
| Published training split | 142 |
| Published held-out split | 36 |
| Numerical features | 13 |
| Categorical features | 0 |
| Training class distribution | class 0: 45, class 1: 55, class 2: 42 |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
