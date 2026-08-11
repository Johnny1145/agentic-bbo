# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `iris` × `DecisionTree`.

## Data Card

| Field | Value |
|---|---|
| Dataset | Iris (`iris`) |
| Task type | classification |
| Total samples | 150 |
| Published training split | 120 |
| Published held-out split | 30 |
| Numerical features | 4 |
| Categorical features | 0 |
| Training class distribution | class 0: 39, class 1: 37, class 2: 44 |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
