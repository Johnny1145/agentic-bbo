# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `diabetes` × `DecisionTree`.

## Data Card

| Field | Value |
|---|---|
| Dataset | Diabetes Progression (`diabetes`) |
| Task type | regression |
| Total samples | 442 |
| Published training split | 353 |
| Published held-out split | 89 |
| Numerical features | 10 |
| Categorical features | 0 |
| Training class distribution | not applicable (regression) |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
