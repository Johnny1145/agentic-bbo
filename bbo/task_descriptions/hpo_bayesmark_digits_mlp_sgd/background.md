# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `digits` × `MLP-SGD`.

## Data Card

| Field | Value |
|---|---|
| Dataset | Optical Digits (`digits`) |
| Task type | classification |
| Total samples | 1797 |
| Published training split | 1437 |
| Published held-out split | 360 |
| Numerical features | 64 |
| Categorical features | 0 |
| Training class distribution | class 0: 151, class 1: 147, class 2: 141, class 3: 154, class 4: 151, class 5: 142, class 6: 137, class 7: 140, class 8: 135, class 9: 139 |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
