# Constraints

The optimizer may change exactly the 8 parameters below. Bounds are inclusive and transforms are applied before numerical optimization.

| Parameter | Type | Transform | Lower | Upper |
|---|---|---:|---:|---:|
| `hidden_layer_sizes` | integer | linear | 50 | 200 |
| `alpha` | real | log | 1e-05 | 10 |
| `batch_size` | integer | linear | 10 | 250 |
| `learning_rate_init` | real | log | 1e-05 | 0.1 |
| `power_t` | real | logit | 0.1 | 0.9 |
| `tol` | real | log | 1e-05 | 0.1 |
| `momentum` | real | logit | 0.001 | 0.999 |
| `validation_fraction` | real | logit | 0.1 | 0.9 |

`linear` is affine scaling, `log` is natural-log scaling, and `logit` is log-odds scaling between the declared physical endpoints. Integer parameters are rounded to the nearest valid integer after inverse transformation.

Fixed estimator parameters:

```json
{
  "early_stopping": true,
  "learning_rate": "invscaling",
  "max_iter": 40,
  "nesterovs_momentum": true,
  "random_state": 0,
  "solver": "sgd"
}
```

The dataset split, feature values, target values, model family, five-fold protocol, scoring rule, and fixed parameters may not be changed. Invalid, missing, extra, non-finite, or out-of-range values fail the evaluation.
