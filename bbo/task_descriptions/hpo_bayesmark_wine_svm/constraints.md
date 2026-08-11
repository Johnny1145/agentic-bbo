# Constraints

The optimizer may change exactly the 3 parameters below. Bounds are inclusive and transforms are applied before numerical optimization.

| Parameter | Type | Transform | Lower | Upper |
|---|---|---:|---:|---:|
| `C` | real | log | 1 | 1000 |
| `gamma` | real | log | 0.0001 | 0.001 |
| `tol` | real | log | 1e-05 | 0.1 |

`linear` is affine scaling, `log` is natural-log scaling, and `logit` is log-odds scaling between the declared physical endpoints. Integer parameters are rounded to the nearest valid integer after inverse transformation.

Fixed estimator parameters:

```json
{
  "kernel": "rbf"
}
```

The dataset split, feature values, target values, model family, five-fold protocol, scoring rule, and fixed parameters may not be changed. Invalid, missing, extra, non-finite, or out-of-range values fail the evaluation.
