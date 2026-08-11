# Constraints

The optimizer may change exactly the 6 parameters below. Bounds are inclusive and transforms are applied before numerical optimization.

| Parameter | Type | Transform | Lower | Upper |
|---|---|---:|---:|---:|
| `max_depth` | integer | linear | 1 | 15 |
| `min_samples_split` | real | logit | 0.01 | 0.99 |
| `min_samples_leaf` | real | logit | 0.01 | 0.49 |
| `min_weight_fraction_leaf` | real | logit | 0.01 | 0.49 |
| `max_features` | real | logit | 0.01 | 0.99 |
| `min_impurity_decrease` | real | linear | 0 | 0.5 |

`linear` is affine scaling, `log` is natural-log scaling, and `logit` is log-odds scaling between the declared physical endpoints. Integer parameters are rounded to the nearest valid integer after inverse transformation.

Fixed estimator parameters:

```json
{
  "max_leaf_nodes": null,
  "n_estimators": 10,
  "random_state": 0
}
```

The dataset split, feature values, target values, model family, five-fold protocol, scoring rule, and fixed parameters may not be changed. Invalid, missing, extra, non-finite, or out-of-range values fail the evaluation.
