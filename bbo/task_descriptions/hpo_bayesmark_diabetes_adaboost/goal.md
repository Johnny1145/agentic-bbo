# Goal

Tune `AdaBoost` on `Diabetes Progression` and **minimize** the primary objective:

```text
mse - minimize
```

One evaluation trains and scores one complete hyperparameter configuration with five-fold cross-validation on the published training split. The held-out split is evaluated only as a logged generalization metric and is never exposed as the optimization objective.

The canonical paper budget is five initialization evaluations followed by 25 optimization evaluations, for 30 evaluations total. For a fixed model and seed, every baseline receives the same five random configurations from the pinned LLAMBO release before its own optimizer starts.
