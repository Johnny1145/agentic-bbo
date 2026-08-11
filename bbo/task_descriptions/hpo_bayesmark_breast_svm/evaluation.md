# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `60390f5fcf50aee9e873ffb65f6536b360c03e3f493a2bf4727e8f7cfe445163`.
- Safe NPZ asset SHA-256: `ded38121ffeede95f4544b7e2d45a8aacedc8da8ff1659b6f49bdb0716b0dace`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
