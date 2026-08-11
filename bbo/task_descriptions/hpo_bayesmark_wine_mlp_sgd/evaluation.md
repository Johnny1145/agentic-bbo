# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `65eb94652a23c71b6c6c4ef0b444cf7ea655f792dfd332d20096dbf5826ee20e`.
- Safe NPZ asset SHA-256: `10beaf6363c8ff5fb9d387dd46772f96103ba4887d4fe776317d3f3bc905ab60`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
