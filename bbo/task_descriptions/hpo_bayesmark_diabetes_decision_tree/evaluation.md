# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `2f5fd7a46947024051d7953dfb0020b36f23aa93e08ee74f5fbe1b48e83aa094`.
- Safe NPZ asset SHA-256: `c27ee3ef90a34cb33dcc13794029ecebfc4d44b7f9e1a6386d2a8b17d070fb60`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
