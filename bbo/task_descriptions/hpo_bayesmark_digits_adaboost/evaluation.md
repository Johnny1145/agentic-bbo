# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `fa483b9ea75539b351ec329d15e36b6980f6856d159c7b4538175b9ffe2bf455`.
- Safe NPZ asset SHA-256: `f6266b028cbcd0547fe40802a1916bee645a3ac93a0fa990911a82b33fc612f6`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
