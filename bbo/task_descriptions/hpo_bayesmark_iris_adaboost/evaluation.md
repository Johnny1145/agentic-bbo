# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `4e70c997d86da134e105be11342e798b7ff66fc70ad98531b37357d12ac28cb6`.
- Safe NPZ asset SHA-256: `1012e66b17c802211d2d86098bd5c538985d7d28af815c3b883bdea920eba999`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
