# Domain Prior Knowledge

The training labels are moderately imbalanced, so a small number of fold errors can move accuracy. All 30 inputs are numerical and remain on their released scales.

Depth, feature subsampling, and leaf-size controls interact. Deeper trees are not guaranteed to improve cross-validation accuracy or MSE, and aggressive leaf constraints can dominate the depth setting.

Practical search guidance:

- Respect the declared transform; equal steps in a physical log or logit parameter are not equal optimizer-space steps.
- Compare configurations using the five-fold objective, not the held-out metric.
- Preserve evaluations for interactions between parameters and avoid duplicate decoded configurations.
- Do not assume that a larger model, deeper model, smaller tolerance, or higher learning rate is always better.

No best-known configuration, task result, or test-set-derived prior is provided.
