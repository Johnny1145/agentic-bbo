# Domain Prior Knowledge

This is the largest and highest-dimensional public dataset: 64 pixel-intensity features and ten classes. Model capacity and regularization can interact more strongly here than on the smaller data cards.

Tree complexity is controlled jointly by depth, sample fractions, feature subsampling, and impurity decrease. Useful searches cover both compact trees and less constrained trees rather than assuming monotonicity.

Practical search guidance:

- Respect the declared transform; equal steps in a physical log or logit parameter are not equal optimizer-space steps.
- Compare configurations using the five-fold objective, not the held-out metric.
- Preserve evaluations for interactions between parameters and avoid duplicate decoded configurations.
- Do not assume that a larger model, deeper model, smaller tolerance, or higher learning rate is always better.

No best-known configuration, task result, or test-set-derived prior is provided.
