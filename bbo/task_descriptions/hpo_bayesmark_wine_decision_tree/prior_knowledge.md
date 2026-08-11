# Domain Prior Knowledge

This is a small three-class dataset. Cross-validation estimates can be sensitive to individual samples, and the 13 numerical features are not rescaled by the benchmark.

Tree complexity is controlled jointly by depth, sample fractions, feature subsampling, and impurity decrease. Useful searches cover both compact trees and less constrained trees rather than assuming monotonicity.

Practical search guidance:

- Respect the declared transform; equal steps in a physical log or logit parameter are not equal optimizer-space steps.
- Compare configurations using the five-fold objective, not the held-out metric.
- Preserve evaluations for interactions between parameters and avoid duplicate decoded configurations.
- Do not assume that a larger model, deeper model, smaller tolerance, or higher learning rate is always better.

No best-known configuration, task result, or test-set-derived prior is provided.
