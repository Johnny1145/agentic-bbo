# Domain Prior Knowledge

The response is standardized using the released training split. MSE is therefore measured in standardized target units; improvements should be judged on that scale.

C, gamma, and tolerance span logarithmic coordinates. C and gamma jointly determine the effective RBF model complexity, so search their interaction instead of varying either parameter in isolation.

Practical search guidance:

- Respect the declared transform; equal steps in a physical log or logit parameter are not equal optimizer-space steps.
- Compare configurations using the five-fold objective, not the held-out metric.
- Preserve evaluations for interactions between parameters and avoid duplicate decoded configurations.
- Do not assume that a larger model, deeper model, smaller tolerance, or higher learning rate is always better.

No best-known configuration, task result, or test-set-derived prior is provided.
