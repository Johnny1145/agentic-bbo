# GIT-BO benchmark adapter

This adapter implements the numerical optimizer described in *Scalable
High-Dimensional Bayesian Optimization with TabPFN Priors* (GIT-BO, ICLR
2026). It intentionally excludes direct-SMILES molecular tasks because the
method operates on bounded numerical vectors.

## Algorithm

For each optimizer step the adapter:

1. maps every numerical parameter to the unit cube, preserving task-owned
   linear/log transforms;
2. refits one frozen TabPFN v2 regressor to all successful observations;
3. evaluates predictive-mean gradients on a scrambled Sobol pool and forms
   the diagnostic/Fisher matrix `H = mean(g g^T)`;
4. takes the leading `min(10, d)` eigenvectors of `H`;
5. samples reduced coordinates uniformly from `[-1, 1]^r` around the
   observed unit-cube centroid, projects them back to the full space, and
   clips to `[0, 1]^d`;
6. selects the non-duplicate candidate maximizing `mean + 2.33 * std`.

The TabPFN model is frozen: gradients are taken only with respect to query
inputs. The dependency is pinned to the official PriorLabs commit
`f87b137bc7c4fe021dda76e62720098541575d37`, whose regressor supports
`fit_with_differentiable_input`.

## Benchmark alignment

Paper defaults retained:

- TabPFN v2, one estimator;
- requested subspace rank `r=10`;
- UCB coefficient `beta=2.33`;
- observed-point centroid as the subspace reference;
- scrambled Sobol gradient sampling and uniform reduced-space sampling.

Benchmark-wide alignment retained:

- exactly 2048 gradient points and 2048 UCB candidates per optimizer step,
  matching the aligned GP-EI/TuRBO internal candidate count;
- shared initialization prefixes: BBOPlace and DBTune `50`, BBOB `20`, and
  HPO the five fixed LLAMBO configurations;
- batch size one and the existing two-seed protocol for every family.

The 2048 queries are evaluated in deterministic chunks to bound peak GPU
memory. The adapter default is 128. The formal remote workflow uses the
largest tested safe size while SGLang remains resident: BBOPlace 1024,
DBTune 128, and BBOB/HPO 2048. Chunking does not change the sampled points,
Fisher sum, UCB values, or selected candidate.

Each acquisition records the dependency commit, model version, effective
rank, Fisher eigenvalues, pool sizes, batch size, UCB statistics, transforms,
and device in `suggestion_metadata`. Startup, degenerate-target, non-finite,
and exhausted-discrete-pool fallbacks are explicitly recorded as Sobol
fallbacks rather than silently presented as GIT-BO acquisitions.
