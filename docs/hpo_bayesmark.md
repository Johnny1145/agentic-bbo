# LLAMBO Bayesmark HPO tasks

The `hpo` task family contains the 25 public tasks from the LLAMBO paper:

```text
5 datasets: breast, wine, iris, digits, diabetes
5 models: random_forest, svm, decision_tree, mlp_sgd, adaboost
```

Task IDs use:

```text
hpo_bayesmark_<dataset>_<model>
```

Classification tasks maximize five-fold cross-validation accuracy. Diabetes
tasks minimize five-fold cross-validation MSE. Each task defaults to the paper
budget of five initialization evaluations plus 25 optimizer evaluations.

## Aligned benchmark protocol

For a fixed model and seed, every protocol-aware baseline receives the same
five random configurations from the pinned LLAMBO release before its own
optimizer starts. The paper-result matrix uses seeds `0..4`; the release also
contains fixed configurations for extension seeds `5..9`. Other seeds are
rejected instead of silently changing the initialization distribution.

The same model/seed prefix is reused across all five datasets. GP-EI, TuRBO,
Sobol, LLAMBO, OPRO, Random Search, and Optuna TPE consume this prefix through
task metadata, so their first five evaluated configurations are identical.
Afterward, each method follows its own search rule for 25 evaluations.

GP-EI `raw_samples` and TuRBO `n_candidates` use one shared task-resolved rule:

```text
min(5000, max(2048, 200 * dimension))
```

All 25 public tasks are 2D--8D, so both values are 2048. Tasks in other
benchmark families can declare a different protocol; without one, both
model-based baselines use the same dimension-scaled rule. An explicit
algorithm-specific CLI override remains available for non-canonical runs.

The local paper is authoritative for the search space. In particular, MLP-SGD
is eight-dimensional and includes both `tol` and `validation_fraction`; the
released LLAMBO repository later commented those two parameters out.
Consequently, its stored MLP initialization files contain six fields. At those
five historical evaluations scikit-learn used `tol=1e-4` and
`validation_fraction=0.1`; the v3 task materializes those effective defaults so
the shared points remain valid in the paper-faithful eight-dimensional space.

## Installation

```bash
uv sync --extra dev --extra hpo
```

## Baselines

Scrambled Sobol search:

```bash
uv run python -m bbo.run \
  --task hpo_bayesmark_breast_svm \
  --algorithm sobol_search \
  --max-evaluations 30
```

BoTorch TuRBO-1:

```bash
uv run python -m bbo.run \
  --task hpo_bayesmark_diabetes_random_forest \
  --algorithm botorch_turbo \
  --max-evaluations 30
```

`sobol` and `turbo` are aliases. TuRBO uses the pinned BoTorch v0.18.1
official tutorial implementation; its source tag, commit, notebook blob hash,
and license are stored with the adapter.

## Validation

Run unit and integration tests:

```bash
uv run pytest
```

Evaluate the default configuration of all 25 real tasks:

```bash
uv run python scripts/validate_bayesmark_hpo.py
```

Regenerate the safe NPZ assets and task cards from the pinned LLAMBO release:

```bash
uv run python scripts/generate_bayesmark_hpo_assets.py
```
