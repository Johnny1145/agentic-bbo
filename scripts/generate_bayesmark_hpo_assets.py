#!/usr/bin/env python3
"""Regenerate safe LLAMBO Bayesmark arrays and v3 task cards.

The upstream files are Python pickles.  This script verifies each downloaded
blob before deserializing it, then stores only plain NumPy arrays in compressed
NPZ files.  Runtime task code never loads pickle data.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np

from bbo.core import FloatParam, IntParam
from bbo.tasks.hpo.catalog import (
    ASSET_ROOT,
    DATASETS,
    DESCRIPTION_ROOT,
    LLAMBO_SOURCE_COMMIT,
    MODELS,
    task_id,
)
from bbo.tasks.hpo.initialization import (
    INITIALIZATION_ASSET,
    INITIALIZATION_ASSET_SHA256,
    INITIALIZATION_COUNT,
    PUBLISHED_CONFIG_SEEDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "breast": "60390f5fcf50aee9e873ffb65f6536b360c03e3f493a2bf4727e8f7cfe445163",
    "wine": "65eb94652a23c71b6c6c4ef0b444cf7ea655f792dfd332d20096dbf5826ee20e",
    "iris": "4e70c997d86da134e105be11342e798b7ff66fc70ad98531b37357d12ac28cb6",
    "diabetes": "2f5fd7a46947024051d7953dfb0020b36f23aa93e08ee74f5fbe1b48e83aa094",
    "digits": "fa483b9ea75539b351ec329d15e36b6980f6856d159c7b4538175b9ffe2bf455",
}

DATASET_PRIORS = {
    "breast": (
        "The training labels are moderately imbalanced, so a small number of fold errors can move accuracy. "
        "All 30 inputs are numerical and remain on their released scales."
    ),
    "wine": (
        "This is a small three-class dataset. Cross-validation estimates can be sensitive to individual samples, "
        "and the 13 numerical features are not rescaled by the benchmark."
    ),
    "iris": (
        "This is the smallest public task and has three classes. Prefer robust regions over conclusions drawn from "
        "a single small improvement in five-fold accuracy."
    ),
    "digits": (
        "This is the largest and highest-dimensional public dataset: 64 pixel-intensity features and ten classes. "
        "Model capacity and regularization can interact more strongly here than on the smaller data cards."
    ),
    "diabetes": (
        "The response is standardized using the released training split. MSE is therefore measured in standardized "
        "target units; improvements should be judged on that scale."
    ),
}

MODEL_PRIORS = {
    "random_forest": (
        "Depth, feature subsampling, and leaf-size controls interact. Deeper trees are not guaranteed to improve "
        "cross-validation accuracy or MSE, and aggressive leaf constraints can dominate the depth setting."
    ),
    "decision_tree": (
        "Tree complexity is controlled jointly by depth, sample fractions, feature subsampling, and impurity decrease. "
        "Useful searches cover both compact trees and less constrained trees rather than assuming monotonicity."
    ),
    "svm": (
        "C, gamma, and tolerance span logarithmic coordinates. C and gamma jointly determine the effective RBF model "
        "complexity, so search their interaction instead of varying either parameter in isolation."
    ),
    "mlp_sgd": (
        "The network uses SGD with inverse-scaling learning rate, momentum, early stopping, and a 40-iteration cap. "
        "Learning rate, momentum, power_t, regularization, batch size, and validation fraction interact strongly."
    ),
    "adaboost": (
        "The estimator count and learning rate trade off update size against ensemble length. Neither coordinate has a "
        "universally favorable direction, so evaluate both conservative and aggressive combinations."
    ),
}

UPSTREAM_INITIALIZATION_DIRECTORIES = {
    "random_forest": "RandomForest",
    "svm": "SVM",
    "decision_tree": "DecisionTree",
    "mlp_sgd": "MLP_SGD",
    "adaboost": "AdaBoost",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def generate_initialization_asset() -> dict[str, object]:
    """Download and consolidate the first five pinned configs for seeds 0..9."""

    payload: dict[str, dict[str, list[dict[str, object]]]] = {}
    source_urls: list[str] = []
    for model_key, upstream_directory in UPSTREAM_INITIALIZATION_DIRECTORIES.items():
        by_seed: dict[str, list[dict[str, object]]] = {}
        for seed in PUBLISHED_CONFIG_SEEDS:
            source_url = (
                "https://raw.githubusercontent.com/tennisonliu/LLAMBO/"
                f"{LLAMBO_SOURCE_COMMIT}/bayesmark/configs/{upstream_directory}/{seed}.json"
            )
            with urllib.request.urlopen(source_url) as response:  # noqa: S310 - pinned GitHub HTTPS source.
                raw = response.read()
            source = json.loads(raw)
            if not isinstance(source, list) or len(source) < INITIALIZATION_COUNT:
                raise RuntimeError(
                    f"{model_key}/seed {seed}: expected at least {INITIALIZATION_COUNT} configurations."
                )
            first_five = source[:INITIALIZATION_COUNT]
            if not all(isinstance(config, dict) for config in first_five):
                raise RuntimeError(f"{model_key}/seed {seed}: configuration entries must be mappings.")
            by_seed[str(seed)] = first_five
            source_urls.append(source_url)
        payload[model_key] = by_seed

    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    observed_sha256 = sha256_bytes(serialized)
    if observed_sha256 != INITIALIZATION_ASSET_SHA256:
        raise RuntimeError(
            "Pinned LLAMBO initialization content changed: "
            f"expected {INITIALIZATION_ASSET_SHA256}, got {observed_sha256}."
        )
    INITIALIZATION_ASSET.write_bytes(serialized)
    return {
        "asset": INITIALIZATION_ASSET.name,
        "asset_sha256": observed_sha256,
        "count_per_model_seed": INITIALIZATION_COUNT,
        "seeds": list(PUBLISHED_CONFIG_SEEDS),
        "source_file_count": len(source_urls),
        "source_commit": LLAMBO_SOURCE_COMMIT,
    }


def generate_assets() -> dict[str, object]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_repository": "https://github.com/tennisonliu/LLAMBO",
        "source_commit": LLAMBO_SOURCE_COMMIT,
        "format": "npz_without_pickle",
        "datasets": {},
    }
    for key, definition in DATASETS.items():
        source_url = (
            "https://raw.githubusercontent.com/tennisonliu/LLAMBO/"
            f"{LLAMBO_SOURCE_COMMIT}/bayesmark/data/{key}.pickle"
        )
        with urllib.request.urlopen(source_url) as response:  # noqa: S310 - pinned GitHub HTTPS source.
            raw = response.read()
        observed_source_sha = sha256_bytes(raw)
        if observed_source_sha != SOURCE_SHA256[key]:
            raise RuntimeError(
                f"Refusing to unpickle {key}: expected {SOURCE_SHA256[key]}, got {observed_source_sha}."
            )
        source = pickle.loads(raw)  # noqa: S301 - hash checked against the pinned trusted release above.
        arrays = {
            name: np.asarray(source[name])
            for name in ("train_x", "train_y", "test_x", "test_y")
        }
        expected_shapes = {
            "train_x": (definition.train_samples, definition.feature_count),
            "train_y": (definition.train_samples,),
            "test_x": (definition.test_samples, definition.feature_count),
            "test_y": (definition.test_samples,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise RuntimeError(f"{key}/{name}: expected {expected_shape}, got {arrays[name].shape}.")
            if not np.all(np.isfinite(arrays[name])):
                raise RuntimeError(f"{key}/{name} contains non-finite values.")

        output = ASSET_ROOT / f"{key}.npz"
        np.savez_compressed(output, **arrays)
        manifest["datasets"][key] = {
            "source_url": source_url,
            "source_pickle_sha256": observed_source_sha,
            "asset": output.name,
            "asset_sha256": sha256_file(output),
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": sha256_bytes(value.tobytes(order="C")),
                }
                for name, value in arrays.items()
            },
        }
    manifest["initializations"] = generate_initialization_asset()
    (ASSET_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parameter_table(model_key: str) -> str:
    model = MODELS[model_key]
    rows = ["| Parameter | Type | Transform | Lower | Upper |", "|---|---|---:|---:|---:|"]
    for parameter in model.search_space:
        if not isinstance(parameter, (FloatParam, IntParam)):
            raise TypeError(f"Unexpected non-numeric parameter: {parameter.name}")
        kind = "integer" if isinstance(parameter, IntParam) else "real"
        rows.append(
            f"| `{parameter.name}` | {kind} | {model.transforms[parameter.name]} | "
            f"{parameter.low:g} | {parameter.high:g} |"
        )
    return "\n".join(rows)


def write_descriptions(manifest: dict[str, object]) -> None:
    dataset_manifest = manifest["datasets"]
    for dataset_key, dataset in DATASETS.items():
        for model_key, model in MODELS.items():
            name = task_id(dataset_key, model_key)
            root = DESCRIPTION_ROOT / name
            root.mkdir(parents=True, exist_ok=True)
            objective = dataset.objective_name
            direction = "maximize" if dataset.problem_type == "classification" else "minimize"
            class_text = (
                "not applicable (regression)"
                if not dataset.class_counts_train
                else ", ".join(f"class {index}: {count}" for index, count in enumerate(dataset.class_counts_train))
            )
            source_sha = dataset_manifest[dataset_key]["source_pickle_sha256"]
            asset_sha = dataset_manifest[dataset_key]["asset_sha256"]
            background = f"""# Background

This task reproduces the public LLAMBO/Bayesmark hyperparameter-tuning pair `{dataset_key}` × `{model.display_name}`.

## Data Card

| Field | Value |
|---|---|
| Dataset | {dataset.display_name} (`{dataset_key}`) |
| Task type | {dataset.problem_type} |
| Total samples | {dataset.total_samples} |
| Published training split | {dataset.train_samples} |
| Published held-out split | {dataset.test_samples} |
| Numerical features | {dataset.feature_count} |
| Categorical features | 0 |
| Training class distribution | {class_text} |

The runtime uses the published LLAMBO 80/20 split stored as an offline, non-pickle NumPy asset. No dataset download or model training outside this task is required.
"""
            goal = f"""# Goal

Tune `{model.display_name}` on `{dataset.display_name}` and **{direction}** the primary objective:

```text
{objective} - {direction}
```

One evaluation trains and scores one complete hyperparameter configuration with five-fold cross-validation on the published training split. The held-out split is evaluated only as a logged generalization metric and is never exposed as the optimization objective.

The canonical paper budget is five initialization evaluations followed by 25 optimization evaluations, for 30 evaluations total. For a fixed model and seed, every baseline receives the same five random configurations from the pinned LLAMBO release before its own optimizer starts.
"""
            constraints = f"""# Constraints

The optimizer may change exactly the {model.dimension} parameters below. Bounds are inclusive and transforms are applied before numerical optimization.

{parameter_table(model_key)}

`linear` is affine scaling, `log` is natural-log scaling, and `logit` is log-odds scaling between the declared physical endpoints. Integer parameters are rounded to the nearest valid integer after inverse transformation.

Fixed estimator parameters:

```json
{json.dumps(model.fixed_parameters, indent=2, sort_keys=True)}
```

The dataset split, feature values, target values, model family, five-fold protocol, scoring rule, and fixed parameters may not be changed. Invalid, missing, extra, non-finite, or out-of-range values fail the evaluation.
"""
            prior = f"""# Domain Prior Knowledge

{DATASET_PRIORS[dataset_key]}

{MODEL_PRIORS[model_key]}

Practical search guidance:

- Respect the declared transform; equal steps in a physical log or logit parameter are not equal optimizer-space steps.
- Compare configurations using the five-fold objective, not the held-out metric.
- Preserve evaluations for interactions between parameters and avoid duplicate decoded configurations.
- Do not assume that a larger model, deeper model, smaller tolerance, or higher learning rate is always better.

No best-known configuration, task result, or test-set-derived prior is provided.
"""
            evaluation = f"""# Evaluation Protocol

- Training/test split: released LLAMBO 80/20 split produced with `random_state=0`.
- Cross-validation: scikit-learn five-fold splitting with no fold shuffling.
- Classification objective: mean CV accuracy, maximized.
- Regression objective: mean CV MSE on the released standardized diabetes target, minimized.
- Model stochasticity: estimator `random_state=0` where supported.
- Optimizer initialization: the same five pinned random configurations are shared by all baselines for a fixed model and seed; paper-result runs use seeds 0 through 4.
- Generalization metric: accuracy or MSE on the fixed held-out split, logged but hidden from optimization.
- Source pickle SHA-256: `{source_sha}`.
- Safe NPZ asset SHA-256: `{asset_sha}`.

The task seed controls optimizer reproducibility; it does not change the black-box data split or estimator seed.
"""
            environment = f"""# Environment Setup

Install the repository with the HPO and development extras:

```bash
uv sync --extra dev --extra hpo
```

The task is CPU-compatible and uses only bundled offline arrays. No external service, network access, or GPU is required for evaluation.

Smoke test:

```bash
uv run python - <<'PY'
from bbo.core import TrialSuggestion
from bbo.tasks import create_task
task = create_task("{name}", max_evaluations=1, seed=0)
print(task.sanity_check().ok)
print(task.evaluate(TrialSuggestion(task.spec.search_space.defaults())))
PY
```
"""
            documents = {
                "background.md": background,
                "goal.md": goal,
                "constraints.md": constraints,
                "prior_knowledge.md": prior,
                "evaluation.md": evaluation,
                "environment.md": environment,
            }
            for filename, content in documents.items():
                (root / filename).write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    manifest = generate_assets()
    write_descriptions(manifest)
    print(
        f"Wrote {len(DATASETS)} dataset assets, one fixed-initialization asset, "
        f"and {len(DATASETS) * len(MODELS)} task-description directories."
    )


if __name__ == "__main__":
    main()
