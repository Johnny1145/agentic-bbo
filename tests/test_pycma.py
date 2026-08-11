from __future__ import annotations

import warnings
from pathlib import Path

from cma.evolution_strategy import InjectionWarning

from bbo.core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.algorithms import create_algorithm
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
from bbo.tasks import create_task


def test_pycma_default_popsize_is_two() -> None:
    algorithm = create_algorithm("pycma")

    assert algorithm.popsize == 2


def test_pycma_runs_on_numeric_task(tmp_path: Path) -> None:
    task = create_task("bbob_f01_d10", max_evaluations=14, seed=5)
    logger = JsonlMetricLogger(tmp_path / "pycma.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=create_algorithm("pycma", sigma_fraction=0.15, popsize=4),
        logger_backend=logger,
        config=ExperimentConfig(seed=5, resume=False, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    records = logger.load_records()

    assert summary.n_completed == 14
    assert len(records) == 14
    assert summary.incumbents
    assert summary.best_primary_objective is not None
    assert summary.best_primary_objective <= records[0].objectives[task.spec.primary_objective.name]


def test_pycma_runs_on_mixed_task_via_onehot_converter() -> None:
    task_spec = TaskSpec(
        name="mixed_pycma_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=8,
    )
    algorithm = create_algorithm("pycma", sigma_fraction=0.15, popsize=4)
    algorithm.setup(task_spec, seed=5)

    seen_activations: set[str] = set()
    for trial_id in range(8):
        suggestion = algorithm.ask()
        seen_activations.add(str(suggestion.config["activation"]))
        loss = float(suggestion.config["lr"]) * 15.0 + float(suggestion.config["depth"])
        loss += {"relu": 0.25, "gelu": 0.1, "tanh": 0.4}[str(suggestion.config["activation"])]
        observation = TrialObservation.from_evaluation(
            TrialSuggestion(
                config=dict(suggestion.config),
                trial_id=trial_id,
                metadata=dict(suggestion.metadata),
            ),
            EvaluationResult(objectives={"loss": loss}),
        )
        algorithm.tell(observation)

    assert seen_activations
    assert seen_activations <= {"relu", "gelu", "tanh"}
    assert algorithm.incumbents()


def test_pycma_tells_exact_latent_vectors_for_rounded_configs() -> None:
    task_spec = TaskSpec(
        name="rounded_pycma_demo",
        search_space=SearchSpace(
            [
                IntParam("workers", low=1, high=64, default=8),
                CategoricalParam("mode", choices=("a", "b", "c"), default="a"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=100,
    )
    algorithm = create_algorithm("pycma", sigma_fraction=0.18, popsize=10)
    algorithm.setup(task_spec, seed=7)

    with warnings.catch_warnings():
        warnings.simplefilter("error", InjectionWarning)
        for trial_id in range(100):
            suggestion = algorithm.ask()
            loss = float(suggestion.config["workers"])
            loss += {"a": 0.0, "b": 0.1, "c": 0.2}[str(suggestion.config["mode"])]
            algorithm.tell(
                TrialObservation.from_evaluation(
                    TrialSuggestion(
                        config=dict(suggestion.config),
                        trial_id=trial_id,
                        metadata=dict(suggestion.metadata),
                    ),
                    EvaluationResult(objectives={"loss": loss}),
                )
            )
