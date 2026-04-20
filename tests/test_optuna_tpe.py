from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from bbo.algorithms import ALGORITHM_REGISTRY, OptunaTpeAlgorithm
from bbo.algorithms.model_based.optuna_utils import build_distributions
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
    TrialStatus,
    TrialSuggestion,
)
from bbo.run import build_arg_parser, run_single_experiment


def _mixed_task_spec(*, max_evaluations: int = 8) -> TaskSpec:
    return TaskSpec(
        name="mixed_optuna_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=max_evaluations,
        metadata={"display_name": "Mixed Optuna Demo"},
    )


def _make_observation(suggestion: TrialSuggestion, trial_id: int) -> TrialObservation:
    normalized = TrialSuggestion(
        config=dict(suggestion.config),
        trial_id=trial_id,
        budget=suggestion.budget,
        metadata=dict(suggestion.metadata),
    )
    if trial_id == 1:
        result = EvaluationResult(
            status=TrialStatus.FAILED,
            error_type="SyntheticFailure",
            error_message="planned failure for replay coverage",
        )
    else:
        activation_penalty = {"relu": 0.25, "gelu": 0.1, "tanh": 0.4}[str(suggestion.config["activation"])]
        loss = float(suggestion.config["lr"]) * 10.0 + float(suggestion.config["depth"]) + activation_penalty
        result = EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"loss": loss},
        )
    return TrialObservation.from_evaluation(normalized, result)


def test_optuna_tpe_search_space_mapping_preserves_mixed_types() -> None:
    distributions = build_distributions(_mixed_task_spec().search_space)

    lr = distributions["lr"]
    depth = distributions["depth"]
    activation = distributions["activation"]
    assert lr.low == pytest.approx(1e-4)
    assert lr.high == pytest.approx(1e-1)
    assert lr.log is True
    assert depth.low == 2
    assert depth.high == 8
    assert tuple(activation.choices) == ("relu", "gelu", "tanh")


def test_optuna_tpe_is_registered_and_cli_visible_without_entering_suite() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "optuna_tpe" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["optuna_tpe"].family == "model_based"
    assert ALGORITHM_REGISTRY["optuna_tpe"].numeric_only is False
    assert "optuna_tpe" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "optuna_tpe"]).algorithm == "optuna_tpe"


def test_optuna_tpe_rejects_multi_objective_tasks() -> None:
    spec = TaskSpec(
        name="multi_objective_demo",
        search_space=SearchSpace([FloatParam("x", low=0.0, high=1.0, default=0.5)]),
        objectives=(
            ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),
            ObjectiveSpec("score", ObjectiveDirection.MAXIMIZE),
        ),
        max_evaluations=4,
    )

    algorithm = OptunaTpeAlgorithm()
    with pytest.raises(ValueError, match="exactly one objective"):
        algorithm.setup(spec, seed=5)


def test_optuna_tpe_replay_reconstructs_mixed_history_and_next_suggestion() -> None:
    task_spec = _mixed_task_spec(max_evaluations=7)
    algorithm = OptunaTpeAlgorithm()
    algorithm.setup(task_spec, seed=17)

    history: list[TrialObservation] = []
    for trial_id in range(task_spec.max_evaluations):
        suggestion = algorithm.ask()
        observation = _make_observation(suggestion, trial_id)
        algorithm.tell(observation)
        history.append(observation)

    replayed = OptunaTpeAlgorithm()
    replayed.setup(task_spec, seed=17)
    replayed.replay(history[:-1])
    next_suggestion = replayed.ask()

    assert next_suggestion.config == history[-1].suggestion.config
    assert next_suggestion.metadata["optuna_trial_number"] == history[-1].suggestion.metadata["optuna_trial_number"]

    replayed.tell(history[-1])
    assert replayed.incumbents() == algorithm.incumbents()


def test_optuna_tpe_branin_summary_and_resume_outputs(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="optuna_tpe",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=False,
    )

    results_path = Path(summary["results_jsonl"])
    summary_path = results_path.with_name("summary.json")
    assert summary["trial_count"] == 6
    assert summary["best_primary_objective"] is not None
    assert results_path.exists()
    assert summary_path.exists()
    assert len(summary["incumbents"]) >= 1
    for plot_path in summary["plot_paths"]:
        assert Path(plot_path).exists()

    resumed = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="optuna_tpe",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=True,
    )
    assert resumed["trial_count"] == 6
    assert resumed["best_primary_objective"] == summary["best_primary_objective"]
