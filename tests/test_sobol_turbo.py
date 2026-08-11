from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from bbo.algorithms import (  # noqa: E402
    ALGORITHM_REGISTRY,
    BotorchTurboAlgorithm,
    SobolSearchAlgorithm,
)
from bbo.algorithms.model_based._vendor import botorch_turbo as vendor  # noqa: E402
from bbo.core import (  # noqa: E402
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
from bbo.run import build_arg_parser  # noqa: E402


def _task_spec(direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE) -> TaskSpec:
    return TaskSpec(
        name="transformed_numeric_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e1, log=True, default=1e-2),
                FloatParam("fraction", low=0.01, high=0.99, default=0.5),
                IntParam("depth", low=1, high=8, default=4),
            ]
        ),
        objectives=(ObjectiveSpec("loss", direction),),
        max_evaluations=12,
        metadata={
            "parameter_transforms": {"lr": "log", "fraction": "logit", "depth": "linear"}
        },
    )


def _observation(suggestion: TrialSuggestion, trial_id: int, direction: ObjectiveDirection) -> TrialObservation:
    config = suggestion.config
    raw = (float(config["lr"]) - 0.1) ** 2 + (float(config["fraction"]) - 0.35) ** 2
    raw += 0.01 * (int(config["depth"]) - 3) ** 2
    value = -raw if direction == ObjectiveDirection.MAXIMIZE else raw
    suggestion.trial_id = trial_id
    return TrialObservation.from_evaluation(suggestion, EvaluationResult(objectives={"loss": value}))


def test_new_baselines_are_registered_and_cli_visible() -> None:
    assert ALGORITHM_REGISTRY["sobol_search"].factory is SobolSearchAlgorithm
    assert ALGORITHM_REGISTRY["botorch_turbo"].factory is BotorchTurboAlgorithm
    choices = next(action.choices for action in build_arg_parser()._actions if action.dest == "algorithm")
    assert {"sobol_search", "sobol", "botorch_turbo", "turbo"} <= set(choices)


def test_sobol_is_seeded_deterministic_and_replayable() -> None:
    spec = _task_spec()
    first = SobolSearchAlgorithm()
    second = SobolSearchAlgorithm()
    first.setup(spec, seed=13)
    second.setup(spec, seed=13)

    history = []
    for trial_id in range(6):
        left = first.ask()
        right = second.ask()
        assert left.config == right.config
        spec.search_space.validate_config(left.config)
        observation = _observation(left, trial_id, ObjectiveDirection.MINIMIZE)
        first.tell(observation)
        second.tell(_observation(right, trial_id, ObjectiveDirection.MINIMIZE))
        history.append(observation)

    resumed = SobolSearchAlgorithm()
    resumed.setup(spec, seed=13)
    resumed.replay(history)
    assert resumed.ask().config == first.ask().config


def test_turbo_uses_same_sobol_startup_then_official_acquisition() -> None:
    spec = _task_spec()
    turbo = BotorchTurboAlgorithm(startup_trials=5, n_candidates=64)
    sobol = SobolSearchAlgorithm()
    turbo.setup(spec, seed=5)
    sobol.setup(spec, seed=5)

    for trial_id in range(5):
        turbo_suggestion = turbo.ask()
        sobol_suggestion = sobol.ask()
        assert turbo_suggestion.config == sobol_suggestion.config
        assert turbo_suggestion.metadata["turbo_phase"] == "startup"
        turbo.tell(_observation(turbo_suggestion, trial_id, ObjectiveDirection.MINIMIZE))
        sobol.tell(_observation(sobol_suggestion, trial_id, ObjectiveDirection.MINIMIZE))

    acquired = turbo.ask()
    spec.search_space.validate_config(acquired.config)
    assert acquired.metadata["turbo_phase"] == "acquisition"
    assert acquired.metadata["turbo_backend"] == "botorch_official_tutorial"
    assert acquired.metadata["turbo_source_tag"] == "v0.18.1"
    assert "turbo_fallback_reason" not in acquired.metadata


def test_turbo_vendor_provenance_is_pinned() -> None:
    assert vendor.SOURCE_TAG == "v0.18.1"
    assert vendor.SOURCE_COMMIT == "cd0249c60b2a81af0d91b5e7d462ef6f574fceec"
    assert vendor.SOURCE_NOTEBOOK_BLOB_SHA == "5d1239c15353e6764bc7ca4955b61c2e0600f472"


def test_gp_ei_and_turbo_default_to_the_same_dimension_scaled_candidate_budget() -> None:
    from bbo.algorithms import GpEiAlgorithm

    spec = _task_spec()
    gp_ei = GpEiAlgorithm()
    turbo = BotorchTurboAlgorithm()
    gp_ei.setup(spec, seed=3)
    turbo.setup(spec, seed=3)

    assert gp_ei.candidate_budget == turbo.candidate_budget == 2048
    assert gp_ei.candidate_budget_policy == turbo.candidate_budget_policy
