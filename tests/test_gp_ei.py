from __future__ import annotations

import pytest

pytest.importorskip("botorch")

from bbo.algorithms import ALGORITHM_REGISTRY, GpEiAlgorithm
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
from bbo.run import build_arg_parser


def _mixed_task_spec() -> TaskSpec:
    return TaskSpec(
        name="mixed_gp_ei_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=0.0, high=1.0, default=0.5),
                IntParam("depth", low=1, high=4, default=2),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=16,
    )


def _observation(config: dict, trial_id: int) -> TrialObservation:
    loss = float(config["lr"]) + 0.1 * float(config["depth"]) + {"relu": 0.2, "gelu": 0.0, "tanh": 0.4}[config["activation"]]
    return TrialObservation.from_evaluation(
        TrialSuggestion(config=dict(config), trial_id=trial_id),
        EvaluationResult(objectives={"loss": loss}),
    )


def test_gp_ei_is_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "gp_ei" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["gp_ei"].family == "model_based"
    assert ALGORITHM_REGISTRY["gp_ei"].categorical_to_continuous == "onehot"
    assert "gp_ei" in algorithm_action.choices


def test_gp_ei_uses_onehot_standardized_acquisition_on_mixed_space() -> None:
    spec = _mixed_task_spec()
    algorithm = GpEiAlgorithm(pool_size=64, startup_trials=2)
    algorithm.setup(spec, seed=11)

    initial_configs = [
        {"lr": 0.1, "depth": 1, "activation": "relu"},
        {"lr": 0.2, "depth": 2, "activation": "gelu"},
        {"lr": 0.8, "depth": 4, "activation": "tanh"},
    ]
    for trial_id, config in enumerate(initial_configs):
        algorithm.tell(_observation(config, trial_id))

    suggestion = algorithm.ask()

    spec.search_space.validate_config(suggestion.config)
    assert suggestion.config not in initial_configs
    assert suggestion.metadata["gp_ei_phase"] == "acquisition"
    assert suggestion.metadata["gp_ei_backend"] == "botorch"
    assert suggestion.metadata["gp_ei_model"] == "SingleTaskGP"
    assert suggestion.metadata["gp_ei_acquisition"] == "ExpectedImprovement"
    assert suggestion.metadata["gp_ei_acquisition_optimizer"] == "optimize_acqf"
    assert suggestion.metadata["gp_ei_feature_encoder"] == "onehot"
    assert suggestion.metadata["gp_ei_feature_standardization"] == "train_mean_std"


def test_gp_ei_diagnostics_degrade_before_surrogate_fit() -> None:
    algorithm = GpEiAlgorithm(pool_size=16)
    algorithm.setup(_mixed_task_spec(), seed=7)
    algorithm.tell(_observation({"lr": 0.2, "depth": 2, "activation": "gelu"}, 0))

    diagnostics = algorithm.diagnose()

    assert diagnostics["fit_status"] == "insufficient_data"
    assert diagnostics["data_sufficient"] is False
    assert diagnostics["cv_r2"] is None
    assert diagnostics["training_points"] == 1


def test_gp_ei_diagnostics_report_cv_fit_and_sensitivity() -> None:
    algorithm = GpEiAlgorithm(pool_size=16)
    algorithm.setup(_mixed_task_spec(), seed=7)
    configs = [
        {"lr": 0.05, "depth": 1, "activation": "relu"},
        {"lr": 0.2, "depth": 2, "activation": "gelu"},
        {"lr": 0.4, "depth": 3, "activation": "tanh"},
        {"lr": 0.7, "depth": 4, "activation": "relu"},
        {"lr": 0.9, "depth": 2, "activation": "tanh"},
    ]
    for trial_id, config in enumerate(configs):
        algorithm.tell(_observation(config, trial_id))

    diagnostics = algorithm.diagnose(max_cv_folds=3)

    assert diagnostics["fit_status"] == "ok"
    assert diagnostics["data_sufficient"] is True
    assert diagnostics["cv_folds"] == 3
    assert isinstance(diagnostics["cv_r2"], float)
    assert diagnostics["hyperparameters"]
    assert set(diagnostics["feature_sensitivities"]) == set(diagnostics["feature_names"])
