from __future__ import annotations

import math
import os

import pytest

from bbo.algorithms import (
    BotorchTurboAlgorithm,
    GpEiAlgorithm,
    LlamboAlgorithm,
    OptunaTpeAlgorithm,
    OproAlgorithm,
    RandomSearchAlgorithm,
    SobolSearchAlgorithm,
)
from bbo.core import EvaluationResult, TrialObservation, TrialSuggestion
from bbo.tasks import HPO_TASK_IDS, TASK_FAMILIES, TASK_REGISTRY, create_task
from bbo.tasks.hpo import (
    DATASETS,
    MODELS,
    PAPER_RESULT_SEEDS,
    preload_sklearn_runtime,
    published_initial_configurations,
)


EXPECTED_DIMENSIONS = {
    "random_forest": 6,
    "svm": 3,
    "decision_tree": 6,
    "mlp_sgd": 8,
    "adaboost": 2,
}


def test_hpo_registry_contains_exactly_25_public_bayesmark_pairs() -> None:
    assert len(HPO_TASK_IDS) == 25
    assert set(TASK_FAMILIES["hpo"]) == set(HPO_TASK_IDS)
    assert {name for name, family in TASK_REGISTRY.items() if family == "hpo"} == set(HPO_TASK_IDS)
    assert len(DATASETS) == 5
    assert len(MODELS) == 5


def test_sklearn_runtime_preload_is_idempotent() -> None:
    preload_sklearn_runtime()
    preload_sklearn_runtime()


def test_paper_search_spaces_and_transforms_are_exact() -> None:
    assert {key: model.dimension for key, model in MODELS.items()} == EXPECTED_DIMENSIONS
    assert MODELS["svm"].transforms == {"C": "log", "gamma": "log", "tol": "log"}
    assert MODELS["mlp_sgd"].search_space.names() == [
        "hidden_layer_sizes",
        "alpha",
        "batch_size",
        "learning_rate_init",
        "power_t",
        "tol",
        "momentum",
        "validation_fraction",
    ]
    assert MODELS["mlp_sgd"].transforms["tol"] == "log"
    assert MODELS["mlp_sgd"].transforms["validation_fraction"] == "logit"


def test_all_hpo_tasks_have_valid_assets_and_v3_descriptions() -> None:
    for name in HPO_TASK_IDS:
        task = create_task(name, max_evaluations=1, seed=7)
        report = task.sanity_check()
        assert report.ok, (name, report.errors)
        assert set(report.metadata["description_sections"]) >= {
            "background",
            "goal",
            "constraints",
            "prior_knowledge",
            "evaluation",
            "environment",
        }
        assert task.spec.metadata["task_family"] == "hpo"
        assert task.spec.metadata["dimension"] == len(task.spec.search_space)
        assert task.spec.max_evaluations == 1


def test_published_fixed_initialization_is_model_seed_shared_and_exact() -> None:
    breast = create_task("hpo_bayesmark_breast_svm", max_evaluations=30, seed=0)
    wine = create_task("hpo_bayesmark_wine_svm", max_evaluations=30, seed=0)
    expected = published_initial_configurations(MODELS["svm"], seed=0)
    breast_configs = breast.spec.metadata["benchmark_protocol"]["initialization"]["configurations"]
    wine_configs = wine.spec.metadata["benchmark_protocol"]["initialization"]["configurations"]

    assert PAPER_RESULT_SEEDS == (0, 1, 2, 3, 4)
    assert len(breast_configs) == 5
    assert breast_configs == wine_configs == [dict(config) for config in expected]
    assert breast_configs[0] == {
        "C": pytest.approx(44.30375245218264),
        "gamma": pytest.approx(0.0005190263017695167),
        "tol": pytest.approx(0.0025766385746135885),
    }


def test_published_mlp_initialization_materializes_effective_sklearn_defaults() -> None:
    configs = published_initial_configurations(MODELS["mlp_sgd"], seed=0)

    assert len(configs) == 5
    assert all(config["tol"] == pytest.approx(1e-4) for config in configs)
    assert all(config["validation_fraction"] == pytest.approx(0.1) for config in configs)
    assert all(set(config) == set(MODELS["mlp_sgd"].search_space.names()) for config in configs)


def test_hpo_rejects_seed_without_a_published_fixed_initialization() -> None:
    with pytest.raises(ValueError, match="seeds 0..9"):
        create_task("hpo_bayesmark_breast_svm", max_evaluations=30, seed=10)


def test_hpo_baselines_share_five_initial_points_and_model_candidate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("optuna")
    task = create_task("hpo_bayesmark_breast_svm", max_evaluations=30, seed=0)
    algorithms = [
        GpEiAlgorithm(pool_size=17, startup_trials=2),
        BotorchTurboAlgorithm(n_candidates=19, startup_trials=5),
        SobolSearchAlgorithm(),
        LlamboAlgorithm(backend="heuristic", n_initial_samples=2),
        OproAlgorithm(backend="heuristic", n_initial_samples=2),
        OptunaTpeAlgorithm(),
        RandomSearchAlgorithm(),
    ]
    for algorithm in algorithms:
        algorithm.setup(task.spec, seed=0, task_description=task.get_description())

    expected = task.spec.metadata["benchmark_protocol"]["initialization"]["configurations"]
    for index, expected_config in enumerate(expected):
        suggestions = [algorithm.ask() for algorithm in algorithms]
        assert all(suggestion.config == expected_config for suggestion in suggestions)
        assert all(suggestion.metadata["initialization_index"] == index for suggestion in suggestions)
        for algorithm, suggestion in zip(algorithms, suggestions, strict=True):
            observation = TrialObservation.from_evaluation(
                TrialSuggestion(
                    config=dict(suggestion.config),
                    trial_id=index,
                    metadata=dict(suggestion.metadata),
                ),
                EvaluationResult(objectives={"accuracy": 0.5 + index * 0.01}),
            )
            algorithm.tell(observation)

    gp_ei = algorithms[0]
    turbo = algorithms[1]
    assert isinstance(gp_ei, GpEiAlgorithm)
    assert isinstance(turbo, BotorchTurboAlgorithm)
    assert gp_ei.candidate_budget == turbo.candidate_budget == 2048
    assert gp_ei.candidate_budget_policy == turbo.candidate_budget_policy

    acquisition_config = task.spec.search_space.defaults()
    monkeypatch.setattr(
        gp_ei,
        "_botorch_ei_suggestion",
        lambda successful: TrialSuggestion(
            config=dict(acquisition_config),
            metadata={"gp_ei_phase": "acquisition", "gp_ei_training_points": len(successful)},
        ),
    )
    monkeypatch.setattr(
        turbo,
        "_turbo_suggestion",
        lambda successful: TrialSuggestion(
            config=dict(acquisition_config),
            metadata={"turbo_phase": "acquisition", "turbo_training_points": len(successful)},
        ),
    )
    assert gp_ei.ask().metadata["gp_ei_phase"] == "acquisition"
    assert turbo.ask().metadata["turbo_phase"] == "acquisition"
    assert algorithms[2].ask().metadata["sobol_draw_index"] == 0


@pytest.mark.parametrize(
    "task_name,objective",
    [
        ("hpo_bayesmark_breast_random_forest", "accuracy"),
        ("hpo_bayesmark_diabetes_decision_tree", "mse"),
    ],
)
def test_representative_real_hpo_evaluations_are_finite(task_name: str, objective: str) -> None:
    pytest.importorskip("sklearn")
    task = create_task(task_name, max_evaluations=1, seed=0)
    result = task.evaluate(TrialSuggestion(task.spec.search_space.defaults()))
    assert result.success, (result.error_type, result.error_message)
    assert math.isfinite(result.objectives[objective])
    assert result.metrics["cv_splits"] == 5
    assert result.metrics["train_samples"] > result.metrics["test_samples"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("BBO_RUN_HPO_ACCEPTANCE") != "1",
    reason="Set BBO_RUN_HPO_ACCEPTANCE=1 to run all 25 real sklearn evaluations.",
)
def test_all_25_default_configs_evaluate_successfully() -> None:
    pytest.importorskip("sklearn")
    for name in HPO_TASK_IDS:
        task = create_task(name, max_evaluations=1, seed=0)
        result = task.evaluate(TrialSuggestion(task.spec.search_space.defaults()))
        assert result.success, (name, result.error_type, result.error_message)
        assert all(math.isfinite(value) for value in result.objectives.values())
