from __future__ import annotations

import math

import pytest

from bbo.algorithms import BotorchTurboAlgorithm, GpEiAlgorithm, RandomSearchAlgorithm, SobolSearchAlgorithm
from bbo.algorithms.agentic.tools.core_tools import agent_visible_metadata, agent_visible_metrics
from bbo.core import TrialSuggestion
from bbo.tasks import (
    BBOB_DIMENSION,
    BBOB_FUNCTION_IDS,
    BBOB_INITIAL_DESIGN_SIZE,
    BBOB_INSTANCE_IDS,
    BBOB_MODEL_CANDIDATES,
    BBOB_OPTIMIZATION_BUDGET,
    BBOB_PROBLEM_REGISTRY,
    BBOB_TASK_IDS,
    BBOB_TOTAL_BUDGET,
    TASK_FAMILIES,
    create_task,
)
from bbo.tasks.registry import SYNTHETIC_PROBLEM_REGISTRY


OLD_SYNTHETIC_TASK_IDS = {
    "ackley_2d_demo",
    "beale_demo",
    "branin_demo",
    "budgeted_sphere_demo",
    "sphere_demo",
}


def test_only_official_bbob_tasks_remain_in_synthetic_registry() -> None:
    assert BBOB_FUNCTION_IDS == tuple(range(1, 25))
    assert len(BBOB_TASK_IDS) == 24
    assert tuple(BBOB_PROBLEM_REGISTRY) == BBOB_TASK_IDS
    assert tuple(SYNTHETIC_PROBLEM_REGISTRY) == BBOB_TASK_IDS
    assert TASK_FAMILIES["synthetic"] == BBOB_TASK_IDS
    assert OLD_SYNTHETIC_TASK_IDS.isdisjoint(SYNTHETIC_PROBLEM_REGISTRY)


def test_bbob_compact_protocol_is_10d_20_plus_100_and_2048_candidates() -> None:
    task = create_task("bbob_f01_d10", seed=2)
    protocol = task.spec.metadata["benchmark_protocol"]

    assert len(task.spec.search_space) == BBOB_DIMENSION == 10
    assert task.spec.max_evaluations == BBOB_TOTAL_BUDGET == 120
    assert BBOB_INITIAL_DESIGN_SIZE == 20
    assert BBOB_OPTIMIZATION_BUDGET == 100
    assert protocol["initialization"]["sampling"] == "scrambled_sobol"
    assert protocol["initialization"]["count"] == 20
    assert len(protocol["initialization"]["configurations"]) == 20
    assert protocol["candidate_budget"]["value"] == BBOB_MODEL_CANDIDATES == 2048


def test_bbob_instances_map_run_seeds_zero_one_two_to_instances_one_two_three() -> None:
    observed = tuple(create_task("bbob_f01_d10", seed=seed).instance_id for seed in range(3))
    assert observed == BBOB_INSTANCE_IDS == (1, 2, 3)
    assert create_task("bbob_f01_d10", seed=3).instance_id == 1


def test_same_seed_shares_identical_sobol_prefix_across_all_24_functions() -> None:
    prefixes = [
        create_task(task_name, seed=1).spec.metadata["benchmark_protocol"]["initialization"][
            "configurations"
        ]
        for task_name in BBOB_TASK_IDS
    ]

    assert all(prefix == prefixes[0] for prefix in prefixes[1:])
    assert len({repr(sorted(config.items())) for config in prefixes[0]}) == 20


@pytest.mark.parametrize(
    "algorithm",
    [RandomSearchAlgorithm(), SobolSearchAlgorithm(), GpEiAlgorithm(), BotorchTurboAlgorithm()],
    ids=lambda algorithm: algorithm.name,
)
def test_comparable_baselines_consume_the_same_first_bbob_configuration(algorithm) -> None:
    task = create_task("bbob_f07_d10", seed=1)
    expected = task.spec.metadata["benchmark_protocol"]["initialization"]["configurations"][0]

    algorithm.setup(task.spec, seed=1)
    suggestion = algorithm.ask()

    assert suggestion.config == expected
    assert suggestion.metadata["benchmark_initialization"] is True
    if isinstance(algorithm, (GpEiAlgorithm, BotorchTurboAlgorithm)):
        assert algorithm.candidate_budget == 2048


def test_bbob_sanity_and_descriptions_do_not_expose_hidden_reference_data() -> None:
    task = create_task("bbob_f24_d10", seed=2)
    report = task.sanity_check()
    assert report.ok, report.errors

    context = task.get_description().rendered_context.lower()
    for forbidden in ("f24", "function 24", "fopt", "xopt", "optimum location", "objective formula"):
        assert forbidden not in context

    visible = agent_visible_metadata(task.spec.metadata)
    assert "bbob_function_id" not in visible
    assert "bbob_instance_id" not in visible
    assert "bbob_problem_id" not in visible


def test_agent_visible_metrics_hide_bbob_reference_regret() -> None:
    visible = agent_visible_metrics(
        {
            "regret": 0.1,
            "log10_regret": -1.0,
            "distance_to_known_optimum": 0.2,
            "dimension": 10.0,
        }
    )
    assert visible == {"dimension": 10.0}


def test_all_24_functions_and_three_instances_evaluate_through_official_coco() -> None:
    for seed, instance_id in enumerate(BBOB_INSTANCE_IDS):
        for task_name in BBOB_TASK_IDS:
            task = create_task(task_name, seed=seed)
            try:
                result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
                expected_problem_id = (
                    f"bbob_f{task.function_id:03d}_i{instance_id:02d}_d{BBOB_DIMENSION}"
                )
                assert task.problem_id == expected_problem_id
                assert result.metadata["bbob_problem_id"] == expected_problem_id
                assert result.success
                assert math.isfinite(result.objectives["loss"])
                assert math.isfinite(result.metrics["regret"])
                assert result.metrics["regret"] >= 0.0
                assert task._problem.evaluations == 1
                assert result.objectives["loss"] - result.metrics["regret"] == pytest.approx(
                    task._f_opt
                )
            finally:
                task.cleanup()
