from __future__ import annotations
import importlib.util

import numpy as np
import pytest

from bbo.algorithms import ALGORITHM_REGISTRY, GitBoAlgorithm
from bbo.core import (
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.run import build_arg_parser

requires_tabpfn = pytest.mark.skipif(
    importlib.util.find_spec("tabpfn") is None,
    reason="git_bo execution requires the optional tabpfn extra",
)


def _observation(config: dict[str, float], loss: float, trial_id: int) -> TrialObservation:
    return TrialObservation.from_evaluation(
        TrialSuggestion(config=config, trial_id=trial_id),
        EvaluationResult(objectives={"loss": loss}),
    )


def _task_spec(*, metadata: dict | None = None) -> TaskSpec:
    return TaskSpec(
        name="git_bo_unit_demo",
        search_space=SearchSpace(
            [
                FloatParam("x1", low=-5.0, high=5.0, default=0.0),
                FloatParam("x2", low=-2.0, high=2.0, default=0.0),
                FloatParam("x3", low=0.1, high=10.0, default=1.0, log=True),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=20,
        metadata={"parameter_transforms": {"x1": "linear", "x2": "linear", "x3": "log"}, **(metadata or {})},
    )


def test_git_bo_is_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "git_bo" in ALGORITHM_REGISTRY
    assert "gitbo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["git_bo"].family == "model_based"
    assert ALGORITHM_REGISTRY["git_bo"].numeric_only is True
    assert "git_bo" in algorithm_action.choices


@requires_tabpfn
def test_git_bo_reuses_task_owned_fixed_initialization() -> None:
    config = {"x1": -1.0, "x2": 0.5, "x3": 2.0}
    protocol = {
        "benchmark_protocol": {
            "name": "unit_fixed_initialization",
            "initialization": {
                "strategy": "fixed_configurations",
                "sampling": "published",
                "source": "unit-test",
                "seed": 7,
                "count": 1,
                "configurations": [config],
            },
        }
    }
    algorithm = GitBoAlgorithm(device="cpu", n_candidates=16)
    algorithm.setup(_task_spec(metadata=protocol), seed=7)

    suggestion = algorithm.ask()

    assert suggestion.config == config
    assert suggestion.metadata["git_bo_phase"] == "benchmark_initialization"
    assert suggestion.metadata["initialization_protocol"] == "unit_fixed_initialization"


@requires_tabpfn
def test_git_bo_builds_fisher_subspace_and_ucb_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    algorithm = GitBoAlgorithm(
        device="cpu",
        n_candidates=64,
        inference_batch_size=16,
        subspace_dim=10,
        beta=2.33,
    )
    algorithm.setup(_task_spec(), seed=3)
    algorithm.tell(_observation({"x1": -4.0, "x2": -1.5, "x3": 0.2}, 8.0, 0))
    algorithm.tell(_observation({"x1": 0.0, "x2": 0.0, "x3": 1.0}, 2.0, 1))
    algorithm.tell(_observation({"x1": 2.0, "x2": 1.0, "x3": 4.0}, 4.0, 2))

    monkeypatch.setattr(
        algorithm,
        "_fit_tabpfn",
        lambda *, x_train, y_train, deps: {"x": x_train, "y": y_train},
    )

    def fake_posterior(*, model, queries, deps):
        del model
        torch = deps["torch"]
        center = torch.tensor([0.75, 0.25, 0.60], dtype=queries.dtype, device=queries.device)
        mean = -((queries - center) ** 2).sum(dim=-1)
        std = 0.05 + 0.02 * queries.mean(dim=-1)
        return mean, std

    monkeypatch.setattr(algorithm, "_tabpfn_posterior", fake_posterior)

    suggestion = algorithm.ask()

    _task_spec().search_space.validate_config(suggestion.config)
    assert suggestion.metadata["git_bo_phase"] == "acquisition"
    assert suggestion.metadata["git_bo_acquisition"] == "UCB"
    assert suggestion.metadata["git_bo_beta"] == pytest.approx(2.33)
    assert suggestion.metadata["git_bo_fisher_points"] == 64
    assert suggestion.metadata["git_bo_candidate_points"] == 64
    assert suggestion.metadata["git_bo_inference_batch_size"] == 16
    assert suggestion.metadata["git_bo_subspace_dim_requested"] == 10
    assert suggestion.metadata["git_bo_subspace_dim_effective"] == 3
    assert len(suggestion.metadata["git_bo_top_fisher_eigenvalues"]) == 3
    assert np.isfinite(suggestion.metadata["git_bo_selected_ucb"])


@requires_tabpfn
def test_git_bo_uses_sobol_for_degenerate_targets() -> None:
    algorithm = GitBoAlgorithm(device="cpu", n_candidates=16)
    algorithm.setup(_task_spec(), seed=2)
    algorithm.tell(_observation({"x1": -1.0, "x2": -1.0, "x3": 0.5}, 1.0, 0))
    algorithm.tell(_observation({"x1": 1.0, "x2": 1.0, "x3": 2.0}, 1.0, 1))

    suggestion = algorithm.ask()

    assert suggestion.metadata["git_bo_phase"] == "sobol_fallback"
    assert suggestion.metadata["git_bo_fallback_reason"] == "degenerate_target"
