from __future__ import annotations

import pytest

from bbo.algorithms.molecular.gpbo import GraphGPBOAlgorithm
from bbo.algorithms.molecular.graph_ga import GraphGAAlgorithm
from bbo.algorithms.molecular.graph_ga_ops import GraphGACandidateOptimizer, morgan_fingerprint_array
from bbo.core import ObjectiveDirection, ObjectiveSpec, SearchSpace, StringParam, TaskSpec, TrialObservation, TrialStatus


def test_graph_ga_candidate_optimizer_runs_with_batch_scoring_callback() -> None:
    pytest.importorskip("rdkit")
    optimizer = GraphGACandidateOptimizer(
        max_generations=1,
        population_size=4,
        offspring_size=4,
        mutation_rate=0.01,
    )

    result = optimizer.maximize(
        starting_population_smiles=["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
        scoring_function=lambda smiles: [float(len(item)) for item in smiles],
        seed=0,
    )

    assert result.queried_smiles
    assert result.scores_by_smiles
    assert len(result.generation_info) == 1
    assert result.generation_info[0]["size"] > 0


def test_morgan_fingerprint_array_uses_explicit_bit_count() -> None:
    pytest.importorskip("rdkit")
    fp = morgan_fingerprint_array("CCO", radius=2, n_bits=128)

    assert fp.shape == (128,)


def test_graph_ga_algorithm_uses_initial_smiles_then_generates_offspring() -> None:
    pytest.importorskip("rdkit")
    spec = TaskSpec(
        name="smiles_demo",
        search_space=SearchSpace([StringParam("smiles", default="", max_length=128)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=8,
    )
    algorithm = GraphGAAlgorithm(
        initial_smiles=["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
        population_size=4,
        offspring_size=4,
        mutation_rate=0.01,
    )
    algorithm.setup(spec, seed=0)

    initial_suggestions = [algorithm.ask() for _ in range(4)]
    assert [item.metadata["graph_ga_phase"] for item in initial_suggestions] == 4 * ["initial_population"]
    for index, suggestion in enumerate(initial_suggestions):
        algorithm.tell(
            TrialObservation(
                suggestion=suggestion,
                status=TrialStatus.SUCCESS,
                objectives={"loss": float(4 - index)},
            )
        )

    offspring = algorithm.ask()
    assert offspring.metadata["graph_ga_phase"] == "offspring"
    assert isinstance(offspring.config["smiles"], str)
    assert offspring.config["smiles"] not in {item.config["smiles"] for item in initial_suggestions}


def test_gpbo_requires_explicit_initial_smiles() -> None:
    pytest.importorskip("rdkit")
    spec = TaskSpec(
        name="smiles_demo",
        search_space=SearchSpace([StringParam("smiles", default="", max_length=128)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=8,
    )
    algorithm = GraphGPBOAlgorithm(initial_population_size=4)

    with pytest.raises(ValueError, match="requires explicit initial_smiles"):
        algorithm.setup(spec, seed=0)


def test_gpbo_initial_population_ask_tell_without_gp_deps() -> None:
    pytest.importorskip("rdkit")
    spec = TaskSpec(
        name="smiles_demo",
        search_space=SearchSpace([StringParam("smiles", default="", max_length=128)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=8,
    )
    algorithm = GraphGPBOAlgorithm(
        initial_smiles=["CCO", "CCN"],
        initial_population_size=2,
        bo_batch_size=1,
        ga_max_generations=1,
        ga_population_size=4,
        ga_offspring_size=4,
    )
    algorithm.setup(spec, seed=0)

    suggestion = algorithm.ask()
    assert suggestion.metadata["gpbo_phase"] == "initial_population"
    algorithm.tell(
        TrialObservation(
            suggestion=suggestion,
            status=TrialStatus.SUCCESS,
            objectives={"loss": 0.5},
        )
    )
    assert algorithm.incumbents()
