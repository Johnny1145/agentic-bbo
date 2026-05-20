from __future__ import annotations

import pytest

from bbo.algorithms.molecular.graph_ga_ops import GraphGACandidateOptimizer, morgan_fingerprint_array


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
