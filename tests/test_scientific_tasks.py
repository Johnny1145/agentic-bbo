from __future__ import annotations

import os
from pathlib import Path

import pytest

from bbo.core import ObjectiveDirection, StringParam, TrialSuggestion
from bbo.run import run_single_experiment
from bbo.tasks import (
    ALL_TASK_NAMES,
    BH_TASK_NAME,
    GUACAMOL_QED_TASK_NAME,
    GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME,
    GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME,
    GUACAMOL_FEXOFENADINE_MPO_TASK_NAME,
    GUACAMOL_MEDIAN1_TASK_NAME,
    GUACAMOL_MEDIAN2_SMILES_TASK_NAME,
    GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME,
    GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME,
    GUACAMOL_QED_SELFIES_TASK_NAME,
    GUACAMOL_QED_SMILES_TASK_NAME,
    GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_SELFIES_TASK_NAMES,
    GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME,
    GUACAMOL_SMILES_TASK_NAMES,
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME,
    GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME,
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME,
    GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME,
    GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME,
    HEA_TASK_NAME,
    HER_FEATURES,
    HER_TASK_NAME,
    MOLECULE_SIMILARITY_TASK_NAME,
    MOLECULE_TASK_NAME,
    OER_TASK_NAME,
    QED_SELFIES_TASK_NAME,
    create_bh_task,
    create_guacamol_selfies_task,
    create_guacamol_smiles_task,
    create_guacamol_qed_task,
    create_hea_task,
    create_her_task,
    create_molecule_similarity_task,
    create_molecule_qed_task,
    create_oer_task,
    create_qed_selfies_task,
)
from bbo.tasks.scientific import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, VENDORED_SOURCE_ROOT
from bbo.tasks.scientific.guacamol_selfies import CELECOXIB_SMILES
from bbo.tasks.scientific.guacamol_smiles import (
    AMLODIPINE_SMILES,
    OSIMERTINIB_SMILES,
    PERINDOPRIL_SMILES,
    RANOLAZINE_SMILES,
    SITAGLIPTIN_SMILES,
    TADALAFIL_SMILES,
    VALSARTAN_SMARTS,
    ZALEPLON_SMILES,
)


def _require_bo_tutorial_source() -> Path:
    source_root = Path(os.environ.get(SOURCE_ROOT_ENV, str(VENDORED_SOURCE_ROOT)))
    if not source_root.exists():
        pytest.skip("Bundled scientific task datasets are not available in the workspace.")
    return source_root


@pytest.fixture
def scientific_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    source_root = _require_bo_tutorial_source()
    monkeypatch.setenv(SOURCE_ROOT_ENV, str(source_root))
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "dataset_cache"))
    return source_root


def test_scientific_registry_contains_all_tasks() -> None:
    assert HER_TASK_NAME in ALL_TASK_NAMES
    assert HEA_TASK_NAME in ALL_TASK_NAMES
    assert OER_TASK_NAME in ALL_TASK_NAMES
    assert BH_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_QED_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_QED_SELFIES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_FEXOFENADINE_MPO_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_MEDIAN1_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_QED_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_MEDIAN1_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_MEDIAN2_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME in ALL_TASK_NAMES
    assert MOLECULE_TASK_NAME in ALL_TASK_NAMES
    assert QED_SELFIES_TASK_NAME in ALL_TASK_NAMES
    assert MOLECULE_SIMILARITY_TASK_NAME in ALL_TASK_NAMES


def test_her_task_spec_and_sanity(scientific_env: Path) -> None:
    task = create_her_task(max_evaluations=3, seed=19, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == HER_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert task.spec.primary_objective.direction == ObjectiveDirection.MINIMIZE
    assert task.spec.search_space.names() == list(HER_FEATURES)
    assert report.metadata["row_count"] == 812
    assert report.metadata["column_count"] == 11

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert "predicted_target" in result.metrics


def test_hea_task_spec_and_transform(scientific_env: Path) -> None:
    pytest.importorskip("openpyxl")
    task = create_hea_task(max_evaluations=3, seed=13, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == HEA_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert report.metadata["transform_residual_max"] < 1e-9

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert all(key.startswith("composition::") for key in result.metrics if key.startswith("composition::"))


def test_oer_task_spec_and_sanity(scientific_env: Path) -> None:
    task = create_oer_task(max_evaluations=3, seed=11, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == OER_TASK_NAME
    assert task.spec.primary_objective.name == "overpotential_mv"
    assert "Metal_1" in report.metadata["categorical_choices"]

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["overpotential_mv"] > 0.0


def test_bh_task_feature_selection_and_sanity(scientific_env: Path) -> None:
    task = create_bh_task(max_evaluations=3, seed=7, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == BH_TASK_NAME
    assert task.spec.primary_objective.name == "regret"
    assert report.metadata["selected_features"]

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.objectives["regret"] >= 0.0
    assert "predicted_yield" in result.metrics


def test_molecule_qed_task_sanity(scientific_env: Path) -> None:
    pytest.importorskip("rdkit")
    task = create_molecule_qed_task(max_evaluations=3, seed=5, source_root=scientific_env)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == MOLECULE_TASK_NAME
    assert task.spec.primary_objective.name == "qed_loss"
    assert report.metadata["item_count"] > 0

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert 0.0 <= result.objectives["qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["qed"] <= 1.0


def test_molecule_qed_task_smiles_limit(scientific_env: Path) -> None:
    pytest.importorskip("rdkit")
    task = create_molecule_qed_task(
        max_evaluations=3,
        seed=5,
        source_root=scientific_env,
        smiles_limit=64,
    )

    assert len(task.spec.search_space["SMILES"].choices) == 64
    assert task.dataset_summary["smiles_limit"] == 64


def test_qed_selfies_task_sanity(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    task = create_qed_selfies_task(
        max_evaluations=3,
        seed=5,
        source_root=source_root,
        cache_root=tmp_path / "dataset_cache",
        max_selfies_tokens=8,
        vocabulary_source_limit=64,
    )
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == QED_SELFIES_TASK_NAME
    assert task.spec.primary_objective.name == "qed_loss"
    assert report.metadata["selfies_vocabulary_size"] > 0
    assert task.spec.search_space.names()[0] == "selfies_token_00"

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.metadata["valid_smiles"]
    assert 0.0 <= result.objectives["qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["qed"] <= 1.0

    ethanol = task.evaluate(TrialSuggestion(config=task.config_from_smiles("CCO")))
    assert ethanol.success
    assert ethanol.metadata["valid_smiles"]
    assert ethanol.metadata["smiles"]


def test_molecule_similarity_task_sanity(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    task = create_molecule_similarity_task(
        max_evaluations=3,
        seed=17,
        source_root=source_root,
        cache_root=tmp_path / "dataset_cache",
        max_selfies_tokens=8,
        vocabulary_source_limit=64,
    )
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == MOLECULE_SIMILARITY_TASK_NAME
    assert task.spec.primary_objective.name == "similarity_loss"
    assert report.metadata["source_item_count"] > 0
    assert report.metadata["selfies_vocabulary_size"] > 0
    assert task.spec.search_space.names()[0] == "selfies_token_00"
    assert report.metadata["target_name"] == "Celecoxib"

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.metadata["valid_smiles"]
    assert 0.0 <= result.objectives["similarity_loss"] <= 1.0
    assert 0.0 <= result.metrics["tanimoto_similarity"] <= 1.0

    ethanol = task.evaluate(TrialSuggestion(config=task.config_from_smiles("CCO")))
    assert ethanol.success
    assert ethanol.metadata["valid_smiles"]
    assert ethanol.metadata["smiles"]


def test_guacamol_qed_task_sanity() -> None:
    pytest.importorskip("rdkit")
    task = create_guacamol_qed_task(max_evaluations=3, seed=23)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == GUACAMOL_QED_TASK_NAME
    assert task.spec.primary_objective.name == "guacamol_qed_loss"
    assert report.metadata["candidate_pool_size"] > 0
    assert report.metadata["valid_candidate_count"] > 0

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert 0.0 <= result.objectives["guacamol_qed_loss"] <= 1.0
    assert 0.0 <= result.metrics["guacamol_qed_score"] <= 1.0


@pytest.mark.parametrize("task_name", GUACAMOL_SELFIES_TASK_NAMES)
def test_guacamol_selfies_task_sanity(task_name: str, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    task = create_guacamol_selfies_task(
        task_name,
        max_evaluations=3,
        seed=29,
        source_root=source_root,
        cache_root=tmp_path / "dataset_cache",
        max_selfies_tokens=16,
        vocabulary_source_limit=64,
    )
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == task_name
    assert task.spec.primary_objective.name.endswith("_loss")
    assert report.metadata["source_item_count"] > 0
    assert report.metadata["selfies_vocabulary_size"] > 0
    assert task.spec.search_space.names()[0] == "selfies_token_00"

    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert result.success
    assert result.metadata["valid_smiles"]
    objective = result.objectives[task.spec.primary_objective.name]
    assert 0.0 <= objective <= 1.0
    assert 0.0 <= result.metrics["guacamol_score"] <= 1.0


@pytest.mark.parametrize("task_name", GUACAMOL_SMILES_TASK_NAMES)
def test_guacamol_smiles_task_sanity(task_name: str) -> None:
    pytest.importorskip("rdkit")
    task = create_guacamol_smiles_task(task_name, max_evaluations=3, seed=31)
    report = task.sanity_check()

    assert report.ok
    assert task.spec.name == task_name
    assert task.spec.search_space.names() == ["smiles"]
    assert isinstance(task.spec.search_space["smiles"], StringParam)
    assert report.metadata["schema_default_role"] == "schema_only_not_initial_population"

    default_result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults()))
    assert default_result.success
    assert default_result.metadata["valid_smiles"] is False
    assert default_result.metrics["guacamol_score"] == 0.0
    assert default_result.objectives[task.spec.primary_objective.name] == 1.0

    valid_smiles = task.dataset_summary["target_smiles"][0] if task.dataset_summary["target_smiles"] else CELECOXIB_SMILES
    valid_result = task.evaluate(TrialSuggestion(config=task.config_from_smiles(valid_smiles)))
    assert valid_result.success
    assert valid_result.metadata["valid_smiles"] is True
    assert valid_result.metadata["canonical_smiles"]
    assert 0.0 <= valid_result.metrics["guacamol_score"] <= 1.0

    invalid_result = task.evaluate(TrialSuggestion(config={"smiles": "not a smiles"}))
    assert invalid_result.success
    assert invalid_result.metadata["valid_smiles"] is False
    assert invalid_result.metrics["guacamol_score"] == 0.0
    assert invalid_result.objectives[task.spec.primary_objective.name] == 1.0


def test_guacamol_smiles_rejects_generic_random_search(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    with pytest.raises(RuntimeError, match="failed during ask") as exc_info:
        run_single_experiment(
            task_name=GUACAMOL_QED_SMILES_TASK_NAME,
            algorithm_name="random_search",
            seed=5,
            max_evaluations=1,
            results_root=tmp_path,
            resume=False,
            generate_plots=False,
        )
    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "generic random sampler" in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    ("task_name", "smiles", "expected_score"),
    [
        (
            GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME,
            "Cc1ccc(C=Nc2ccc(S(N)(=O)=O)cc2)cc1",
            0.4588235294117647,
        ),
        (
            GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME,
            "Cc1c(C)c2c(c(C)c1O)CCC(C)(C(=O)NCOc1ccc(O)cc1)O2",
            0.5094339622641509,
        ),
        (
            GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME,
            "O=C1NCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2O1",
            0.9866666666666666,
        ),
        (
            GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME,
            "COC(=O)C1=CC(F)=CC=S1NC(=O)COCC(O)N1CCC(C(c2ccccc2)c2ccccc2)CC1",
            0.784002259753152,
        ),
        (
            GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME,
            AMLODIPINE_SMILES,
            0.3678794411714423,
        ),
        (
            GUACAMOL_MEDIAN2_SMILES_TASK_NAME,
            TADALAFIL_SMILES,
            0.3623715376697393,
        ),
        (
            GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME,
            OSIMERTINIB_SMILES,
            0.1333417190026552,
        ),
        (
            GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME,
            PERINDOPRIL_SMILES,
            0.01831563888873418,
        ),
        (
            GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME,
            RANOLAZINE_SMILES,
            0.04923735914329035,
        ),
        (
            GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME,
            SITAGLIPTIN_SMILES,
            3.726653172078671e-06,
        ),
        (
            GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME,
            VALSARTAN_SMARTS,
            1.555420075811193e-19,
        ),
        (
            GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME,
            ZALEPLON_SMILES,
            0.4664987211828484,
        ),
    ],
)
def test_guacamol_smiles_matches_local_pmo_reference_scores(
    task_name: str,
    smiles: str,
    expected_score: float,
) -> None:
    # Reference values are copied from local GuacaMol/PMO scorer checks.
    pytest.importorskip("rdkit")
    task = create_guacamol_smiles_task(task_name, max_evaluations=1)
    result = task.evaluate(TrialSuggestion(config={"smiles": smiles}))

    assert result.success
    assert result.metadata["valid_smiles"] is True
    assert result.metrics["guacamol_score"] == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "task_name",
    [HER_TASK_NAME, HEA_TASK_NAME, OER_TASK_NAME, BH_TASK_NAME],
)
def test_scientific_random_search_smoke(
    task_name: str,
    scientific_env: Path,
    tmp_path: Path,
) -> None:
    summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_molecule_random_search_smoke(scientific_env: Path, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    summary = run_single_experiment(
        task_name=MOLECULE_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_qed_selfies_optuna_smoke(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=QED_SELFIES_TASK_NAME,
        algorithm_name="optuna_tpe",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 8,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()


def test_qed_selfies_random_search_smoke(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=QED_SELFIES_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 8,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()


def test_molecule_similarity_random_search_smoke(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=MOLECULE_SIMILARITY_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 8,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()


def test_guacamol_qed_random_search_smoke(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    summary = run_single_experiment(
        task_name=GUACAMOL_QED_TASK_NAME,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        results_root=tmp_path,
        resume=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert len(summary["plot_paths"]) == 4
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.parametrize("task_name", GUACAMOL_SELFIES_TASK_NAMES)
def test_guacamol_selfies_random_search_smoke(task_name: str, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    source_root = _require_bo_tutorial_source()
    summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=5,
        max_evaluations=3,
        task_kwargs={
            "source_root": source_root,
            "cache_root": tmp_path / "dataset_cache",
            "max_selfies_tokens": 16,
            "vocabulary_source_limit": 64,
        },
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["trial_count"] == 3
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
