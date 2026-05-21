"""Scientific-task registry for BO tutorial benchmark tasks."""

from __future__ import annotations

from typing import Callable

from ...core import Task
from .bh import BH_TASK_NAME, create_bh_task
from .guacamol import GUACAMOL_QED_TASK_NAME, create_guacamol_qed_task
from .guacamol_selfies import (
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME,
    GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME,
    GUACAMOL_FEXOFENADINE_MPO_TASK_NAME,
    GUACAMOL_MEDIAN1_TASK_NAME,
    GUACAMOL_QED_SELFIES_TASK_NAME,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME,
    create_guacamol_aripiprazole_similarity_task,
    create_guacamol_celecoxib_rediscovery_task,
    create_guacamol_fexofenadine_mpo_task,
    create_guacamol_median1_task,
    create_guacamol_qed_selfies_task,
    create_guacamol_troglitazone_rediscovery_task,
)
from .guacamol_smiles import (
    GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME,
    GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME,
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME,
    GUACAMOL_MEDIAN2_SMILES_TASK_NAME,
    GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME,
    GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME,
    GUACAMOL_QED_SMILES_TASK_NAME,
    GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME,
    GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME,
    GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME,
    GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME,
    create_guacamol_amlodipine_mpo_smiles_task,
    create_guacamol_aripiprazole_similarity_smiles_task,
    create_guacamol_celecoxib_rediscovery_smiles_task,
    create_guacamol_fexofenadine_mpo_smiles_task,
    create_guacamol_median1_smiles_task,
    create_guacamol_median2_smiles_task,
    create_guacamol_osimertinib_mpo_smiles_task,
    create_guacamol_perindopril_mpo_smiles_task,
    create_guacamol_qed_smiles_task,
    create_guacamol_ranolazine_mpo_smiles_task,
    create_guacamol_sitagliptin_mpo_smiles_task,
    create_guacamol_troglitazone_rediscovery_smiles_task,
    create_guacamol_valsartan_smarts_smiles_task,
    create_guacamol_zaleplon_mpo_smiles_task,
)
from .hea import HEA_TASK_NAME, create_hea_task
from .her import HER_TASK_NAME, create_her_task
from .molecule import MOLECULE_TASK_NAME, create_molecule_qed_task
from .molecule_similarity import MOLECULE_SIMILARITY_TASK_NAME, create_molecule_similarity_task
from .oer import OER_TASK_NAME, create_oer_task
from .qed_selfies import QED_SELFIES_TASK_NAME, create_qed_selfies_task

ScientificTaskFactory = Callable[..., Task]

SCIENTIFIC_TASK_REGISTRY: dict[str, str] = {
    HER_TASK_NAME: "HER random-forest demo task",
    HEA_TASK_NAME: "HEA random-forest demo task",
    OER_TASK_NAME: "OER mixed-feature random-forest demo task",
    BH_TASK_NAME: "BH feature-selected random-forest demo task",
    GUACAMOL_QED_TASK_NAME: "GuacaMol-inspired QED categorical demo task",
    GUACAMOL_QED_SELFIES_TASK_NAME: "GuacaMol QED SELFIES token demo task",
    GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME: "GuacaMol Celecoxib rediscovery SELFIES token demo task",
    GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME: "GuacaMol Troglitazone rediscovery SELFIES token demo task",
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME: "GuacaMol Aripiprazole similarity SELFIES token demo task",
    GUACAMOL_FEXOFENADINE_MPO_TASK_NAME: "GuacaMol Fexofenadine MPO SELFIES token demo task",
    GUACAMOL_MEDIAN1_TASK_NAME: "GuacaMol Median Molecules 1 SELFIES token demo task",
    GUACAMOL_QED_SMILES_TASK_NAME: "GuacaMol QED direct-SMILES demo task",
    GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME: "GuacaMol Celecoxib rediscovery direct-SMILES demo task",
    GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME: "GuacaMol Troglitazone rediscovery direct-SMILES demo task",
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME: "GuacaMol Aripiprazole similarity direct-SMILES demo task",
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME: "GuacaMol Fexofenadine MPO direct-SMILES demo task",
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME: "GuacaMol Median Molecules 1 direct-SMILES demo task",
    GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME: "GuacaMol Amlodipine MPO direct-SMILES demo task",
    GUACAMOL_MEDIAN2_SMILES_TASK_NAME: "GuacaMol Median Molecules 2 direct-SMILES demo task",
    GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME: "GuacaMol Osimertinib MPO direct-SMILES demo task",
    GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME: "GuacaMol Perindopril MPO direct-SMILES demo task",
    GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME: "GuacaMol Ranolazine MPO direct-SMILES demo task",
    GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME: "GuacaMol Sitagliptin MPO direct-SMILES demo task",
    GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME: "GuacaMol Valsartan SMARTS direct-SMILES demo task",
    GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME: "GuacaMol Zaleplon MPO direct-SMILES demo task",
    MOLECULE_TASK_NAME: "Molecule QED categorical demo task",
    QED_SELFIES_TASK_NAME: "QED generative SELFIES token demo task",
    MOLECULE_SIMILARITY_TASK_NAME: "Molecule ECFP4/Tanimoto similarity SELFIES token demo task",
}

SCIENTIFIC_TASK_FACTORIES: dict[str, ScientificTaskFactory] = {
    HER_TASK_NAME: create_her_task,
    HEA_TASK_NAME: create_hea_task,
    OER_TASK_NAME: create_oer_task,
    BH_TASK_NAME: create_bh_task,
    GUACAMOL_QED_TASK_NAME: create_guacamol_qed_task,
    GUACAMOL_QED_SELFIES_TASK_NAME: create_guacamol_qed_selfies_task,
    GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME: create_guacamol_celecoxib_rediscovery_task,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME: create_guacamol_troglitazone_rediscovery_task,
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME: create_guacamol_aripiprazole_similarity_task,
    GUACAMOL_FEXOFENADINE_MPO_TASK_NAME: create_guacamol_fexofenadine_mpo_task,
    GUACAMOL_MEDIAN1_TASK_NAME: create_guacamol_median1_task,
    GUACAMOL_QED_SMILES_TASK_NAME: create_guacamol_qed_smiles_task,
    GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME: create_guacamol_celecoxib_rediscovery_smiles_task,
    GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME: create_guacamol_troglitazone_rediscovery_smiles_task,
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME: create_guacamol_aripiprazole_similarity_smiles_task,
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME: create_guacamol_fexofenadine_mpo_smiles_task,
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME: create_guacamol_median1_smiles_task,
    GUACAMOL_AMLODIPINE_MPO_SMILES_TASK_NAME: create_guacamol_amlodipine_mpo_smiles_task,
    GUACAMOL_MEDIAN2_SMILES_TASK_NAME: create_guacamol_median2_smiles_task,
    GUACAMOL_OSIMERTINIB_MPO_SMILES_TASK_NAME: create_guacamol_osimertinib_mpo_smiles_task,
    GUACAMOL_PERINDOPRIL_MPO_SMILES_TASK_NAME: create_guacamol_perindopril_mpo_smiles_task,
    GUACAMOL_RANOLAZINE_MPO_SMILES_TASK_NAME: create_guacamol_ranolazine_mpo_smiles_task,
    GUACAMOL_SITAGLIPTIN_MPO_SMILES_TASK_NAME: create_guacamol_sitagliptin_mpo_smiles_task,
    GUACAMOL_VALSARTAN_SMARTS_SMILES_TASK_NAME: create_guacamol_valsartan_smarts_smiles_task,
    GUACAMOL_ZALEPLON_MPO_SMILES_TASK_NAME: create_guacamol_zaleplon_mpo_smiles_task,
    MOLECULE_TASK_NAME: create_molecule_qed_task,
    QED_SELFIES_TASK_NAME: create_qed_selfies_task,
    MOLECULE_SIMILARITY_TASK_NAME: create_molecule_similarity_task,
}


def create_scientific_task(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    **kwargs,
) -> Task:
    if name not in SCIENTIFIC_TASK_FACTORIES:
        available = ", ".join(sorted(SCIENTIFIC_TASK_FACTORIES))
        raise ValueError(f"Unknown scientific task `{name}`. Available: {available}")
    return SCIENTIFIC_TASK_FACTORIES[name](
        max_evaluations=max_evaluations,
        seed=seed,
        **kwargs,
    )


__all__ = [
    "SCIENTIFIC_TASK_FACTORIES",
    "SCIENTIFIC_TASK_REGISTRY",
    "ScientificTaskFactory",
    "create_scientific_task",
]
