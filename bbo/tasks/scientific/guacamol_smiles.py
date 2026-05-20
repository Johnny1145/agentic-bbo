"""GuacaMol goal-directed benchmarks over direct SMILES strings."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ...core import (
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    StringParam,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
from .guacamol import GUACAMOL_SOURCE_REPO_URL
from .guacamol_selfies import (
    ARIPIPRAZOLE_SMILES,
    CAMPHOR_SMILES,
    CELECOXIB_SMILES,
    FEXOFENADINE_SMILES,
    MENTHOL_SMILES,
    TROGLITAZONE_SMILES,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS = 40
GUACAMOL_SMILES_DEFAULT_MAX_LENGTH = 512
GUACAMOL_SMILES_SCHEMA_DEFAULT = ""
GUACAMOL_SMILES_SOURCE = "guacamol.goal_directed_suite"

GUACAMOL_QED_SMILES_TASK_NAME = "guacamol_qed_smiles_demo"
GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME = "guacamol_celecoxib_rediscovery_smiles_demo"
GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME = "guacamol_troglitazone_rediscovery_smiles_demo"
GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME = "guacamol_aripiprazole_similarity_smiles_demo"
GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME = "guacamol_fexofenadine_mpo_smiles_demo"
GUACAMOL_MEDIAN1_SMILES_TASK_NAME = "guacamol_median1_smiles_demo"


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The GuacaMol SMILES tasks require RDKit. Install with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors


@dataclass(frozen=True)
class GuacamolSmilesBenchmarkDefinition:
    """Static definition for one GuacaMol-derived SMILES task."""

    task_name: str
    display_name: str
    objective_name: str
    metric_name: str
    source_benchmark: str
    target_smiles: tuple[str, ...] = ()
    fingerprint_types: tuple[str, ...] = ()
    description: str = ""
    category: str = "guacamol"

    @property
    def description_dir(self) -> Path:
        return TASK_DESCRIPTION_ROOT / self.task_name


@dataclass
class GuacamolSmilesTaskConfig:
    """Configuration for one direct-SMILES GuacaMol benchmark task."""

    task_name: str
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    max_smiles_length: int = GUACAMOL_SMILES_DEFAULT_MAX_LENGTH
    metadata: dict[str, Any] = field(default_factory=dict)


class GuacamolSmilesTask(Task):
    """GuacaMol-derived objective over an open SMILES string parameter.

    Scoring formulas are adapted from
    /home/trx/lty/agentic-bbo/bbo/tasks/scientific/guacamol_selfies.py.
    The empty default SMILES is schema-only and must not be used as an optimizer
    initial population or seed.
    """

    def __init__(self, config: GuacamolSmilesTaskConfig):
        if config.task_name not in GUACAMOL_SMILES_TASK_DEFINITIONS:
            available = ", ".join(sorted(GUACAMOL_SMILES_TASK_DEFINITIONS))
            raise ValueError(f"Unknown GuacaMol SMILES task `{config.task_name}`. Available: {available}")
        if config.max_smiles_length <= 0:
            raise ValueError("max_smiles_length must be positive.")

        self.config = config
        self.definition = GUACAMOL_SMILES_TASK_DEFINITIONS[config.task_name]

        Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors = _require_rdkit()
        self._chem = Chem
        self._data_structs = DataStructs
        self._all_chem = AllChem
        self._crippen = Crippen
        self._descriptors = Descriptors
        self._rd_mol_descriptors = rdMolDescriptors
        self._atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(maxDistance=10)
        self._target_fingerprints = self._build_target_fingerprints()

        search_space = SearchSpace(
            [
                StringParam(
                    "smiles",
                    default=GUACAMOL_SMILES_SCHEMA_DEFAULT,
                    min_length=0,
                    max_length=int(config.max_smiles_length),
                )
            ]
        )
        description_dir = config.description_dir or self.definition.description_dir
        self._dataset_summary = {
            "smiles_max_length": int(config.max_smiles_length),
            "schema_default_smiles": GUACAMOL_SMILES_SCHEMA_DEFAULT,
            "schema_default_role": "schema_only_not_initial_population",
            "target_smiles": list(self.definition.target_smiles),
            "fingerprint_types": list(self.definition.fingerprint_types),
        }
        self._spec = TaskSpec(
            name=self.definition.task_name,
            search_space=search_space,
            objectives=(ObjectiveSpec(self.definition.objective_name, ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(self.definition.task_name, description_dir),
            metadata={
                "display_name": self.definition.display_name,
                "source_repo": GUACAMOL_SOURCE_REPO_URL,
                "source_benchmark": self.definition.source_benchmark,
                "source_suite": GUACAMOL_SMILES_SOURCE,
                "representation": "smiles_string",
                "dimension": 1,
                "category": self.definition.category,
                "schema_default_role": "schema_only_not_initial_population",
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    def config_from_smiles(self, smiles: str) -> dict[str, str]:
        """Return the canonical task config for one SMILES string."""

        return self.spec.search_space.coerce_config({"smiles": str(smiles)}, use_defaults=False)

    def _build_target_fingerprints(self) -> dict[tuple[str, str], Any]:
        fingerprints: dict[tuple[str, str], Any] = {}
        for smiles in self.definition.target_smiles:
            mol = self._chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid GuacaMol target SMILES: {smiles!r}")
            for fp_type in self.definition.fingerprint_types:
                fingerprints[(smiles, fp_type)] = self._fingerprint(mol, fp_type)
        return fingerprints

    def _fingerprint(self, mol: Any, fp_type: str) -> Any:
        if fp_type == "ECFP4":
            return self._all_chem.GetMorganFingerprint(mol, 2)
        if fp_type == "ECFP6":
            return self._all_chem.GetMorganFingerprint(mol, 3)
        if fp_type == "FCFP4":
            return self._all_chem.GetMorganFingerprint(mol, 2, useFeatures=True)
        if fp_type == "AP":
            return self._atom_pair_generator.GetSparseCountFingerprint(mol)
        raise ValueError(f"Unsupported fingerprint type for GuacaMol SMILES task: {fp_type}")

    def _tanimoto(self, mol: Any, target_smiles: str, fp_type: str) -> float:
        return float(
            self._data_structs.TanimotoSimilarity(
                self._fingerprint(mol, fp_type),
                self._target_fingerprints[(target_smiles, fp_type)],
            )
        )

    @staticmethod
    def _clipped_score(x: float, upper_x: float, lower_x: float = 0.0) -> float:
        y = (x - lower_x) / (upper_x - lower_x)
        return float(min(max(y, 0.0), 1.0))

    @staticmethod
    def _gaussian(x: float, mu: float, sigma: float) -> float:
        return float(math.exp(-0.5 * ((x - mu) / sigma) ** 2))

    @classmethod
    def _min_gaussian(cls, x: float, mu: float, sigma: float) -> float:
        return cls._gaussian(max(x, mu), mu, sigma)

    @classmethod
    def _max_gaussian(cls, x: float, mu: float, sigma: float) -> float:
        return cls._gaussian(min(x, mu), mu, sigma)

    @staticmethod
    def _geometric_mean(values: Iterable[float]) -> float:
        vals = [max(float(value), 0.0) for value in values]
        if not vals:
            return 0.0
        if any(value == 0.0 for value in vals):
            return 0.0
        return float(math.prod(vals) ** (1.0 / len(vals)))

    def _score_mol(self, mol: Any) -> float:
        name = self.definition.task_name
        if name == GUACAMOL_QED_SMILES_TASK_NAME:
            return float(self._descriptors.qed(mol))
        if name == GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, CELECOXIB_SMILES, "ECFP4"), 1.0)
        if name == GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, TROGLITAZONE_SMILES, "ECFP4"), 1.0)
        if name == GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, ARIPIPRAZOLE_SMILES, "FCFP4"), 0.75)
        if name == GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME:
            similarity = self._clipped_score(self._tanimoto(mol, FEXOFENADINE_SMILES, "AP"), 0.8)
            tpsa_over_90 = self._max_gaussian(float(self._rd_mol_descriptors.CalcTPSA(mol)), 90.0, 10.0)
            logp_under_4 = self._min_gaussian(float(self._crippen.MolLogP(mol)), 4.0, 1.0)
            return self._geometric_mean((similarity, tpsa_over_90, logp_under_4))
        if name == GUACAMOL_MEDIAN1_SMILES_TASK_NAME:
            return self._geometric_mean(
                (
                    self._tanimoto(mol, CAMPHOR_SMILES, "ECFP4"),
                    self._tanimoto(mol, MENTHOL_SMILES, "ECFP4"),
                )
            )
        raise RuntimeError(f"No scoring implementation for task `{name}`.")

    def _score_smiles(self, smiles: str) -> tuple[float, bool, str]:
        if not str(smiles).strip():
            return 0.0, False, ""
        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, False, ""
        try:
            canonical_smiles = self._chem.MolToSmiles(mol, canonical=True)
        except Exception:
            canonical_smiles = str(smiles)
        score = self._score_mol(mol)
        if not math.isfinite(score):
            return 0.0, False, str(canonical_smiles)
        return float(min(max(score, 0.0), 1.0)), True, str(canonical_smiles)

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        smiles = str(config["smiles"])
        score, valid, canonical_smiles = self._score_smiles(smiles)
        loss = 1.0 - score
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={self.definition.objective_name: loss},
            metrics={
                self.definition.metric_name: score,
                "guacamol_score": score,
            },
            elapsed_seconds=elapsed,
            metadata={
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "valid_smiles": valid,
                "source_benchmark": self.definition.source_benchmark,
                "target_smiles": list(self.definition.target_smiles),
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            loss = float(default_result.objectives[self.definition.objective_name])
            score = float(default_result.metrics[self.definition.metric_name])
            if not math.isfinite(loss):
                report.add_error("non_finite_objective", "The GuacaMol SMILES task produced a non-finite loss.")
            if not (0.0 <= score <= 1.0):
                report.add_error("invalid_score_range", "GuacaMol score must be in [0.0, 1.0].")
            if default_result.metadata.get("valid_smiles"):
                report.add_error("schema_default_is_molecule", "The schema-only default must not be a valid molecule.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("guacamol_smiles_failed", f"The task could not score the schema default: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def _definition(
    task_name: str,
    display_name: str,
    objective_stem: str,
    source_benchmark: str,
    *,
    target_smiles: tuple[str, ...] = (),
    fingerprint_types: tuple[str, ...] = (),
    category: str,
    description: str,
) -> GuacamolSmilesBenchmarkDefinition:
    return GuacamolSmilesBenchmarkDefinition(
        task_name=task_name,
        display_name=display_name,
        objective_name=f"{objective_stem}_loss",
        metric_name=f"{objective_stem}_score",
        source_benchmark=source_benchmark,
        target_smiles=target_smiles,
        fingerprint_types=fingerprint_types,
        category=category,
        description=description,
    )


GUACAMOL_SMILES_TASK_DEFINITIONS: dict[str, GuacamolSmilesBenchmarkDefinition] = {
    GUACAMOL_QED_SMILES_TASK_NAME: _definition(
        GUACAMOL_QED_SMILES_TASK_NAME,
        "GuacaMol QED SMILES",
        "guacamol_qed",
        "guacamol.standard_benchmarks.qed_benchmark",
        category="property",
        description="Maximize RDKit QED.",
    ),
    GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME: _definition(
        GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME,
        "GuacaMol Celecoxib Rediscovery SMILES",
        "celecoxib_rediscovery",
        "guacamol.standard_benchmarks.similarity(rediscovery=True, name='Celecoxib')",
        target_smiles=(CELECOXIB_SMILES,),
        fingerprint_types=("ECFP4",),
        category="rediscovery",
        description="Rediscover Celecoxib with ECFP4 Tanimoto similarity.",
    ),
    GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME: _definition(
        GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME,
        "GuacaMol Troglitazone Rediscovery SMILES",
        "troglitazone_rediscovery",
        "guacamol.standard_benchmarks.similarity(rediscovery=True, name='Troglitazone')",
        target_smiles=(TROGLITAZONE_SMILES,),
        fingerprint_types=("ECFP4",),
        category="rediscovery",
        description="Rediscover Troglitazone with ECFP4 Tanimoto similarity.",
    ),
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME: _definition(
        GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME,
        "GuacaMol Aripiprazole Similarity SMILES",
        "aripiprazole_similarity",
        "guacamol.standard_benchmarks.similarity(name='Aripiprazole', fp_type='FCFP4', threshold=0.75)",
        target_smiles=(ARIPIPRAZOLE_SMILES,),
        fingerprint_types=("FCFP4",),
        category="similarity",
        description="Generate molecules similar to Aripiprazole using thresholded FCFP4 Tanimoto similarity.",
    ),
    GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME: _definition(
        GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME,
        "GuacaMol Fexofenadine MPO SMILES",
        "fexofenadine_mpo",
        "guacamol.standard_benchmarks.hard_fexofenadine",
        target_smiles=(FEXOFENADINE_SMILES,),
        fingerprint_types=("AP",),
        category="mpo",
        description="Optimize Fexofenadine similarity together with TPSA and logP modifiers.",
    ),
    GUACAMOL_MEDIAN1_SMILES_TASK_NAME: _definition(
        GUACAMOL_MEDIAN1_SMILES_TASK_NAME,
        "GuacaMol Median Molecules 1 SMILES",
        "median1",
        "guacamol.standard_benchmarks.median_camphor_menthol",
        target_smiles=(CAMPHOR_SMILES, MENTHOL_SMILES),
        fingerprint_types=("ECFP4",),
        category="median",
        description="Generate molecules between camphor and menthol by geometric mean ECFP4 similarity.",
    ),
}


def create_guacamol_smiles_task(
    task_name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    description_dir: Path | None = None,
    max_smiles_length: int = GUACAMOL_SMILES_DEFAULT_MAX_LENGTH,
    metadata: dict[str, Any] | None = None,
) -> GuacamolSmilesTask:
    return GuacamolSmilesTask(
        GuacamolSmilesTaskConfig(
            task_name=task_name,
            max_evaluations=max_evaluations,
            seed=seed,
            description_dir=description_dir,
            max_smiles_length=max_smiles_length,
            metadata=dict(metadata or {}),
        )
    )


def _factory(task_name: str) -> Callable[..., GuacamolSmilesTask]:
    def create_task(**kwargs: Any) -> GuacamolSmilesTask:
        return create_guacamol_smiles_task(task_name, **kwargs)

    return create_task


create_guacamol_qed_smiles_task = _factory(GUACAMOL_QED_SMILES_TASK_NAME)
create_guacamol_celecoxib_rediscovery_smiles_task = _factory(GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME)
create_guacamol_troglitazone_rediscovery_smiles_task = _factory(GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME)
create_guacamol_aripiprazole_similarity_smiles_task = _factory(GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME)
create_guacamol_fexofenadine_mpo_smiles_task = _factory(GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME)
create_guacamol_median1_smiles_task = _factory(GUACAMOL_MEDIAN1_SMILES_TASK_NAME)

GUACAMOL_SMILES_TASK_NAMES: tuple[str, ...] = tuple(GUACAMOL_SMILES_TASK_DEFINITIONS)

__all__ = [
    "GUACAMOL_ARIPIPRAZOLE_SIMILARITY_SMILES_TASK_NAME",
    "GUACAMOL_CELECOXIB_REDISCOVERY_SMILES_TASK_NAME",
    "GUACAMOL_FEXOFENADINE_MPO_SMILES_TASK_NAME",
    "GUACAMOL_MEDIAN1_SMILES_TASK_NAME",
    "GUACAMOL_QED_SMILES_TASK_NAME",
    "GUACAMOL_SMILES_DEFAULT_MAX_EVALUATIONS",
    "GUACAMOL_SMILES_DEFAULT_MAX_LENGTH",
    "GUACAMOL_SMILES_SCHEMA_DEFAULT",
    "GUACAMOL_SMILES_TASK_DEFINITIONS",
    "GUACAMOL_SMILES_TASK_NAMES",
    "GUACAMOL_TROGLITAZONE_REDISCOVERY_SMILES_TASK_NAME",
    "GuacamolSmilesBenchmarkDefinition",
    "GuacamolSmilesTask",
    "GuacamolSmilesTaskConfig",
    "create_guacamol_aripiprazole_similarity_smiles_task",
    "create_guacamol_celecoxib_rediscovery_smiles_task",
    "create_guacamol_fexofenadine_mpo_smiles_task",
    "create_guacamol_median1_smiles_task",
    "create_guacamol_qed_smiles_task",
    "create_guacamol_smiles_task",
    "create_guacamol_troglitazone_rediscovery_smiles_task",
]
