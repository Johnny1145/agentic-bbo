"""GuacaMol goal-directed benchmarks over fixed-length SELFIES tokens."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ...core import (
    CategoricalParam,
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
from .data_assets import SOURCE_REPO_URL, DatasetAsset, stage_dataset_asset
from .molecule import MOLECULE_ARCHIVE_MEMBER, MOLECULE_DATASET_RELATIVE_PATH, load_zinc_smiles

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
GUACAMOL_SELFIES_DEFAULT_MAX_EVALUATIONS = 40
GUACAMOL_SELFIES_DEFAULT_MAX_TOKENS = 64
GUACAMOL_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT = 4096
GUACAMOL_SELFIES_PAD_TOKEN = "__PAD__"
GUACAMOL_SELFIES_EOS_TOKEN = "__EOS__"
GUACAMOL_SELFIES_SOURCE = "guacamol.goal_directed_suite"
GUACAMOL_SELFIES_FALLBACK_SMILES = (
    "C",
    "CC",
    "CCO",
    "CCN",
    "COC",
    "c1ccccc1",
    "CC(=O)O",
)

GUACAMOL_QED_SELFIES_TASK_NAME = "guacamol_qed_selfies_demo"
GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME = "guacamol_celecoxib_rediscovery_demo"
GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME = "guacamol_troglitazone_rediscovery_demo"
GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME = "guacamol_aripiprazole_similarity_demo"
GUACAMOL_FEXOFENADINE_MPO_TASK_NAME = "guacamol_fexofenadine_mpo_demo"
GUACAMOL_MEDIAN1_TASK_NAME = "guacamol_median1_demo"

CELECOXIB_SMILES = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
TROGLITAZONE_SMILES = "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O"
ARIPIPRAZOLE_SMILES = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
FEXOFENADINE_SMILES = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
CAMPHOR_SMILES = "CC1(C)C2CCC1(C)C(=O)C2"
MENTHOL_SMILES = "CC(C)C1CCC(C)CC1O"


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The GuacaMol SELFIES tasks require RDKit. Install with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors


def _require_selfies():
    try:
        import selfies as sf
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The GuacaMol SELFIES tasks require the `selfies` package. Install with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return sf


@dataclass(frozen=True)
class SelfiesDecodeResult:
    """Decoded molecule representation for one token configuration."""

    selfies: str
    smiles: str
    tokens: tuple[str, ...]
    valid_selfies: bool
    decode_error: str | None = None


@dataclass(frozen=True)
class GuacamolSelfiesBenchmarkDefinition:
    """Static definition for one GuacaMol-derived SELFIES task."""

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
class GuacamolSelfiesTaskConfig:
    """Configuration for one fixed-length SELFIES GuacaMol benchmark task."""

    task_name: str
    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    max_selfies_tokens: int = GUACAMOL_SELFIES_DEFAULT_MAX_TOKENS
    vocabulary_source_limit: int = GUACAMOL_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT
    metadata: dict[str, Any] = field(default_factory=dict)


class GuacamolSelfiesTask(Task):
    """GuacaMol-derived objective over a fixed-length SELFIES token search space."""

    def __init__(self, config: GuacamolSelfiesTaskConfig):
        if config.task_name not in GUACAMOL_SELFIES_TASK_DEFINITIONS:
            available = ", ".join(sorted(GUACAMOL_SELFIES_TASK_DEFINITIONS))
            raise ValueError(f"Unknown GuacaMol SELFIES task `{config.task_name}`. Available: {available}")
        if config.max_selfies_tokens <= 0:
            raise ValueError("max_selfies_tokens must be positive.")
        if config.vocabulary_source_limit <= 0:
            raise ValueError("vocabulary_source_limit must be positive.")

        self.config = config
        self.definition = GUACAMOL_SELFIES_TASK_DEFINITIONS[config.task_name]
        self._asset = stage_dataset_asset(
            MOLECULE_DATASET_RELATIVE_PATH,
            label=f"{self.definition.display_name}/SELFIES",
            task_name=self.definition.task_name,
            source_root=config.source_root,
            cache_root=config.cache_root,
        )
        self._smiles_list = load_zinc_smiles(self._asset.cache_path)
        if not self._smiles_list:
            raise ValueError("GuacaMol SELFIES tasks require at least one source SMILES string.")

        Chem, DataStructs, AllChem, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors = _require_rdkit()
        self._chem = Chem
        self._data_structs = DataStructs
        self._all_chem = AllChem
        self._crippen = Crippen
        self._descriptors = Descriptors
        self._rd_mol_descriptors = rdMolDescriptors
        self._atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(maxDistance=10)
        self._selfies = _require_selfies()
        self._max_tokens = int(config.max_selfies_tokens)
        self._target_fingerprints = self._build_target_fingerprints()

        vocab, default_smiles, default_selfies, default_tokens, default_score = self._build_vocabulary_and_default()
        choices = (GUACAMOL_SELFIES_PAD_TOKEN, GUACAMOL_SELFIES_EOS_TOKEN, *vocab)
        default_config = self._tokens_to_config(default_tokens)
        search_space = SearchSpace(
            [
                CategoricalParam(
                    self._token_param_name(index),
                    choices=choices,
                    default=default_config[self._token_param_name(index)],
                )
                for index in range(self._max_tokens)
            ]
        )
        description_dir = config.description_dir or self.definition.description_dir
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "source_item_count": int(len(self._smiles_list)),
            "vocabulary_source_limit": int(config.vocabulary_source_limit),
            "selfies_vocabulary_size": int(len(vocab)),
            "search_token_choices": int(len(choices)),
            "max_selfies_tokens": int(self._max_tokens),
            "archive_member": MOLECULE_ARCHIVE_MEMBER,
            "default_smiles": default_smiles,
            "default_selfies": default_selfies,
            "default_score": float(default_score),
            "target_smiles": list(self.definition.target_smiles),
            "fingerprint_types": list(self.definition.fingerprint_types),
        }
        self._spec = TaskSpec(
            name=self.definition.task_name,
            search_space=search_space,
            objectives=(ObjectiveSpec(self.definition.objective_name, ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or GUACAMOL_SELFIES_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(self.definition.task_name, description_dir),
            metadata={
                "display_name": self.definition.display_name,
                "source_repo": SOURCE_REPO_URL,
                "source_benchmark": self.definition.source_benchmark,
                "source_ref": self._asset.source_ref,
                "dataset_cache_path": str(self._asset.cache_path),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "representation": "fixed_length_selfies_tokens",
                "selfies_pad_token": GUACAMOL_SELFIES_PAD_TOKEN,
                "selfies_eos_token": GUACAMOL_SELFIES_EOS_TOKEN,
                "dimension": self._max_tokens,
                "category": self.definition.category,
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def dataset_asset(self) -> DatasetAsset:
        return self._asset

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    @staticmethod
    def _token_param_name(index: int) -> str:
        return f"selfies_token_{index:02d}"

    @staticmethod
    def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            unique.append(text)
            seen.add(text)
        return tuple(unique)

    def _encode_smiles(self, smiles: str) -> tuple[str, tuple[str, ...]] | None:
        try:
            selfies = self._selfies.encoder(smiles)
        except Exception:
            return None
        if not selfies:
            return None
        tokens = tuple(self._selfies.split_selfies(selfies))
        if not tokens:
            return None
        return str(selfies), tokens

    def _build_vocabulary_and_default(self) -> tuple[tuple[str, ...], str, str, tuple[str, ...], float]:
        vocabulary: set[str] = set()
        scored_candidates: list[tuple[str, str, tuple[str, ...], float]] = []
        source_smiles = list(GUACAMOL_SELFIES_FALLBACK_SMILES)
        source_smiles.extend(self.definition.target_smiles)
        source_smiles.extend(self._smiles_list[: self.config.vocabulary_source_limit])

        for smiles in self._unique_strings(source_smiles):
            encoded = self._encode_smiles(smiles)
            if encoded is None:
                continue
            selfies, tokens = encoded
            vocabulary.update(tokens)
            if len(tokens) > self._max_tokens:
                continue
            score, valid = self._score_smiles(smiles)
            if valid:
                scored_candidates.append((smiles, selfies, tokens, score))

        if not vocabulary:
            raise ValueError("Could not build a SELFIES token vocabulary from source molecules.")
        if not scored_candidates:
            raise ValueError("Could not find a valid default molecule within max_selfies_tokens.")

        default_smiles, default_selfies, default_tokens, default_score = max(
            scored_candidates,
            key=lambda item: item[3],
        )
        return tuple(sorted(vocabulary)), default_smiles, default_selfies, default_tokens, default_score

    def _tokens_to_config(self, tokens: tuple[str, ...]) -> dict[str, str]:
        values = list(tokens[: self._max_tokens])
        if len(values) < self._max_tokens:
            values.append(GUACAMOL_SELFIES_EOS_TOKEN)
        while len(values) < self._max_tokens:
            values.append(GUACAMOL_SELFIES_PAD_TOKEN)
        return {self._token_param_name(index): values[index] for index in range(self._max_tokens)}

    def config_from_smiles(self, smiles: str) -> dict[str, str]:
        """Convert one SMILES string into the canonical fixed-length SELFIES config."""

        encoded = self._encode_smiles(smiles)
        if encoded is None:
            raise ValueError(f"Could not encode SMILES as SELFIES: {smiles!r}")
        _, tokens = encoded
        return self.spec.search_space.coerce_config(self._tokens_to_config(tokens), use_defaults=False)

    def config_from_selfies(self, selfies: str) -> dict[str, str]:
        """Convert a SELFIES string into the canonical fixed-length SELFIES config."""

        tokens = tuple(self._selfies.split_selfies(str(selfies)))
        if not tokens:
            raise ValueError("SELFIES string must contain at least one token.")
        return self.spec.search_space.coerce_config(self._tokens_to_config(tokens), use_defaults=False)

    def _decode_config(self, config: dict[str, Any]) -> SelfiesDecodeResult:
        tokens: list[str] = []
        for index in range(self._max_tokens):
            token = str(config[self._token_param_name(index)])
            if token == GUACAMOL_SELFIES_EOS_TOKEN:
                break
            if token == GUACAMOL_SELFIES_PAD_TOKEN:
                continue
            tokens.append(token)
        selfies = "".join(tokens)
        if not selfies:
            return SelfiesDecodeResult("", "", tuple(tokens), False, "empty_selfies")
        try:
            smiles = str(self._selfies.decoder(selfies))
        except Exception as exc:
            return SelfiesDecodeResult(selfies, "", tuple(tokens), False, f"{type(exc).__name__}: {exc}")
        return SelfiesDecodeResult(selfies, smiles, tuple(tokens), bool(smiles), None)

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
        raise ValueError(f"Unsupported fingerprint type for GuacaMol SELFIES task: {fp_type}")

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
        if name == GUACAMOL_QED_SELFIES_TASK_NAME:
            return float(self._descriptors.qed(mol))
        if name == GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, CELECOXIB_SMILES, "ECFP4"), 1.0)
        if name == GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, TROGLITAZONE_SMILES, "ECFP4"), 1.0)
        if name == GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME:
            return self._clipped_score(self._tanimoto(mol, ARIPIPRAZOLE_SMILES, "FCFP4"), 0.75)
        if name == GUACAMOL_FEXOFENADINE_MPO_TASK_NAME:
            similarity = self._clipped_score(self._tanimoto(mol, FEXOFENADINE_SMILES, "AP"), 0.8)
            tpsa_over_90 = self._max_gaussian(float(self._rd_mol_descriptors.CalcTPSA(mol)), 90.0, 10.0)
            logp_under_4 = self._min_gaussian(float(self._crippen.MolLogP(mol)), 4.0, 1.0)
            return self._geometric_mean((similarity, tpsa_over_90, logp_under_4))
        if name == GUACAMOL_MEDIAN1_TASK_NAME:
            return self._geometric_mean(
                (
                    self._tanimoto(mol, CAMPHOR_SMILES, "ECFP4"),
                    self._tanimoto(mol, MENTHOL_SMILES, "ECFP4"),
                )
            )
        raise RuntimeError(f"No scoring implementation for task `{name}`.")

    def _score_smiles(self, smiles: str) -> tuple[float, bool]:
        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, False
        score = self._score_mol(mol)
        if not math.isfinite(score):
            return 0.0, False
        return float(min(max(score, 0.0), 1.0)), True

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        decoded = self._decode_config(config)
        score, valid = self._score_smiles(decoded.smiles) if decoded.valid_selfies else (0.0, False)
        loss = 1.0 - score
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={self.definition.objective_name: loss},
            metrics={
                self.definition.metric_name: score,
                "guacamol_score": score,
                "selfies_token_count": len(decoded.tokens),
            },
            elapsed_seconds=elapsed,
            metadata={
                **self._asset.as_metadata(),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "selfies": decoded.selfies,
                "smiles": decoded.smiles,
                "selfies_tokens": list(decoded.tokens),
                "valid_selfies": decoded.valid_selfies,
                "valid_smiles": valid,
                "decode_error": decoded.decode_error,
                "source_benchmark": self.definition.source_benchmark,
                "target_smiles": list(self.definition.target_smiles),
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self._dataset_summary["selfies_vocabulary_size"] <= 0:
            report.add_error("empty_selfies_vocabulary", "The GuacaMol SELFIES task has an empty token vocabulary.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            loss = float(default_result.objectives[self.definition.objective_name])
            score = float(default_result.metrics[self.definition.metric_name])
            if not math.isfinite(loss):
                report.add_error("non_finite_objective", "The GuacaMol SELFIES task produced a non-finite loss.")
            if not (0.0 <= score <= 1.0):
                report.add_error("invalid_score_range", "GuacaMol score must be in [0.0, 1.0].")
            if not default_result.metadata.get("valid_smiles"):
                report.add_error("invalid_default_molecule", "The default SELFIES config did not decode to a valid molecule.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("guacamol_selfies_failed", f"The task could not score the default SELFIES config: {exc}")
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
) -> GuacamolSelfiesBenchmarkDefinition:
    return GuacamolSelfiesBenchmarkDefinition(
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


GUACAMOL_SELFIES_TASK_DEFINITIONS: dict[str, GuacamolSelfiesBenchmarkDefinition] = {
    GUACAMOL_QED_SELFIES_TASK_NAME: _definition(
        GUACAMOL_QED_SELFIES_TASK_NAME,
        "GuacaMol QED SELFIES",
        "guacamol_qed",
        "guacamol.standard_benchmarks.qed_benchmark",
        category="property",
        description="Maximize RDKit QED.",
    ),
    GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME: _definition(
        GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME,
        "GuacaMol Celecoxib Rediscovery",
        "celecoxib_rediscovery",
        "guacamol.standard_benchmarks.similarity(rediscovery=True, name='Celecoxib')",
        target_smiles=(CELECOXIB_SMILES,),
        fingerprint_types=("ECFP4",),
        category="rediscovery",
        description="Rediscover Celecoxib with ECFP4 Tanimoto similarity.",
    ),
    GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME: _definition(
        GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME,
        "GuacaMol Troglitazone Rediscovery",
        "troglitazone_rediscovery",
        "guacamol.standard_benchmarks.similarity(rediscovery=True, name='Troglitazone')",
        target_smiles=(TROGLITAZONE_SMILES,),
        fingerprint_types=("ECFP4",),
        category="rediscovery",
        description="Rediscover Troglitazone with ECFP4 Tanimoto similarity.",
    ),
    GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME: _definition(
        GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME,
        "GuacaMol Aripiprazole Similarity",
        "aripiprazole_similarity",
        "guacamol.standard_benchmarks.similarity(name='Aripiprazole', fp_type='FCFP4', threshold=0.75)",
        target_smiles=(ARIPIPRAZOLE_SMILES,),
        fingerprint_types=("FCFP4",),
        category="similarity",
        description="Generate molecules similar to Aripiprazole using thresholded FCFP4 Tanimoto similarity.",
    ),
    GUACAMOL_FEXOFENADINE_MPO_TASK_NAME: _definition(
        GUACAMOL_FEXOFENADINE_MPO_TASK_NAME,
        "GuacaMol Fexofenadine MPO",
        "fexofenadine_mpo",
        "guacamol.standard_benchmarks.hard_fexofenadine",
        target_smiles=(FEXOFENADINE_SMILES,),
        fingerprint_types=("AP",),
        category="mpo",
        description="Optimize Fexofenadine similarity together with TPSA and logP modifiers.",
    ),
    GUACAMOL_MEDIAN1_TASK_NAME: _definition(
        GUACAMOL_MEDIAN1_TASK_NAME,
        "GuacaMol Median Molecules 1",
        "median1",
        "guacamol.standard_benchmarks.median_camphor_menthol",
        target_smiles=(CAMPHOR_SMILES, MENTHOL_SMILES),
        fingerprint_types=("ECFP4",),
        category="median",
        description="Generate molecules between camphor and menthol by geometric mean ECFP4 similarity.",
    ),
}


def create_guacamol_selfies_task(
    task_name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    max_selfies_tokens: int = GUACAMOL_SELFIES_DEFAULT_MAX_TOKENS,
    vocabulary_source_limit: int = GUACAMOL_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT,
    metadata: dict[str, Any] | None = None,
) -> GuacamolSelfiesTask:
    return GuacamolSelfiesTask(
        GuacamolSelfiesTaskConfig(
            task_name=task_name,
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            max_selfies_tokens=max_selfies_tokens,
            vocabulary_source_limit=vocabulary_source_limit,
            metadata=dict(metadata or {}),
        )
    )


def _factory(task_name: str) -> Callable[..., GuacamolSelfiesTask]:
    def create_task(**kwargs: Any) -> GuacamolSelfiesTask:
        return create_guacamol_selfies_task(task_name, **kwargs)

    return create_task


create_guacamol_qed_selfies_task = _factory(GUACAMOL_QED_SELFIES_TASK_NAME)
create_guacamol_celecoxib_rediscovery_task = _factory(GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME)
create_guacamol_troglitazone_rediscovery_task = _factory(GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME)
create_guacamol_aripiprazole_similarity_task = _factory(GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME)
create_guacamol_fexofenadine_mpo_task = _factory(GUACAMOL_FEXOFENADINE_MPO_TASK_NAME)
create_guacamol_median1_task = _factory(GUACAMOL_MEDIAN1_TASK_NAME)

GUACAMOL_SELFIES_TASK_NAMES: tuple[str, ...] = tuple(GUACAMOL_SELFIES_TASK_DEFINITIONS)

__all__ = [
    "GUACAMOL_ARIPIPRAZOLE_SIMILARITY_TASK_NAME",
    "GUACAMOL_CELECOXIB_REDISCOVERY_TASK_NAME",
    "GUACAMOL_FEXOFENADINE_MPO_TASK_NAME",
    "GUACAMOL_MEDIAN1_TASK_NAME",
    "GUACAMOL_QED_SELFIES_TASK_NAME",
    "GUACAMOL_SELFIES_DEFAULT_MAX_EVALUATIONS",
    "GUACAMOL_SELFIES_DEFAULT_MAX_TOKENS",
    "GUACAMOL_SELFIES_DEFAULT_VOCABULARY_SOURCE_LIMIT",
    "GUACAMOL_SELFIES_EOS_TOKEN",
    "GUACAMOL_SELFIES_PAD_TOKEN",
    "GUACAMOL_SELFIES_TASK_DEFINITIONS",
    "GUACAMOL_SELFIES_TASK_NAMES",
    "GUACAMOL_TROGLITAZONE_REDISCOVERY_TASK_NAME",
    "GuacamolSelfiesBenchmarkDefinition",
    "GuacamolSelfiesTask",
    "GuacamolSelfiesTaskConfig",
    "SelfiesDecodeResult",
    "create_guacamol_aripiprazole_similarity_task",
    "create_guacamol_celecoxib_rediscovery_task",
    "create_guacamol_fexofenadine_mpo_task",
    "create_guacamol_median1_task",
    "create_guacamol_qed_selfies_task",
    "create_guacamol_selfies_task",
    "create_guacamol_troglitazone_rediscovery_task",
]
