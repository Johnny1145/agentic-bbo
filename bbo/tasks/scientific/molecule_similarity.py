"""Molecule similarity scientific benchmark task using RDKit fingerprints."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
MOLECULE_SIMILARITY_DATASET_FILENAME = "zinc.txt.gz"
MOLECULE_SIMILARITY_TASK_NAME = "molecule_similarity_demo"
MOLECULE_SIMILARITY_DEFAULT_MAX_EVALUATIONS = 40
MOLECULE_SIMILARITY_DEFAULT_MAX_TOKENS = 16
MOLECULE_SIMILARITY_DEFAULT_VOCABULARY_SOURCE_LIMIT = 4096
MOLECULE_SIMILARITY_PAD_TOKEN = "__PAD__"
MOLECULE_SIMILARITY_EOS_TOKEN = "__EOS__"
MOLECULE_SIMILARITY_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
MOLECULE_SIMILARITY_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / MOLECULE_SIMILARITY_TASK_NAME
MOLECULE_SIMILARITY_TARGET_NAME = "Celecoxib"
MOLECULE_SIMILARITY_TARGET_SMILES = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
MOLECULE_SIMILARITY_FINGERPRINT = "ECFP4"
MOLECULE_SIMILARITY_RADIUS = 2
MOLECULE_SIMILARITY_FALLBACK_SMILES = (
    "C",
    "CC",
    "CCO",
    "CCN",
    "COC",
    "c1ccccc1",
    "CC(=O)O",
)


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The molecule/similarity task requires RDKit. Install it with "
            "`uv sync --extra dev --extra bo-tutorial` or provide a compatible conda environment."
        ) from exc
    return Chem, rdFingerprintGenerator, DataStructs


def _require_selfies():
    try:
        import selfies as sf
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The molecule/similarity SELFIES task requires the `selfies` package. Install it with "
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


@dataclass
class MoleculeSimilarityTaskConfig:
    """Configuration for one molecule similarity benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    target_smiles: str = MOLECULE_SIMILARITY_TARGET_SMILES
    target_name: str = MOLECULE_SIMILARITY_TARGET_NAME
    max_selfies_tokens: int = MOLECULE_SIMILARITY_DEFAULT_MAX_TOKENS
    vocabulary_source_limit: int = MOLECULE_SIMILARITY_DEFAULT_VOCABULARY_SOURCE_LIMIT
    metadata: dict[str, Any] = field(default_factory=dict)


class MoleculeSimilarityTask(Task):
    """Task wrapper around SELFIES-token molecule generation and ECFP4/Tanimoto similarity."""

    def __init__(self, config: MoleculeSimilarityTaskConfig | None = None):
        self.config = config or MoleculeSimilarityTaskConfig()
        if self.config.max_selfies_tokens <= 0:
            raise ValueError("max_selfies_tokens must be positive.")
        if self.config.vocabulary_source_limit <= 0:
            raise ValueError("vocabulary_source_limit must be positive.")

        self._asset = stage_dataset_asset(
            MOLECULE_DATASET_RELATIVE_PATH,
            label="Molecule/Similarity SELFIES",
            task_name=MOLECULE_SIMILARITY_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        self._smiles_list = load_zinc_smiles(self._asset.cache_path)
        if not self._smiles_list:
            raise ValueError("The molecule/similarity dataset must contain at least one SMILES string.")

        Chem, rdFingerprintGenerator, DataStructs = _require_rdkit()
        self._chem = Chem
        self._data_structs = DataStructs
        self._fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MOLECULE_SIMILARITY_RADIUS)
        self._target_mol = Chem.MolFromSmiles(self.config.target_smiles)
        if self._target_mol is None:
            raise ValueError(f"The target SMILES for molecule similarity is invalid: {self.config.target_smiles!r}")
        self._target_fp = self._fingerprint(self._target_mol)
        self._selfies = _require_selfies()
        self._max_tokens = int(self.config.max_selfies_tokens)

        vocab, default_smiles, default_selfies, default_tokens, default_similarity = self._build_vocabulary_and_default()
        choices = (MOLECULE_SIMILARITY_PAD_TOKEN, MOLECULE_SIMILARITY_EOS_TOKEN, *vocab)
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
        description_dir = self.config.description_dir or MOLECULE_SIMILARITY_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "source_item_count": int(len(self._smiles_list)),
            "vocabulary_source_limit": int(self.config.vocabulary_source_limit),
            "selfies_vocabulary_size": int(len(vocab)),
            "search_token_choices": int(len(choices)),
            "max_selfies_tokens": int(self._max_tokens),
            "archive_member": MOLECULE_ARCHIVE_MEMBER,
            "default_smiles": default_smiles,
            "default_selfies": default_selfies,
            "default_tanimoto_similarity": float(default_similarity),
            "target_name": self.config.target_name,
            "target_smiles": self.config.target_smiles,
            "fingerprint": MOLECULE_SIMILARITY_FINGERPRINT,
        }
        self._spec = TaskSpec(
            name=MOLECULE_SIMILARITY_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("similarity_loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or MOLECULE_SIMILARITY_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(MOLECULE_SIMILARITY_TASK_NAME, description_dir),
            metadata={
                "display_name": "Molecule Similarity Demo",
                "source_paper": MOLECULE_SIMILARITY_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": MOLECULE_SIMILARITY_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "target_name": self.config.target_name,
                "target_smiles": self.config.target_smiles,
                "fingerprint": MOLECULE_SIMILARITY_FINGERPRINT,
                "representation": "fixed_length_selfies_tokens",
                "selfies_pad_token": MOLECULE_SIMILARITY_PAD_TOKEN,
                "selfies_eos_token": MOLECULE_SIMILARITY_EOS_TOKEN,
                "dimension": self._max_tokens,
                **self.config.metadata,
            },
        )

    def _fingerprint(self, mol: Any) -> Any:
        return self._fingerprint_generator.GetSparseCountFingerprint(mol)

    def _score_smiles(self, smiles: str) -> tuple[float, bool]:
        molecule = self._chem.MolFromSmiles(smiles)
        if molecule is None:
            return 0.0, False
        fp = self._fingerprint(molecule)
        return float(self._data_structs.TanimotoSimilarity(fp, self._target_fp)), True

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

        source_smiles = list(MOLECULE_SIMILARITY_FALLBACK_SMILES)
        source_smiles.append(self.config.target_smiles)
        source_smiles.extend(self._smiles_list[: self.config.vocabulary_source_limit])
        for smiles in self._unique_strings(source_smiles):
            encoded = self._encode_smiles(smiles)
            if encoded is None:
                continue
            selfies, tokens = encoded
            vocabulary.update(tokens)
            if len(tokens) > self._max_tokens:
                continue
            similarity, valid = self._score_smiles(smiles)
            if valid:
                scored_candidates.append((smiles, selfies, tokens, similarity))

        if not vocabulary:
            raise ValueError("Could not build a SELFIES token vocabulary from the source molecules.")
        if not scored_candidates:
            raise ValueError("Could not find a valid default molecule within max_selfies_tokens.")

        default_smiles, default_selfies, default_tokens, default_similarity = max(
            scored_candidates,
            key=lambda item: item[3],
        )
        return tuple(sorted(vocabulary)), default_smiles, default_selfies, default_tokens, default_similarity

    def _tokens_to_config(self, tokens: tuple[str, ...]) -> dict[str, str]:
        values = list(tokens[: self._max_tokens])
        if len(values) < self._max_tokens:
            values.append(MOLECULE_SIMILARITY_EOS_TOKEN)
        while len(values) < self._max_tokens:
            values.append(MOLECULE_SIMILARITY_PAD_TOKEN)
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
            if token == MOLECULE_SIMILARITY_EOS_TOKEN:
                break
            if token == MOLECULE_SIMILARITY_PAD_TOKEN:
                continue
            tokens.append(token)
        selfies = "".join(tokens)
        if not selfies:
            return SelfiesDecodeResult(
                selfies="",
                smiles="",
                tokens=tuple(tokens),
                valid_selfies=False,
                decode_error="empty_selfies",
            )
        try:
            smiles = str(self._selfies.decoder(selfies))
        except Exception as exc:
            return SelfiesDecodeResult(
                selfies=selfies,
                smiles="",
                tokens=tuple(tokens),
                valid_selfies=False,
                decode_error=f"{type(exc).__name__}: {exc}",
            )
        return SelfiesDecodeResult(
            selfies=selfies,
            smiles=smiles,
            tokens=tuple(tokens),
            valid_selfies=bool(smiles),
            decode_error=None,
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

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        decoded = self._decode_config(config)
        similarity, valid = self._score_smiles(decoded.smiles) if decoded.valid_selfies else (0.0, False)
        similarity_loss = 1.0 - similarity
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"similarity_loss": similarity_loss},
            metrics={
                "tanimoto_similarity": similarity,
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
                "target_name": self.config.target_name,
                "target_smiles": self.config.target_smiles,
                "fingerprint": MOLECULE_SIMILARITY_FINGERPRINT,
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self._dataset_summary["selfies_vocabulary_size"] <= 0:
            report.add_error("empty_selfies_vocabulary", "The molecule/similarity task has an empty token vocabulary.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            objective = float(default_result.objectives["similarity_loss"])
            similarity = float(default_result.metrics["tanimoto_similarity"])
            if not math.isfinite(objective):
                report.add_error("non_finite_prediction", "The molecule/similarity task produced a non-finite loss.")
            if not (0.0 <= similarity <= 1.0):
                report.add_error("invalid_similarity_range", "Tanimoto similarity must be between 0.0 and 1.0.")
            if not default_result.metadata.get("valid_smiles"):
                report.add_error("invalid_default_molecule", "The default SELFIES config did not decode to a valid molecule.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error(
                "similarity_evaluation_failed",
                f"The molecule/similarity task could not score the default SELFIES config: {exc}",
            )
        report.metadata.update(self._dataset_summary)
        return report


def create_molecule_similarity_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    target_smiles: str = MOLECULE_SIMILARITY_TARGET_SMILES,
    target_name: str = MOLECULE_SIMILARITY_TARGET_NAME,
    max_selfies_tokens: int = MOLECULE_SIMILARITY_DEFAULT_MAX_TOKENS,
    vocabulary_source_limit: int = MOLECULE_SIMILARITY_DEFAULT_VOCABULARY_SOURCE_LIMIT,
    metadata: dict[str, Any] | None = None,
) -> MoleculeSimilarityTask:
    return MoleculeSimilarityTask(
        MoleculeSimilarityTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            target_smiles=target_smiles,
            target_name=target_name,
            max_selfies_tokens=max_selfies_tokens,
            vocabulary_source_limit=vocabulary_source_limit,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "MOLECULE_SIMILARITY_DATASET_FILENAME",
    "MOLECULE_SIMILARITY_DEFAULT_MAX_EVALUATIONS",
    "MOLECULE_SIMILARITY_DEFAULT_MAX_TOKENS",
    "MOLECULE_SIMILARITY_DEFAULT_VOCABULARY_SOURCE_LIMIT",
    "MOLECULE_SIMILARITY_DESCRIPTION_DIR",
    "MOLECULE_SIMILARITY_EOS_TOKEN",
    "MOLECULE_SIMILARITY_FINGERPRINT",
    "MOLECULE_SIMILARITY_PAD_TOKEN",
    "MOLECULE_SIMILARITY_RADIUS",
    "MOLECULE_SIMILARITY_SOURCE_PAPER",
    "MOLECULE_SIMILARITY_TARGET_NAME",
    "MOLECULE_SIMILARITY_TARGET_SMILES",
    "MOLECULE_SIMILARITY_TASK_NAME",
    "MoleculeSimilarityTask",
    "MoleculeSimilarityTaskConfig",
    "SelfiesDecodeResult",
    "create_molecule_similarity_task",
]
