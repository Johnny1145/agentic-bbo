"""HER scientific benchmark task backed by a random-forest oracle."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core import (
    EvaluationResult,
    FloatParam,
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
from .tabular_oracles import fit_random_forest_regressor, numeric_summary, require_pandas

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
HER_DATASET_RELATIVE_PATH = "examples/HER/HER_virtual_data.csv"
HER_DATASET_FILENAME = "HER_virtual_data.csv"
HER_TASK_NAME = "her_demo"
HER_DEFAULT_MAX_EVALUATIONS = 40
HER_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
HER_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / HER_TASK_NAME
HER_FEATURES = (
    "AcidRed871_0gL",
    "L-Cysteine-50gL",
    "MethyleneB_250mgL",
    "NaCl-3M",
    "NaOH-1M",
    "P10-MIX1",
    "PVP-1wt",
    "RhodamineB1_0gL",
    "SDS-1wt",
    "Sodiumsilicate-1wt",
)
HER_TARGET_COLUMN = "Target"


@dataclass
class HerTaskConfig:
    """Configuration for one HER benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HerTask(Task):
    """Task wrapper around the HER tutorial dataset and random-forest oracle."""

    def __init__(self, config: HerTaskConfig | None = None):
        self.config = config or HerTaskConfig()
        self._asset = stage_dataset_asset(
            HER_DATASET_RELATIVE_PATH,
            label="HER",
            task_name=HER_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        pd = require_pandas()
        self._raw_data = pd.read_csv(self._asset.cache_path)
        self._validate_dataset_columns(self._raw_data)

        self._target_max = float(self._raw_data[HER_TARGET_COLUMN].max())
        self._training_data = self._raw_data.loc[:, [*HER_FEATURES, HER_TARGET_COLUMN]].copy()
        self._training_data[HER_TARGET_COLUMN] = self._target_max - self._training_data[HER_TARGET_COLUMN]
        self._oracle = fit_random_forest_regressor(
            self._training_data.loc[:, list(HER_FEATURES)],
            self._training_data[HER_TARGET_COLUMN],
            random_state=self.config.seed,
            n_estimators=100,
        )

        search_space = SearchSpace(
            [FloatParam(name, low=0.0, high=5.0, default=2.5) for name in HER_FEATURES]
        )
        description_dir = self.config.description_dir or HER_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "row_count": int(len(self._training_data)),
            "column_count": int(len(self._training_data.columns)),
            "columns": list(self._raw_data.columns),
            "target_stats": numeric_summary(self._raw_data[HER_TARGET_COLUMN]),
            "regret_stats": numeric_summary(self._training_data[HER_TARGET_COLUMN]),
        }
        self._spec = TaskSpec(
            name=HER_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("regret", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or HER_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(HER_TASK_NAME, description_dir),
            metadata={
                "display_name": "HER Demo",
                "source_paper": HER_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": HER_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "oracle_type": "RandomForestRegressor(n_estimators=100, random_state=<seed>)",
                "dimension": len(HER_FEATURES),
                "cma_initial_config": search_space.defaults(),
                **self.config.metadata,
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

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        pd = require_pandas()
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        features = pd.DataFrame([[config[name] for name in HER_FEATURES]], columns=HER_FEATURES)
        predicted_regret = float(self._oracle.predict(features)[0])
        elapsed = time.perf_counter() - start

        metrics = {
            "predicted_target": self._target_max - predicted_regret,
            "raw_target_max": self._target_max,
        }
        for name in HER_FEATURES:
            metrics[f"coord::{name}"] = float(config[name])

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"regret": predicted_regret},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata=self._asset.as_metadata(),
        )

    def sanity_check(self):
        report = super().sanity_check()
        missing = [name for name in HER_FEATURES if name not in self._training_data.columns]
        if missing:
            report.add_error("missing_feature_columns", f"HER dataset is missing feature columns: {missing!r}")
        if HER_TARGET_COLUMN not in self._training_data.columns:
            report.add_error("missing_target", "HER dataset must contain the `Target` column.")
        if len(self._training_data) == 0:
            report.add_error("empty_dataset", "HER dataset must contain at least one row.")
        if not math.isfinite(self._target_max):
            report.add_error("invalid_target_max", "HER dataset must have a finite maximum target value.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["regret"])):
                report.add_error("non_finite_prediction", "HER oracle produced a non-finite prediction.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("oracle_predict_failed", f"HER oracle could not score the default config: {exc}")
        report.metadata.update(self._dataset_summary)
        return report

    @staticmethod
    def _validate_dataset_columns(frame: Any) -> None:
        expected = [*HER_FEATURES, HER_TARGET_COLUMN]
        missing = [name for name in expected if name not in frame.columns]
        if missing:
            raise ValueError(f"HER dataset is missing required columns: {missing!r}")


def create_her_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> HerTask:
    return HerTask(
        HerTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "HER_DATASET_FILENAME",
    "HER_DATASET_RELATIVE_PATH",
    "HER_DEFAULT_MAX_EVALUATIONS",
    "HER_DESCRIPTION_DIR",
    "HER_FEATURES",
    "HER_SOURCE_PAPER",
    "HER_TASK_NAME",
    "HerTask",
    "HerTaskConfig",
    "create_her_task",
]
