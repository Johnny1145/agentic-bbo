"""BH scientific benchmark task backed by a feature-selected random-forest oracle."""

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
from .tabular_oracles import (
    FeatureSelectionSummary,
    fit_random_forest_regressor,
    numeric_summary,
    require_pandas,
    select_feature_columns_by_importance,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
BH_DATASET_RELATIVE_PATH = "examples/BH/BH_dataset.csv"
BH_DATASET_FILENAME = "BH_dataset.csv"
BH_TASK_NAME = "bh_demo"
BH_DEFAULT_MAX_EVALUATIONS = 40
BH_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
BH_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / BH_TASK_NAME
BH_TARGET_COLUMN = "yield"
BH_FEATURE_SELECTOR = "random_forest"
BH_MIN_IMPORTANCE = 0.01
BH_MAX_CUM_IMPORTANCE = 0.8
BH_MAX_FEATURES = 20


@dataclass
class BhTaskConfig:
    """Configuration for one BH benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BhTask(Task):
    """Task wrapper around the BH tutorial dataset and selected continuous features."""

    def __init__(self, config: BhTaskConfig | None = None):
        self.config = config or BhTaskConfig()
        self._asset = stage_dataset_asset(
            BH_DATASET_RELATIVE_PATH,
            label="BH",
            task_name=BH_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        pd = require_pandas()
        raw_frame = pd.read_csv(self._asset.cache_path)
        if BH_TARGET_COLUMN not in raw_frame.columns:
            raise ValueError("BH dataset is missing the `yield` column.")

        self._raw_yield_max = float(raw_frame[BH_TARGET_COLUMN].max())
        processed = raw_frame.copy()
        processed[BH_TARGET_COLUMN] = self._raw_yield_max - processed[BH_TARGET_COLUMN]
        features = processed.drop(columns=[BH_TARGET_COLUMN, "cost", "new_index"], errors="ignore")
        self._feature_selection = select_feature_columns_by_importance(
            features,
            processed[BH_TARGET_COLUMN],
            extractor=BH_FEATURE_SELECTOR,
            max_n=BH_MAX_FEATURES,
            max_cum_imp=BH_MAX_CUM_IMPORTANCE,
            min_imp=BH_MIN_IMPORTANCE,
            random_state=self.config.seed,
        )
        self._selected_features = list(self._feature_selection.selected_columns)
        self._training_data = processed.loc[:, [*self._selected_features, BH_TARGET_COLUMN]].copy()
        self._oracle = fit_random_forest_regressor(
            self._training_data.loc[:, self._selected_features],
            self._training_data[BH_TARGET_COLUMN],
            random_state=self.config.seed,
            n_estimators=100,
        )

        search_space = SearchSpace(
            [
                FloatParam(
                    feature,
                    low=float(self._training_data[feature].min()),
                    high=float(self._training_data[feature].max()),
                    default=float(self._training_data[feature].mean()),
                )
                for feature in self._selected_features
            ]
        )
        description_dir = self.config.description_dir or BH_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "row_count": int(len(self._training_data)),
            "column_count": int(len(self._training_data.columns)),
            "columns": list(raw_frame.columns),
            "raw_yield_stats": numeric_summary(raw_frame[BH_TARGET_COLUMN]),
            "regret_yield_stats": numeric_summary(self._training_data[BH_TARGET_COLUMN]),
            "selected_features": list(self._selected_features),
            "feature_importances": dict(self._feature_selection.importances),
            "selected_feature_bounds": {
                feature: {
                    "min": float(self._training_data[feature].min()),
                    "max": float(self._training_data[feature].max()),
                }
                for feature in self._selected_features
            },
        }
        self._spec = TaskSpec(
            name=BH_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("regret", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or BH_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(BH_TASK_NAME, description_dir),
            metadata={
                "display_name": "BH Demo",
                "source_paper": BH_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": BH_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "oracle_type": "RandomForestRegressor(n_estimators=100, random_state=<seed>)",
                "dimension": len(self._selected_features),
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

    @property
    def feature_selection(self) -> FeatureSelectionSummary:
        return self._feature_selection

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        pd = require_pandas()
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        features = pd.DataFrame([{name: config[name] for name in self._selected_features}])
        predicted_regret = float(self._oracle.predict(features)[0])
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"regret": predicted_regret},
            metrics={
                "predicted_yield": self._raw_yield_max - predicted_regret,
                "raw_yield_max": self._raw_yield_max,
            },
            elapsed_seconds=elapsed,
            metadata=self._asset.as_metadata(),
        )

    def sanity_check(self):
        report = super().sanity_check()
        if len(self._selected_features) == 0:
            report.add_error("no_features_selected", "BH feature selection must keep at least one feature.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["regret"])):
                report.add_error("non_finite_prediction", "BH oracle produced a non-finite prediction.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("oracle_predict_failed", f"BH oracle could not score the default config: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def create_bh_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> BhTask:
    return BhTask(
        BhTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "BH_DATASET_FILENAME",
    "BH_DATASET_RELATIVE_PATH",
    "BH_DEFAULT_MAX_EVALUATIONS",
    "BH_DESCRIPTION_DIR",
    "BH_TASK_NAME",
    "BhTask",
    "BhTaskConfig",
    "create_bh_task",
]
