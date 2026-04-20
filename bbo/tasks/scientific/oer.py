"""OER scientific benchmark task backed by a mixed-feature random-forest oracle."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    IntParam,
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
    align_dummy_columns,
    encode_categorical_frame,
    fit_random_forest_regressor,
    numeric_summary,
    require_pandas,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
OER_DATASET_RELATIVE_PATH = "examples/OER/OER.csv"
OER_CLEAN_REFERENCE_RELATIVE_PATH = "examples/OER/OER_clean.csv"
OER_DATASET_FILENAME = "OER.csv"
OER_TASK_NAME = "oer_demo"
OER_DEFAULT_MAX_EVALUATIONS = 40
OER_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
OER_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / OER_TASK_NAME
OER_TARGET_COLUMN = "Overpotential mV @10 mA cm-2"
OER_CATEGORICAL_FEATURES = ("Metal_1", "Metal_2", "Metal_3")
OER_INTEGER_FEATURES = (
    "Hydrothermal Temp degree",
    "Hydrothermal Time min",
    "Annealing Temp degree",
    "Annealing Time min",
)
OER_FLOAT_FEATURES = (
    "Metal_1_Proportion",
    "Metal_2_Proportion",
    "Metal_3_Proportion",
    "Proton Concentration M",
    "Catalyst_Loading mg cm -2",
)
OER_NUMERICAL_FEATURES = (*OER_FLOAT_FEATURES[:3], *OER_INTEGER_FEATURES, *OER_FLOAT_FEATURES[3:])


def clean_oer_frame(frame: Any):
    """Mirror the tutorial's OER cleaning logic without CV side effects."""

    pd = require_pandas()
    clean = frame.dropna(subset=[OER_TARGET_COLUMN]).copy()
    clean = clean.drop_duplicates()

    for column in clean.columns:
        if clean[column].dtype == "object":
            clean[column] = clean[column].astype(str).str.replace(",", "", regex=False).str.replace('"', "", regex=False)

    for column in OER_CATEGORICAL_FEATURES:
        clean[column] = clean[column].fillna("None")
        clean[column] = clean[column].astype(str).str.strip()
        clean[column] = clean[column].replace(["nan", "NaN", "NA", ""], "None")

    for column in OER_NUMERICAL_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].fillna(0)

    clipped_columns = (
        "Hydrothermal Temp degree",
        "Hydrothermal Time min",
        "Annealing Temp degree",
        "Annealing Time min",
        "Proton Concentration M",
        "Catalyst_Loading mg cm -2",
    )
    for column in clipped_columns:
        q1 = float(clean[column].quantile(0.25))
        q3 = float(clean[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        clean[column] = clean[column].clip(lower=lower, upper=upper)

    clean[OER_TARGET_COLUMN] = pd.to_numeric(clean[OER_TARGET_COLUMN], errors="coerce")
    clean = clean.dropna(subset=[OER_TARGET_COLUMN])
    lower_target = float(clean[OER_TARGET_COLUMN].quantile(0.05))
    upper_target = float(clean[OER_TARGET_COLUMN].quantile(0.95))
    clean = clean[
        (clean[OER_TARGET_COLUMN] >= lower_target)
        & (clean[OER_TARGET_COLUMN] <= upper_target)
    ].reset_index(drop=True)
    return clean


@dataclass
class OerTaskConfig:
    """Configuration for one OER benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OerTask(Task):
    """Task wrapper around the OER tutorial dataset and cleaned mixed search space."""

    def __init__(self, config: OerTaskConfig | None = None):
        self.config = config or OerTaskConfig()
        self._asset = stage_dataset_asset(
            OER_DATASET_RELATIVE_PATH,
            label="OER",
            task_name=OER_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        self._clean_reference_asset = stage_dataset_asset(
            OER_CLEAN_REFERENCE_RELATIVE_PATH,
            label="OER clean reference",
            task_name=OER_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        pd = require_pandas()
        raw_frame = pd.read_csv(self._asset.cache_path)
        self._clean_data = clean_oer_frame(raw_frame)
        feature_frame = self._clean_data.loc[:, [*OER_CATEGORICAL_FEATURES, *OER_NUMERICAL_FEATURES]].copy()
        target = self._clean_data[OER_TARGET_COLUMN]
        encoded, self._train_columns = encode_categorical_frame(feature_frame, OER_CATEGORICAL_FEATURES)
        self._oracle = fit_random_forest_regressor(
            encoded,
            target,
            random_state=42,
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        parameters = [
            CategoricalParam(
                name=column,
                choices=tuple(self._categorical_choices(column)),
            )
            for column in OER_CATEGORICAL_FEATURES
        ]
        parameters.extend(
            [
                FloatParam(name, low=0.0, high=100.0, default=50.0)
                for name in OER_FLOAT_FEATURES[:3]
            ]
        )
        for column in OER_INTEGER_FEATURES:
            low = int(math.floor(float(self._clean_data[column].min())))
            high = int(math.ceil(float(self._clean_data[column].max())))
            default = int(round(float(self._clean_data[column].median())))
            default = min(max(default, low), high)
            parameters.append(
                IntParam(
                    name=column,
                    low=low,
                    high=high,
                    default=default,
                )
            )
        for column in OER_FLOAT_FEATURES[3:]:
            parameters.append(
                FloatParam(
                    name=column,
                    low=float(self._clean_data[column].min()),
                    high=float(self._clean_data[column].max()),
                    default=float(self._clean_data[column].median()),
                )
            )

        search_space = SearchSpace(parameters)
        description_dir = self.config.description_dir or OER_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "clean_reference_path": str(self._clean_reference_asset.cache_path),
            "row_count": int(len(self._clean_data)),
            "column_count": int(len(self._clean_data.columns)),
            "columns": list(self._clean_data.columns),
            "target_stats": numeric_summary(self._clean_data[OER_TARGET_COLUMN]),
            "categorical_choices": {
                column: self._categorical_choices(column)
                for column in OER_CATEGORICAL_FEATURES
            },
            "numeric_bounds": {
                column: {
                    "min": float(self._clean_data[column].min()),
                    "max": float(self._clean_data[column].max()),
                }
                for column in OER_NUMERICAL_FEATURES
            },
        }
        self._spec = TaskSpec(
            name=OER_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("overpotential_mv", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or OER_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(OER_TASK_NAME, description_dir),
            metadata={
                "display_name": "OER Demo",
                "source_paper": OER_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": OER_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "oracle_type": (
                    "RandomForestRegressor("
                    "n_estimators=200, max_depth=15, min_samples_split=5, "
                    "min_samples_leaf=2, random_state=42, n_jobs=-1)"
                ),
                "dimension": len(search_space),
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
        frame = pd.DataFrame([{name: config[name] for name in [*OER_CATEGORICAL_FEATURES, *OER_NUMERICAL_FEATURES]}])
        aligned = align_dummy_columns(frame, OER_CATEGORICAL_FEATURES, self._train_columns)
        predicted = float(self._oracle.predict(aligned)[0])
        elapsed = time.perf_counter() - start
        metrics = {"predicted_overpotential_mv": predicted}
        for name in OER_CATEGORICAL_FEATURES:
            metrics[f"choice::{name}"] = config[name]
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"overpotential_mv": predicted},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata=self._asset.as_metadata(),
        )

    def sanity_check(self):
        report = super().sanity_check()
        if len(self._clean_data) == 0:
            report.add_error("empty_dataset", "OER dataset must contain at least one cleaned row.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["overpotential_mv"])):
                report.add_error("non_finite_prediction", "OER oracle produced a non-finite prediction.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("oracle_predict_failed", f"OER oracle could not score the default config: {exc}")
        report.metadata.update(self._dataset_summary)
        return report

    def _categorical_choices(self, column: str) -> list[str]:
        values = [str(value) for value in self._clean_data[column].unique().tolist()]
        if "None" not in values:
            values.append("None")
        return values


def create_oer_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> OerTask:
    return OerTask(
        OerTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "OER_CATEGORICAL_FEATURES",
    "OER_DATASET_FILENAME",
    "OER_DATASET_RELATIVE_PATH",
    "OER_DEFAULT_MAX_EVALUATIONS",
    "OER_DESCRIPTION_DIR",
    "OER_NUMERICAL_FEATURES",
    "OER_TARGET_COLUMN",
    "OER_TASK_NAME",
    "OerTask",
    "OerTaskConfig",
    "clean_oer_frame",
    "create_oer_task",
]
