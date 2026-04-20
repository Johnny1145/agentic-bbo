"""HEA scientific benchmark task backed by a random-forest oracle."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
    fit_random_forest_regressor,
    numeric_summary,
    require_openpyxl,
    require_pandas,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
HEA_DATASET_RELATIVE_PATH = "examples/HEA/data/oracle_data.xlsx"
HEA_DATASET_FILENAME = "oracle_data.xlsx"
HEA_TASK_NAME = "hea_demo"
HEA_DEFAULT_MAX_EVALUATIONS = 40
HEA_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
HEA_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / HEA_TASK_NAME
HEA_COMPONENTS = ("Co", "Fe", "Mn", "V", "Cu")
HEA_DESIGN_FEATURES = ("x1", "x2", "x3", "x4")
HEA_TARGET_COLUMN = "target"
HEA_LOWER_BOUNDS = np.asarray([0.05, 0.05, 0.05, 0.05, 0.05], dtype=float)
HEA_UPPER_BOUNDS = np.asarray([0.35, 0.35, 0.35, 0.35, 0.35], dtype=float)
HEA_EPS = 1e-12


def _tail_sums(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = len(lower)
    lower_tail = np.zeros(n_components + 2, dtype=float)
    upper_tail = np.zeros(n_components + 2, dtype=float)
    for index in range(n_components, 0, -1):
        lower_tail[index] = lower_tail[index + 1] + lower[index - 1]
        upper_tail[index] = upper_tail[index + 1] + upper[index - 1]
    return lower_tail, upper_tail


def _phi_inv(design: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_components = len(lower)
    if len(design) != n_components - 1:
        raise ValueError(f"Expected {n_components - 1} design variables, got {len(design)}.")
    lower_tail, upper_tail = _tail_sums(lower, upper)
    raw = np.zeros(n_components, dtype=float)
    remaining = 1.0
    for index in range(1, n_components):
        left = max(lower[index - 1], remaining - upper_tail[index + 1])
        right = min(upper[index - 1], remaining - lower_tail[index + 1])
        if right + HEA_EPS < left:
            raise ValueError(f"Infeasible HEA interval at index {index}: [{left}, {right}]")
        span = right - left
        raw_value = left if span <= HEA_EPS else left + float(design[index - 1]) * span
        raw[index - 1] = raw_value
        remaining -= raw_value
    raw[-1] = remaining
    if not (lower[-1] - HEA_EPS <= raw[-1] <= upper[-1] + HEA_EPS):
        raise ValueError("The final HEA component falls outside the feasible simplex bounds.")
    return raw


def _phi(raw: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_components = len(lower)
    if len(raw) != n_components:
        raise ValueError(f"Expected {n_components} raw variables, got {len(raw)}.")
    lower_tail, upper_tail = _tail_sums(lower, upper)
    design = np.zeros(n_components - 1, dtype=float)
    remaining = 1.0
    for index in range(1, n_components):
        left = max(lower[index - 1], remaining - upper_tail[index + 1])
        right = min(upper[index - 1], remaining - lower_tail[index + 1])
        raw_value = float(raw[index - 1])
        if not (left - HEA_EPS <= raw_value <= right + HEA_EPS):
            raise ValueError(
                f"Raw HEA value at index {index} is outside the feasible interval: {raw_value} not in [{left}, {right}]"
            )
        span = right - left
        design[index - 1] = 0.0 if span <= HEA_EPS else (raw_value - left) / span
        remaining -= raw_value
    if not (abs(float(raw[-1]) - remaining) <= 1e-9):
        raise ValueError("HEA inverse transform failed the simplex conservation check.")
    return design


def design_to_raw_frame(frame: Any):
    pd = require_pandas()
    if list(frame.columns) != list(HEA_DESIGN_FEATURES):
        raise ValueError(f"HEA design columns must be ordered as {list(HEA_DESIGN_FEATURES)!r}.")
    rows = [_phi_inv(np.asarray(row, dtype=float), HEA_LOWER_BOUNDS, HEA_UPPER_BOUNDS) for row in frame.to_numpy()]
    return pd.DataFrame(rows, columns=list(HEA_COMPONENTS))


def raw_to_design_frame(frame: Any):
    pd = require_pandas()
    if list(frame.columns) != list(HEA_COMPONENTS):
        raise ValueError(f"HEA raw columns must be ordered as {list(HEA_COMPONENTS)!r}.")
    rows = [_phi(np.asarray(row, dtype=float), HEA_LOWER_BOUNDS, HEA_UPPER_BOUNDS) for row in frame.to_numpy()]
    return pd.DataFrame(rows, columns=list(HEA_DESIGN_FEATURES))


@dataclass
class HeaTaskConfig:
    """Configuration for one HEA benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HeaTask(Task):
    """Task wrapper around the HEA tutorial dataset and transform-aware oracle."""

    def __init__(self, config: HeaTaskConfig | None = None):
        self.config = config or HeaTaskConfig()
        self._asset = stage_dataset_asset(
            HEA_DATASET_RELATIVE_PATH,
            label="HEA",
            task_name=HEA_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        require_openpyxl()
        pd = require_pandas()
        raw_frame = pd.read_excel(self._asset.cache_path)
        self._raw_data = raw_frame.loc[:, [*HEA_COMPONENTS, HEA_TARGET_COLUMN]].copy()
        missing = [name for name in [*HEA_COMPONENTS, HEA_TARGET_COLUMN] if name not in raw_frame.columns]
        if missing:
            raise ValueError(f"HEA dataset is missing required columns: {missing!r}")

        self._target_max = float(self._raw_data[HEA_TARGET_COLUMN].max())
        self._oracle = fit_random_forest_regressor(
            self._raw_data.loc[:, list(HEA_COMPONENTS)],
            self._raw_data[HEA_TARGET_COLUMN],
            random_state=self.config.seed,
            n_estimators=100,
        )
        self._design_projection = raw_to_design_frame(self._raw_data.loc[:, list(HEA_COMPONENTS)])
        search_space = SearchSpace(
            [
                FloatParam(
                    name,
                    low=0.0,
                    high=1.0,
                    default=float(self._design_projection[name].median()),
                )
                for name in HEA_DESIGN_FEATURES
            ]
        )
        description_dir = self.config.description_dir or HEA_DESCRIPTION_DIR
        transformed = design_to_raw_frame(self._design_projection)
        transform_residual = float(np.abs(transformed.to_numpy() - self._raw_data.loc[:, list(HEA_COMPONENTS)].to_numpy()).max())
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "row_count": int(len(self._raw_data)),
            "column_count": int(len(self._raw_data.columns)),
            "columns": list(self._raw_data.columns),
            "target_stats": numeric_summary(self._raw_data[HEA_TARGET_COLUMN]),
            "design_bounds": {
                name: {
                    "min": float(self._design_projection[name].min()),
                    "max": float(self._design_projection[name].max()),
                }
                for name in HEA_DESIGN_FEATURES
            },
            "component_bounds": {
                name: {
                    "min": float(self._raw_data[name].min()),
                    "max": float(self._raw_data[name].max()),
                }
                for name in HEA_COMPONENTS
            },
            "transform_residual_max": transform_residual,
        }
        self._spec = TaskSpec(
            name=HEA_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("regret", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or HEA_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(HEA_TASK_NAME, description_dir),
            metadata={
                "display_name": "HEA Demo",
                "source_paper": HEA_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": HEA_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "oracle_type": "RandomForestRegressor(n_estimators=100, random_state=<seed>)",
                "dimension": len(HEA_DESIGN_FEATURES),
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
        design_frame = pd.DataFrame([[config[name] for name in HEA_DESIGN_FEATURES]], columns=HEA_DESIGN_FEATURES)
        raw_frame = design_to_raw_frame(design_frame)
        predicted_target = float(self._oracle.predict(raw_frame)[0])
        predicted_regret = self._target_max - predicted_target
        elapsed = time.perf_counter() - start

        metrics = {"predicted_target": predicted_target}
        for component in HEA_COMPONENTS:
            metrics[f"composition::{component}"] = float(raw_frame.iloc[0][component])

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"regret": predicted_regret},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata=self._asset.as_metadata(),
        )

    def sanity_check(self):
        report = super().sanity_check()
        if len(self._raw_data) == 0:
            report.add_error("empty_dataset", "HEA dataset must contain at least one row.")
        component_sums = self._raw_data.loc[:, list(HEA_COMPONENTS)].sum(axis=1)
        if not bool(np.allclose(component_sums.to_numpy(), 1.0, atol=1e-6)):
            report.add_error("invalid_simplex", "HEA compositions must sum to approximately 1.0.")
        within_bounds = (
            (self._raw_data.loc[:, list(HEA_COMPONENTS)].to_numpy() >= HEA_LOWER_BOUNDS - 1e-9).all()
            and (self._raw_data.loc[:, list(HEA_COMPONENTS)].to_numpy() <= HEA_UPPER_BOUNDS + 1e-9).all()
        )
        if not within_bounds:
            report.add_error("invalid_component_bounds", "HEA compositions fall outside the expected [0.05, 0.35] range.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["regret"])):
                report.add_error("non_finite_prediction", "HEA oracle produced a non-finite prediction.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("oracle_predict_failed", f"HEA oracle could not score the default config: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def create_hea_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> HeaTask:
    return HeaTask(
        HeaTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "HEA_COMPONENTS",
    "HEA_DATASET_FILENAME",
    "HEA_DATASET_RELATIVE_PATH",
    "HEA_DEFAULT_MAX_EVALUATIONS",
    "HEA_DESIGN_FEATURES",
    "HEA_DESCRIPTION_DIR",
    "HEA_TASK_NAME",
    "HeaTask",
    "HeaTaskConfig",
    "create_hea_task",
    "design_to_raw_frame",
    "raw_to_design_frame",
]
