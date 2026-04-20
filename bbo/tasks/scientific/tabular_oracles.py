"""Shared tabular-data helpers for scientific benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureSelectionSummary:
    """Summary of feature importance filtering for the BH task."""

    selected_columns: tuple[str, ...]
    importances: dict[str, float]


def _dependency_error(packages: Sequence[str]) -> ImportError:
    joined = ", ".join(packages)
    return ImportError(
        "Scientific tutorial tasks require optional dependencies "
        f"({joined}). Install them with `uv sync --extra dev --extra bo-tutorial`."
    )


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise _dependency_error(("pandas",)) from exc
    return pd


def require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise _dependency_error(("openpyxl",)) from exc


def fit_random_forest_regressor(
    features: Any,
    target: Any,
    *,
    random_state: int,
    **kwargs: Any,
):
    """Fit a random-forest regressor with a clear optional-dependency error."""

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise _dependency_error(("scikit-learn",)) from exc

    model = RandomForestRegressor(random_state=random_state, **kwargs)
    model.fit(features, target)
    return model


def encode_categorical_frame(frame: Any, categorical_columns: Sequence[str]):
    """One-hot encode categorical columns with pandas.get_dummies."""

    pd = require_pandas()
    encoded = pd.get_dummies(frame, columns=list(categorical_columns))
    return encoded, tuple(encoded.columns)


def align_dummy_columns(frame: Any, categorical_columns: Sequence[str], train_columns: Sequence[str]):
    """One-hot encode and align feature columns to the training matrix."""

    encoded, _ = encode_categorical_frame(frame, categorical_columns)
    return encoded.reindex(columns=list(train_columns), fill_value=0)


def numeric_summary(values: Any) -> dict[str, float]:
    """Return min/max/mean/std for a numeric sequence."""

    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def select_feature_columns_by_importance(
    features: Any,
    target: Any,
    *,
    extractor: str = "random_forest",
    max_n: int = -1,
    max_cum_imp: float = 0.0,
    min_imp: float = 0.0,
    random_state: int = 0,
) -> FeatureSelectionSummary:
    """Mirror the tutorial's BH feature-selection pipeline."""

    if extractor != "random_forest":
        raise ValueError(f"Unsupported feature selector `{extractor}`.")

    model = fit_random_forest_regressor(
        features,
        target,
        random_state=random_state,
        n_estimators=100,
    )
    importances = np.asarray(model.feature_importances_, dtype=float)
    indices = np.argsort(importances)[::-1]

    if max_n > 0:
        indices = indices[:max_n]
    if max_cum_imp > 0:
        cumulative = np.cumsum(importances[indices])
        indices = indices[cumulative <= max_cum_imp]
    if min_imp > 0:
        indices = indices[importances[indices] >= min_imp]
    if len(indices) == 0:
        raise ValueError("Feature selection removed every BH feature; relax the thresholds.")

    selected_columns = tuple(str(features.columns[index]) for index in indices)
    return FeatureSelectionSummary(
        selected_columns=selected_columns,
        importances={str(features.columns[index]): float(importances[index]) for index in indices},
    )


__all__ = [
    "FeatureSelectionSummary",
    "align_dummy_columns",
    "encode_categorical_frame",
    "fit_random_forest_regressor",
    "numeric_summary",
    "require_openpyxl",
    "require_pandas",
    "select_feature_columns_by_importance",
]
