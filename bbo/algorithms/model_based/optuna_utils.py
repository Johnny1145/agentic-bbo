"""Helpers for mapping the benchmark search space onto Optuna primitives."""

from __future__ import annotations

from typing import Any

from ...core import CategoricalParam, FloatParam, IntParam, ObjectiveDirection, SearchSpace


def require_optuna() -> Any:
    """Import Optuna lazily so the base install stays lightweight."""

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "`optuna_tpe` requires the optional Optuna dependency. "
            "Install it with `uv sync --extra optuna` or `uv sync --extra dev --extra optuna`."
        ) from exc
    return optuna


def objective_direction_to_optuna(direction: ObjectiveDirection) -> str:
    """Translate the core objective direction enum into an Optuna study direction."""

    if direction == ObjectiveDirection.MAXIMIZE:
        return "maximize"
    return "minimize"


def suggest_from_param(trial: Any, param: FloatParam | IntParam | CategoricalParam) -> Any:
    """Request one Optuna suggestion matching a structured benchmark parameter."""

    if isinstance(param, FloatParam):
        return trial.suggest_float(param.name, param.low, param.high, log=param.log)
    if isinstance(param, IntParam):
        return trial.suggest_int(param.name, param.low, param.high, log=param.log)
    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(param.name, tuple(param.choices))
    raise TypeError(f"Unsupported parameter type: {type(param).__name__}")


def build_distributions(search_space: SearchSpace) -> dict[str, Any]:
    """Materialize Optuna distributions for tests and fixed-space validation."""

    optuna = require_optuna()
    distributions: dict[str, Any] = {}
    for param in search_space:
        if isinstance(param, FloatParam):
            distributions[param.name] = optuna.distributions.FloatDistribution(
                low=param.low,
                high=param.high,
                log=param.log,
            )
        elif isinstance(param, IntParam):
            distributions[param.name] = optuna.distributions.IntDistribution(
                low=param.low,
                high=param.high,
                log=param.log,
            )
        elif isinstance(param, CategoricalParam):
            distributions[param.name] = optuna.distributions.CategoricalDistribution(
                choices=tuple(param.choices),
            )
        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
    return distributions


__all__ = [
    "build_distributions",
    "objective_direction_to_optuna",
    "require_optuna",
    "suggest_from_param",
]
