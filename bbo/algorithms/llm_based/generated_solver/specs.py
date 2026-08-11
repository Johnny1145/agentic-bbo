"""Serialize SearchSpace into JSON-friendly specs for generated strategies."""

from __future__ import annotations

from typing import Any

from ....core.prompting import search_space_to_schema
from ....core.space import CategoricalParam, FloatParam, IntParam, SearchSpace, StringParam


def parameter_specs_to_search_space(specs: list[dict[str, Any]]) -> SearchSpace:
    """Rebuild a structured ``SearchSpace`` from JSON-ish parameter specs."""

    params: list[FloatParam | IntParam | CategoricalParam] = []
    for spec in specs:
        name = spec["name"]
        typ = spec["type"]
        if typ == "float":
            params.append(
                FloatParam(
                    name=name,
                    low=float(spec["low"]),
                    high=float(spec["high"]),
                    log=bool(spec.get("log", False)),
                    default=spec.get("default"),
                )
            )
        elif typ == "int":
            params.append(
                IntParam(
                    name=name,
                    low=int(spec["low"]),
                    high=int(spec["high"]),
                    log=bool(spec.get("log", False)),
                    default=spec.get("default"),
                )
            )
        elif typ == "categorical":
            params.append(
                CategoricalParam(
                    name=name,
                    choices=tuple(spec["choices"]),
                    default=spec.get("default"),
                )
            )
        elif typ == "string":
            params.append(
                StringParam(
                    name=name,
                    min_length=int(spec.get("min_length", 0)),
                    max_length=None if spec.get("max_length") is None else int(spec["max_length"]),
                    pattern=spec.get("pattern"),
                    default=spec.get("default"),
                )
            )
        else:
            raise TypeError(f"Unsupported parameter type {typ!r} for `{name}`.")
    return SearchSpace(params)


def search_space_to_parameter_specs(space: SearchSpace) -> list[dict[str, Any]]:
    """Convert a SearchSpace to a list of dict specs for suggest_next_config."""
    return search_space_to_schema(space)
