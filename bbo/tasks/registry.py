"""Task registries and convenience constructors."""

from __future__ import annotations

from ..core import Task
from .dbtune.cli_http_surrogate import (
    DBTUNE_SURROGATE_SERVICE_TASK_NAMES,
    create_dbtune_surrogate_service_task_for_registry,
    dbtune_surrogate_service_registry_entries,
)
from .dbtune import SURROGATE_BENCHMARKS
from .dbtune.cli_offline_surrogate import (
    INPROC_SURROGATE_TASK_NAMES,
    create_inproc_surrogate_task_for_registry,
    inproc_surrogate_registry_entries,
)
from .dbtune.http_surrogate_specs import DBTUNE_SURROGATE_SERVICE_TASK_IDS, HTTP_SURROGATE_TASK_IDS
from .bboplace import BBOPLACE_TASK_KEY, create_bboplace_task
from .hpo import HPO_TASK_IDS, create_hpo_task
from .scientific import SCIENTIFIC_TASK_REGISTRY, create_scientific_task
from .synthetic import (
    BBOB_PROBLEM_REGISTRY,
    BBOB_TASK_IDS,
    BbobProblemDefinition,
    create_bbob_task,
)

# Compatibility name retained for callers; v4 contains BBOB entries only.
SYNTHETIC_PROBLEM_REGISTRY: dict[str, BbobProblemDefinition] = dict(BBOB_PROBLEM_REGISTRY)
TASK_REGISTRY: dict[str, str] = {
    **{name: "synthetic" for name in SYNTHETIC_PROBLEM_REGISTRY},
    **{name: "scientific" for name in SCIENTIFIC_TASK_REGISTRY},
    **inproc_surrogate_registry_entries(),
    **dbtune_surrogate_service_registry_entries(),
    BBOPLACE_TASK_KEY: "bboplace",
    **{name: "hpo" for name in HPO_TASK_IDS},
}
ALL_TASK_NAMES: tuple[str, ...] = tuple(sorted(TASK_REGISTRY))

SURROGATE_TASK_IDS: tuple[str, ...] = tuple(sorted(SURROGATE_BENCHMARKS))

TASK_FAMILIES: dict[str, tuple[str, ...]] = {
    "scientific": tuple(sorted(SCIENTIFIC_TASK_REGISTRY)),
    "synthetic": BBOB_TASK_IDS,
    "dbtune_surrogate_service": tuple(sorted(DBTUNE_SURROGATE_SERVICE_TASK_NAMES)),
    "bboplace": (BBOPLACE_TASK_KEY,),
    "hpo": tuple(sorted(HPO_TASK_IDS)),
}

ALL_DEMO_TASK_NAMES: tuple[str, ...] = tuple(
    sorted([*BBOB_TASK_IDS, BBOPLACE_TASK_KEY]),
)


def get_synthetic_problem(name: str) -> BbobProblemDefinition:
    if name not in SYNTHETIC_PROBLEM_REGISTRY:
        available = ", ".join(sorted(SYNTHETIC_PROBLEM_REGISTRY))
        raise ValueError(f"Unknown synthetic problem `{name}`. Available: {available}")
    return SYNTHETIC_PROBLEM_REGISTRY[name]


def create_demo_task(
    problem: str = "bbob_f01_d10",
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
    **kwargs,
) -> Task:
    if problem in BBOB_PROBLEM_REGISTRY:
        if noise_std != 0.0:
            raise ValueError("Official COCO/BBOB tasks do not support injected observation noise.")
        return create_bbob_task(
            problem,
            max_evaluations=max_evaluations,
            seed=seed,
            **kwargs,
        )
    if problem in SCIENTIFIC_TASK_REGISTRY:
        return create_scientific_task(
            problem,
            max_evaluations=max_evaluations,
            seed=seed,
            **kwargs,
        )
    if problem in INPROC_SURROGATE_TASK_NAMES:
        return create_inproc_surrogate_task_for_registry(
            problem,
            max_evaluations=max_evaluations,
            seed=seed,
            noise_std=noise_std,
            **kwargs,
        )
    if problem in DBTUNE_SURROGATE_SERVICE_TASK_NAMES:
        return create_dbtune_surrogate_service_task_for_registry(
            problem,
            max_evaluations=max_evaluations,
            seed=seed,
            noise_std=noise_std,
            **kwargs,
        )
    if problem == BBOPLACE_TASK_KEY:
        return create_bboplace_task(
            max_evaluations=max_evaluations,
            seed=seed,
            **kwargs,
        )
    if problem in HPO_TASK_IDS:
        return create_hpo_task(
            problem,
            max_evaluations=max_evaluations,
            seed=seed,
            **kwargs,
        )
    available = ", ".join(ALL_TASK_NAMES)
    raise ValueError(f"Unknown task `{problem}`. Available: {available}")


def create_task(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
    **kwargs,
) -> Task:
    return create_demo_task(
        problem=name,
        max_evaluations=max_evaluations,
        seed=seed,
        noise_std=noise_std,
        **kwargs,
    )


def get_scientific_task(name: str) -> str:
    if name not in SCIENTIFIC_TASK_REGISTRY:
        available = ", ".join(sorted(SCIENTIFIC_TASK_REGISTRY))
        raise ValueError(f"Unknown scientific task `{name}`. Available: {available}")
    return SCIENTIFIC_TASK_REGISTRY[name]


__all__ = [
    "ALL_DEMO_TASK_NAMES",
    "BBOPLACE_TASK_KEY",
    "BBOB_PROBLEM_REGISTRY",
    "BBOB_TASK_IDS",
    "ALL_TASK_NAMES",
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "HPO_TASK_IDS",
    "SURROGATE_TASK_IDS",
    "SCIENTIFIC_TASK_REGISTRY",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "TASK_REGISTRY",
    "create_demo_task",
    "create_task",
    "get_scientific_task",
    "get_synthetic_problem",
]
