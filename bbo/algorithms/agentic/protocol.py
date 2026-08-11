"""Runtime-independent protocol for agentic optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeAlias

from ...core import Incumbent, TaskDescriptionBundle, TaskSpec, TrialObservation


@dataclass(frozen=True)
class OptimizationContext:
    """Immutable view supplied to one policy deliberation."""

    task_spec: TaskSpec
    history: Sequence[TrialObservation]
    seed: int
    evaluation_index: int
    description: TaskDescriptionBundle | None = None
    incumbent: Incumbent | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitCandidate:
    """Commit exactly one candidate to the expensive evaluator."""

    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    budget: float | None = None


@dataclass(frozen=True)
class StopOptimization:
    """A policy request to stop before exhausting the outer budget."""

    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


PolicyDecision: TypeAlias = CommitCandidate | StopOptimization


class AgenticPolicy(Protocol):
    """Minimum extension point required to add an agentic BBO method."""

    @property
    def name(self) -> str: ...

    def setup(self, context: OptimizationContext) -> None: ...

    def deliberate(self, context: OptimizationContext) -> PolicyDecision: ...

    def observe(self, observation: TrialObservation) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...


__all__ = [
    "AgenticPolicy",
    "CommitCandidate",
    "OptimizationContext",
    "PolicyDecision",
    "StopOptimization",
]
