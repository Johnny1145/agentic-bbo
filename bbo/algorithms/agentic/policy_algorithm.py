"""Adapter that runs any AgenticPolicy in the existing ask/tell core."""

from __future__ import annotations

from typing import Any

from ...core import Algorithm, Incumbent, TaskDescriptionBundle, TaskSpec, TrialObservation, TrialSuggestion
from .protocol import AgenticPolicy, CommitCandidate, OptimizationContext, StopOptimization


class PolicyStopped(RuntimeError):
    """Raised when a policy asks an outer loop without stop support to stop."""


class AgenticPolicyAlgorithm(Algorithm):
    """Stable bridge from a lightweight policy to the benchmark kernel."""

    def __init__(self, policy: AgenticPolicy) -> None:
        self.policy = policy
        self._task_spec: TaskSpec | None = None
        self._description: TaskDescriptionBundle | None = None
        self._history: list[TrialObservation] = []
        self._seed = 0

    @property
    def name(self) -> str:
        return self.policy.name

    @property
    def artifact_paths(self) -> dict[str, str]:
        return dict(getattr(self.policy, "artifact_paths", {}))

    @property
    def routing_table(self) -> dict[str, str]:
        return dict(getattr(self.policy, "routing_table", {}))

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        self._task_spec = task_spec
        self._seed = int(seed)
        description = kwargs.get("task_description")
        self._description = description if isinstance(description, TaskDescriptionBundle) else None
        self._history = []
        self.policy.setup(self._context())

    def ask(self) -> TrialSuggestion:
        decision = self.policy.deliberate(self._context())
        if isinstance(decision, StopOptimization):
            raise PolicyStopped(decision.reason)
        if not isinstance(decision, CommitCandidate):
            raise TypeError(f"Policy returned unsupported decision {type(decision).__name__}")
        assert self._task_spec is not None
        config = self._task_spec.search_space.coerce_config(decision.config, use_defaults=False)
        return TrialSuggestion(config=config, budget=decision.budget, metadata=dict(decision.metadata))

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self.policy.observe(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        self._history = []
        for observation in history:
            self.tell(observation)

    def incumbents(self) -> list[Incumbent]:
        incumbents = getattr(self.policy, "incumbents", None)
        return list(incumbents()) if callable(incumbents) else []

    def _context(self) -> OptimizationContext:
        if self._task_spec is None:
            raise RuntimeError("AgenticPolicyAlgorithm.setup() must be called first")
        return OptimizationContext(
            task_spec=self._task_spec,
            history=tuple(self._history),
            seed=self._seed,
            evaluation_index=len(self._history),
            description=self._description,
        )


__all__ = ["AgenticPolicyAlgorithm", "PolicyStopped"]
