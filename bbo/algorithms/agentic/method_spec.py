"""Single declarative source of truth for agentic methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ...core import Algorithm
from .components import RoleSpec
from .prompt_profiles import PABLO_WORKFLOW_PROMPT_PROFILE, WorkflowPromptProfile, resolve_prompt_profile
from .evented_algorithm import EventMapper, EventedAlgorithm

AgenticFactory = Callable[..., Algorithm]


def _agentic_bo_factory(**kwargs: Any) -> Algorithm:
    from .agentic_bo import create_agentic_bo
    return create_agentic_bo(**kwargs)


def _pablo_factory(**kwargs: Any) -> Algorithm:
    from .pablo import PabloAlgorithm
    return PabloAlgorithm(**kwargs)


def _llambo_factory(**kwargs: Any) -> Algorithm:
    from ..llm_based import LlamboAlgorithm
    kwargs.pop("run_dir", None)
    return LlamboAlgorithm(**kwargs)


def _agent_metadata(metadata: Mapping[str, Any]):
    return {"source": metadata.get("agent_source")}, []


def _llambo_metadata(metadata: Mapping[str, Any]):
    return {"phase": metadata.get("llambo_phase"),
            "backend": metadata.get("llambo_backend"),
            "candidate_count": metadata.get("llambo_candidate_count")}, []


def _pablo_metadata(metadata: Mapping[str, Any]):
    role = metadata.get("pablo_role")
    events = [] if not role else [("role_call", {"role": role, "source": metadata.get("pablo_source")})]
    return {"role": role, "source": metadata.get("pablo_source"),
            "round": metadata.get("pablo_round")}, events


@dataclass(frozen=True)
class AgenticMethodSpec:
    name: str
    factory: AgenticFactory
    controller: str
    capabilities: frozenset[str]
    requires_agent_runtime: bool
    roles: tuple[RoleSpec, ...] = ()
    aliases: tuple[str, ...] = ()
    supports_optimizer_tools: bool = False
    supports_multiple_roles: bool = False
    supports_resume: bool = True
    max_objectives: int = 1
    event_mapper: EventMapper | None = field(default=None, repr=False, compare=False)
    prompt_profile: WorkflowPromptProfile | None = None

    def __post_init__(self) -> None:
        role_names = {role.name for role in self.roles}
        if self.prompt_profile is not None:
            self.prompt_profile.validate_roles(role_names)


AGENTIC_BO_WORKFLOW_PROMPTS = WorkflowPromptProfile.single(
    "agentic_bo", "proposer", resolve_prompt_profile("agentic_bo")
)


AGENTIC_METHOD_REGISTRY: dict[str, AgenticMethodSpec] = {
    "agentic_bo": AgenticMethodSpec(
        name="agentic_bo", factory=_agentic_bo_factory, controller="single_agent",
        roles=(RoleSpec(name="proposer", model_route="global", tool_profile="agentic_bo", prompt_profile="agentic_bo"),),
        capabilities=frozenset({"probe", "propose", "reconfigure", "commit"}),
        requires_agent_runtime=True, supports_optimizer_tools=True, event_mapper=_agent_metadata,
        prompt_profile=AGENTIC_BO_WORKFLOW_PROMPTS),
    "llambo": AgenticMethodSpec(
        name="llambo", factory=_llambo_factory, controller="prompt_surrogate_acquisition",
        capabilities=frozenset({"probe", "propose", "commit"}),
        requires_agent_runtime=False, event_mapper=_llambo_metadata),
    "pablo": AgenticMethodSpec(
        name="pablo", factory=_pablo_factory, aliases=("palbo",), controller="hierarchical_roles",
        roles=(RoleSpec(name="planner", model_route="planner", prompt_profile="pablo.planner"),
               RoleSpec(name="explorer", model_route="explorer", prompt_profile="pablo.explorer"),
               RoleSpec(name="worker", model_route="worker", prompt_profile="pablo.worker")),
        capabilities=frozenset({"role_call", "propose", "commit"}),
        requires_agent_runtime=False, supports_multiple_roles=True, event_mapper=_pablo_metadata,
        prompt_profile=PABLO_WORKFLOW_PROMPT_PROFILE),
}

_AGENTIC_ALIASES = {alias: spec.name for spec in AGENTIC_METHOD_REGISTRY.values() for alias in spec.aliases}


def get_agentic_method_spec(name: str) -> AgenticMethodSpec:
    canonical = _AGENTIC_ALIASES.get(name, name)
    try:
        return AGENTIC_METHOD_REGISTRY[canonical]
    except KeyError as exc:
        available = ", ".join(sorted((*AGENTIC_METHOD_REGISTRY, *_AGENTIC_ALIASES)))
        raise KeyError(f"Unknown agentic method {name!r}; available: {available}") from exc


def create_agentic_method(name: str, *, run_dir: Path | str | None = None, **kwargs: Any) -> Algorithm:
    spec = get_agentic_method_spec(name)
    if run_dir is not None:
        kwargs.setdefault("run_dir", run_dir)
    algorithm = spec.factory(**kwargs)
    return EventedAlgorithm(algorithm, method=spec.name, run_dir=run_dir, event_mapper=spec.event_mapper)


__all__ = ["AGENTIC_METHOD_REGISTRY", "AgenticMethodSpec", "create_agentic_method", "get_agentic_method_spec"]
