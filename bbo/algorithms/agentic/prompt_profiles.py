"""Composable prompt profiles for agentic benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptProfile:
    """Method-owned prompt additions layered over the benchmark protocol."""

    name: str
    protocol_instructions: str = ""
    round_instructions: str = ""

    def compose(self, base: str, *, stage: str) -> str:
        addition = self.protocol_instructions if stage == "protocol" else self.round_instructions
        addition = addition.strip()
        return base.rstrip() if not addition else f"{base.rstrip()}\n\n# Method profile: {self.name}\n{addition}"
    def compose_bundle(self, bundle: Any) -> Any:
        """Add this role's round instructions to a PromptBundle-like value."""
        return replace(bundle, user=self.compose(bundle.user, stage="round"))


@dataclass(frozen=True)
class WorkflowPromptProfile:
    """Complete role-to-prompt contract for a single- or multi-agent method."""

    name: str
    roles: Mapping[str, PromptProfile]

    def __post_init__(self) -> None:
        normalized = {str(role).strip(): profile for role, profile in self.roles.items()}
        if not normalized or any(not role for role in normalized):
            raise ValueError("WorkflowPromptProfile requires at least one named role.")
        object.__setattr__(self, "roles", normalized)

    def for_role(self, role: str) -> PromptProfile:
        try:
            return self.roles[role]
        except KeyError as exc:
            available = ", ".join(sorted(self.roles))
            raise ValueError(
                f"Workflow prompt profile {self.name!r} has no role {role!r}; available: {available}"
            ) from exc

    def validate_roles(self, roles: set[str]) -> None:
        configured = set(self.roles)
        if configured != roles:
            missing = sorted(roles - configured)
            extra = sorted(configured - roles)
            raise ValueError(
                f"Workflow prompt roles do not match method roles; missing={missing}, extra={extra}."
            )

    @classmethod
    def single(cls, name: str, role: str, profile: PromptProfile) -> "WorkflowPromptProfile":
        return cls(name=name, roles={role: profile})


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "general_bbo": PromptProfile(name="general_bbo"),
    "native_harness": PromptProfile(
        name="native_harness",
        protocol_instructions=(
            "Use the harness's native reasoning, file, and shell capabilities. "
            "The benchmark protocol constrains only the final candidate handoff; "
            "it does not prescribe a search workflow."
        ),
        round_instructions=(
            "Choose the next candidate using the native harness and currently visible evidence."
        ),
    ),
    "native_tools": PromptProfile(
        name="native_tools",
        protocol_instructions=(
            "Use only the tools exposed by this run's ToolProfile. Tool availability is a "
            "capability boundary, not a mandatory sequence of calls."
        ),
        round_instructions=(
            "Select tool calls based on the decision needed this round, then commit exactly one candidate."
        ),
    ),
    "agentic_bo": PromptProfile(
        name="agentic_bo",
        protocol_instructions=(
            "Act as the decision maker in a surrogate-assisted optimization loop; the optimizer "
            "is an uncertainty-aware instrument, not an autopilot. Preserve the benchmark's "
            "candidate-submission protocol. Inspect optimizer state and trial evidence, optionally "
            "probe or reconfigure the optimizer, request or score proposals, then commit exactly "
            "one candidate. Never record predictions as observations, fabricate outcomes, or "
            "update optimizer history directly. Adapt the opening strategy to prior strength: test "
            "a concrete hypothesis when the task gives a strong prior, focus bounds when only a "
            "region is credible, and favor space-filling exploration when no useful prior exists. "
            "Use diagnostics to decide how much to trust the surrogate; do not discard informative "
            "task context merely because it conflicts with an uncertain posterior."
        ),
        round_instructions=(
            "Interpret the newest real observation before choosing the next evaluation. Before "
            "submission, state your current belief and what this evaluation should learn. Make an "
            "explicit optimizer decision and record whether the committed candidate adopts, "
            "refines, overrides, or directly scores an optimizer proposal."
        ),
    ),
}


def resolve_prompt_profile(profile: str | PromptProfile | None) -> PromptProfile:
    if isinstance(profile, PromptProfile):
        return profile
    name = "general_bbo" if profile is None else str(profile).strip().lower().replace("-", "_")
    try:
        return PROMPT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_PROFILES))
        raise ValueError(f"Unknown prompt profile {name!r}; available: {available}") from exc


PABLO_WORKFLOW_PROMPT_PROFILE = WorkflowPromptProfile(
    name="pablo",
    roles={
        "planner": PromptProfile(
            name="pablo.planner",
            round_instructions="Return bounded, distinct search tasks for the Worker roles.",
        ),
        "explorer": PromptProfile(
            name="pablo.explorer",
            round_instructions="Make one globally exploratory proposal from c_global evidence.",
        ),
        "worker": PromptProfile(
            name="pablo.worker",
            round_instructions="Refine only the assigned task and current seed for this feedback step.",
        ),
    },
)


def resolve_workflow_prompt_profile(
    profile: WorkflowPromptProfile | PromptProfile | str,
    *,
    roles: set[str],
    single_role: str | None = None,
) -> WorkflowPromptProfile:
    if isinstance(profile, WorkflowPromptProfile):
        resolved = profile
    else:
        atomic = resolve_prompt_profile(profile)
        if len(roles) != 1:
            raise ValueError(
                "A single PromptProfile cannot configure a multi-role workflow; "
                "provide WorkflowPromptProfile with one profile per role."
            )
        role = single_role or next(iter(roles))
        resolved = WorkflowPromptProfile.single(atomic.name, role, atomic)
    resolved.validate_roles(roles)
    return resolved
__all__ = ["PABLO_WORKFLOW_PROMPT_PROFILE", "PROMPT_PROFILES", "PromptProfile", "WorkflowPromptProfile", "resolve_prompt_profile", "resolve_workflow_prompt_profile"]
