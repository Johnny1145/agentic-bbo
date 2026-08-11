"""First-class Agentic BO method assembled from shared agent/tool layers."""

from __future__ import annotations

from typing import Any

from .components import RoleSpec, SingleAgentMethod, ToolProfile
from .general_agent import GeneralAgentBBOAlgorithm


AGENTIC_BO_TOOL_NAMES = (
    "get_trial_history",
    "validate_candidate",
    "optimizer_suggest",
    "optimizer_predict",
    "optimizer_score",
    "optimizer_diagnostics",
    "optimizer_status",
    "optimizer_set_backend",
    "optimizer_set_bounds",
    "optimizer_set_acquisition",
    "optimizer_reset_policy",
)

AGENTIC_BO_COMPONENT = SingleAgentMethod(
    name="agentic_bo",
    role=RoleSpec(name="proposer", model_route="global", tool_profile="agentic_bo"),
    tools=ToolProfile(
        name="agentic_bo",
        tools=AGENTIC_BO_TOOL_NAMES,
        optimizer_backends=("gp_ei",),
        require_optimizer_decision=True,
    ),
    prompt_profile="agentic_bo",
)


def create_agentic_bo(**kwargs: Any) -> GeneralAgentBBOAlgorithm:
    """Build Agentic BO from reusable role and tool-profile components."""

    optimizer_backend = str(kwargs.pop("optimizer_backend", "gp_ei"))
    if not kwargs.get("optimizer_backend_allowlist"):
        kwargs["optimizer_backend_allowlist"] = (optimizer_backend,)
    kwargs.setdefault("experiment_condition", "agentic_bo")
    return AGENTIC_BO_COMPONENT.build(**kwargs)


class AgenticBOAlgorithm(GeneralAgentBBOAlgorithm):
    """Agent-controlled GP BO with probe/propose/reconfigure/commit actions.

    This is the reusable method hidden inside workflow 58's T3 condition. The
    workflow may still add analysis tools for ablations, while this entrypoint
    provides the minimal paper-level optimizer control surface.
    """

    def __init__(
        self,
        *,
        framework: str = "nanobot",
        optimizer_backend: str = "gp_ei",
        optimizer_max_calls_per_round: int = 4,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("algorithm_name", "agentic_bo")
        kwargs.setdefault("tool_mode", "function_calling")

        if kwargs.get("enabled_tool_names") is None:
            kwargs["enabled_tool_names"] = AGENTIC_BO_TOOL_NAMES
        if not kwargs.get("optimizer_backend_allowlist"):
            kwargs["optimizer_backend_allowlist"] = (optimizer_backend,)
        kwargs["require_optimizer_decision_per_round"] = True
        kwargs.setdefault("experiment_condition", "agentic_bo")
        super().__init__(
            framework=framework,
            optimizer_max_calls_per_round=optimizer_max_calls_per_round,
            **kwargs,
        )


__all__ = ["AGENTIC_BO_COMPONENT", "AGENTIC_BO_TOOL_NAMES", "AgenticBOAlgorithm", "create_agentic_bo"]
