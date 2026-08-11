"""Algorithm registry grouped by algorithm family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core.algo import Algorithm
from .agentic import (
    AgenticBOAlgorithm,
    create_agentic_method,
    ClaudeCodeBBOAlgorithm,
    CodexBBOAlgorithm,
    NanobotBBOAlgorithm,
    OpenAICompatibleBBOAlgorithm,
    PabloAlgorithm,
)
from .llm_based import LlamboAlgorithm, OproAlgorithm
from .llm_based.skydiscover_interleaved import SkydiscoverInterleavedAlgorithm
from .model_based import (
    BotorchTurboAlgorithm,
    CustomPfnsBoAlgorithm,
    GitBoAlgorithm,
    GpEiAlgorithm,
    OptunaTpeAlgorithm,
    Pfns4BoAlgorithm,
    TabPfnV2BoAlgorithm,
)
from .molecular import GraphGAAlgorithm, GraphGPBOAlgorithm
from .traditional import (
    LocalPerturbationAlgorithm,
    PyCmaAlgorithm,
    RandomSearchAlgorithm,
    SobolSearchAlgorithm,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata for one algorithm entrypoint."""

    factory: Callable[..., Algorithm]
    description: str
    family: str
    numeric_only: bool = False
    categorical_to_continuous: str | None = None
def _create_agentic_bo(**kwargs: Any) -> Algorithm:
    return create_agentic_method("agentic_bo", **kwargs)


def _create_llambo(**kwargs: Any) -> Algorithm:
    return create_agentic_method("llambo", **kwargs)


def _create_pablo(**kwargs: Any) -> Algorithm:
    return create_agentic_method("pablo", **kwargs)




ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "random_search": AlgorithmSpec(
        factory=RandomSearchAlgorithm,
        description="Uniform random search over the declared search space.",
        family="traditional",
    ),
    "random": AlgorithmSpec(
        factory=RandomSearchAlgorithm,
        description="Alias for random_search.",
        family="traditional",
    ),
    "local_perturbation": AlgorithmSpec(
        factory=LocalPerturbationAlgorithm,
        description="Incumbent-centred deterministic local perturbation search.",
        family="traditional",
    ),
    "pycma": AlgorithmSpec(
        factory=PyCmaAlgorithm,
        description="CMA-ES via the external `pycma` package.",
        family="traditional",
        numeric_only=True,
        categorical_to_continuous="onehot",
    ),
    "cma_es": AlgorithmSpec(
        factory=PyCmaAlgorithm,
        description="Alias for pycma.",
        family="traditional",
        numeric_only=True,
        categorical_to_continuous="onehot",
    ),
    "sobol_search": AlgorithmSpec(
        factory=SobolSearchAlgorithm,
        description="Scrambled Sobol search over a transformed numeric unit cube.",
        family="traditional",
        numeric_only=True,
    ),
    "sobol": AlgorithmSpec(
        factory=SobolSearchAlgorithm,
        description="Alias for sobol_search.",
        family="traditional",
        numeric_only=True,
    ),
    "optuna_tpe": AlgorithmSpec(
        factory=OptunaTpeAlgorithm,
        description="Optuna TPE via the optional `optuna` package.",
        family="model_based",
    ),
    "gp_ei": AlgorithmSpec(
        factory=GpEiAlgorithm,
        description="BoTorch Gaussian-process Bayesian optimization with expected improvement.",
        family="model_based",
        categorical_to_continuous="onehot",
    ),
    "gpei": AlgorithmSpec(
        factory=GpEiAlgorithm,
        description="Alias for gp_ei.",
        family="model_based",
        categorical_to_continuous="onehot",
    ),
    "gp_bo": AlgorithmSpec(
        factory=GpEiAlgorithm,
        description="Alias for gp_ei.",
        family="model_based",
        categorical_to_continuous="onehot",
    ),
    "botorch_turbo": AlgorithmSpec(
        factory=BotorchTurboAlgorithm,
        description="TuRBO-1 using the pinned BoTorch official tutorial implementation.",
        family="model_based",
        numeric_only=True,
    ),
    "turbo": AlgorithmSpec(
        factory=BotorchTurboAlgorithm,
        description="Alias for botorch_turbo.",
        family="model_based",
        numeric_only=True,
    ),
    "git_bo": AlgorithmSpec(
        factory=GitBoAlgorithm,
        description="GIT-BO with differentiable TabPFN v2, a gradient-informed active subspace, and UCB.",
        family="model_based",
        numeric_only=True,
    ),
    "gitbo": AlgorithmSpec(
        factory=GitBoAlgorithm,
        description="Alias for git_bo.",
        family="model_based",
        numeric_only=True,
    ),
    "pfns4bo": AlgorithmSpec(
        factory=Pfns4BoAlgorithm,
        description="PFNs4BO with fixed continuous/pool routing for benchmark smoke tasks.",
        family="model_based",
        categorical_to_continuous="onehot",
    ),
    "pfns4bo_tabpfn_v2": AlgorithmSpec(
        factory=TabPfnV2BoAlgorithm,
        description="TabPFN v2 surrogate over a deterministic candidate pool for arbitrary-dimensional BO tasks.",
        family="model_based",
    ),
    "pfns4bo_custom": AlgorithmSpec(
        factory=CustomPfnsBoAlgorithm,
        description="Custom-trained PFN surrogate over a deterministic candidate pool for arbitrary-dimensional BO tasks.",
        family="model_based",
        categorical_to_continuous="onehot",
    ),
    "graph_ga": AlgorithmSpec(
        factory=GraphGAAlgorithm,
        description="PMO Graph GA over direct SMILES with ask/tell oracle evaluation.",
        family="molecular",
    ),
    "gpbo": AlgorithmSpec(
        factory=GraphGPBOAlgorithm,
        description="PMO GPBO with Morgan fingerprints, Tanimoto GP, UCB, and Graph GA acquisition search.",
        family="molecular",
    ),
    "graph_gpbo": AlgorithmSpec(
        factory=GraphGPBOAlgorithm,
        description="Alias for gpbo.",
        family="molecular",
    ),
    "llambo": AlgorithmSpec(
        factory=_create_llambo,
        description="LLAMBO-style prompt optimizer with pluggable chat backends and an offline heuristic mode.",
        family="llm_based",
    ),
    "opro": AlgorithmSpec(
        factory=OproAlgorithm,
        description="OPRO-style prompt optimizer over prior configuration/objective pairs.",
        family="llm_based",
    ),
    "skydiscover_interleaved": AlgorithmSpec(
        factory=SkydiscoverInterleavedAlgorithm,
        description=(
            "Interleave SkyDiscover meta-evolution of suggest_next_config with BBO dict optimization."
        ),
        family="llm_based",
    ),
    "skydiscover_meta": AlgorithmSpec(
        factory=SkydiscoverInterleavedAlgorithm,
        description="Alias for skydiscover_interleaved.",
        family="llm_based",
    ),
    "pablo": AlgorithmSpec(
        factory=_create_pablo,
        description="Stateless Planner/Explorer/Worker agentic optimizer with mock and OpenAI-compatible providers.",
        family="agentic",
    ),
    "palbo": AlgorithmSpec(
        factory=_create_pablo,
        description="Alias for pablo.",
        family="agentic",
    ),
    "agentic_bo": AlgorithmSpec(
        factory=_create_agentic_bo,
        description="Agent-controlled GP BO with probe, propose, reconfigure, and commit actions.",
        family="agentic",
    ),
    "agentic_nanobot": AlgorithmSpec(
        factory=NanobotBBOAlgorithm,
        description="General-agent BBO optimizer backed by the Nanobot CLI agent.",
        family="agentic",
    ),
    "nanobot": AlgorithmSpec(
        factory=NanobotBBOAlgorithm,
        description="Alias for agentic_nanobot.",
        family="agentic",
    ),
    "agentic_claude_code": AlgorithmSpec(
        factory=ClaudeCodeBBOAlgorithm,
        description="General-agent BBO optimizer backed by Claude Code.",
        family="agentic",
    ),
    "claude_code": AlgorithmSpec(
        factory=ClaudeCodeBBOAlgorithm,
        description="Alias for agentic_claude_code.",
        family="agentic",
    ),
    "claude-code": AlgorithmSpec(
        factory=ClaudeCodeBBOAlgorithm,
        description="Alias for agentic_claude_code.",
        family="agentic",
    ),
    "agentic_codex": AlgorithmSpec(
        factory=CodexBBOAlgorithm,
        description="General-agent BBO optimizer backed by Codex CLI.",
        family="agentic",
    ),
    "codex": AlgorithmSpec(
        factory=CodexBBOAlgorithm,
        description="Alias for agentic_codex.",
        family="agentic",
    ),
    "agentic_openai_compatible": AlgorithmSpec(
        factory=OpenAICompatibleBBOAlgorithm,
        description="General-agent BBO optimizer backed by OpenAI-compatible function calling.",
        family="agentic",
    ),
    "openai_compatible_agent": AlgorithmSpec(
        factory=OpenAICompatibleBBOAlgorithm,
        description="Alias for agentic_openai_compatible.",
        family="agentic",
    ),
}


def create_algorithm(name: str, **kwargs: Any) -> Algorithm:
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unknown algorithm `{name}`. Available: {available}")
    return ALGORITHM_REGISTRY[name].factory(**kwargs)


def algorithms_by_family() -> dict[str, dict[str, AlgorithmSpec]]:
    grouped: dict[str, dict[str, AlgorithmSpec]] = {}
    for name, spec in ALGORITHM_REGISTRY.items():
        grouped.setdefault(spec.family, {})[name] = spec
    return grouped


__all__ = ["ALGORITHM_REGISTRY", "AlgorithmSpec", "algorithms_by_family", "create_algorithm"]
