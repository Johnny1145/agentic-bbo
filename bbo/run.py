"""CLI and helpers for running standalone agentic BBO demos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .algorithms import (
    ALGORITHM_REGISTRY,
    OpenAICompatibleLlamboBackend,
    OpenAICompatibleOproBackend,
    comparable_baseline_kwargs,
    create_algorithm,
)
from .algorithms.agentic import AGENT_TOOL_MODE_CLI_CHOICES, normalize_agent_tool_mode
from .algorithms.kimi import (
    DEFAULT_KIMI_API_KEY_ENV,
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_MODEL,
    normalize_kimi_openai_base_url,
)
from .core import (
    CumulativeEvalTimeComparisonPlotter,
    ExperimentConfig,
    Experimenter,
    JsonlMetricLogger,
    Landscape2DPlotter,
    ObjectiveDistributionPlotter,
    OptimizationTracePlotter,
    CumulativeEvalTimePlotter,
    OptimizerComparisonPlotter,
    PerTrialEvalTimePlotter,
    RegretTracePlotter,
    ScalarBarPlotter,
    Task,
)
from .tasks import ALL_TASK_NAMES, create_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "runs" / "demo"
NATIVE_HARNESS_ALGORITHM_NAMES = {
        "agentic_bo",
    "agentic_nanobot",
    "nanobot",
    "agentic_codex",
    "codex",
    "agentic_claude_code",
    "claude_code",
    "claude-code",
}


class BBOArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with algorithm-aware agent defaults."""

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        _apply_algorithm_aware_agent_defaults(parsed)
        return parsed


def _apply_algorithm_aware_agent_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "agent_tool_mode", None) is None:
        args.agent_tool_mode = (
            "no_tool"
            if getattr(args, "algorithm", None) in NATIVE_HARNESS_ALGORITHM_NAMES
            else "function_calling"
        )
    else:
        args.agent_tool_mode = normalize_agent_tool_mode(args.agent_tool_mode)


def _sanitize_path_segment(value: str) -> str:
    """Make a user-provided token safe as one filesystem directory name."""
    fragments: list[str] = []
    for ch in value.strip():
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            fragments.append(ch)
        else:
            fragments.append("_")
    cleaned = "".join(fragments).strip("._-")
    return cleaned if cleaned else "unknown"


def _runs_subdirectory_name(
    algorithm_name: str,
    *,
    skydiscover_search_type: str,
) -> str:
    """Result root path segment: plain algorithm id, except skydiscover (includes search type)."""
    if algorithm_name not in {"skydiscover_interleaved", "skydiscover_meta"}:
        return algorithm_name
    safe_search = _sanitize_path_segment(skydiscover_search_type)
    return f"{algorithm_name}_{safe_search}"


def _resolve_optional_env(*names: str) -> str | None:
    for name in names:
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_llambo_algorithm_kwargs(
    *,
    llambo_backend: str,
    llambo_model: str,
    llambo_initial_samples: int,
    llambo_candidates: int,
    llambo_templates: int,
    llambo_predictions: int,
    llambo_alpha: float,
    llambo_openai_api_key_env: str,
    llambo_openai_base_url: str | None,
    llambo_openai_organization: str | None,
    llambo_openai_project: str | None,
    llambo_openai_timeout_seconds: float,
    llambo_openai_max_retries: int = 3,
    llambo_openai_use_structured_outputs: bool = True,
    llambo_openai_include_seed: bool = True,
    llambo_openai_include_store: bool = True,
    kimi_api_key_env: str = DEFAULT_KIMI_API_KEY_ENV,
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL,
    kimi_model: str = DEFAULT_KIMI_MODEL,
) -> dict[str, Any]:
    is_kimi = llambo_backend == "kimi"
    effective_model = kimi_model if is_kimi and llambo_model == "gpt-4o-mini" else llambo_model
    kwargs: dict[str, Any] = {
        "backend": llambo_backend,
        "model": effective_model,
        "n_initial_samples": llambo_initial_samples,
        "n_candidates": llambo_candidates,
        "n_templates": llambo_templates,
        "n_predictions": llambo_predictions,
        "alpha": llambo_alpha,
    }
    if llambo_backend not in {"openai", "kimi"}:
        return kwargs

    api_key_env = (
        kimi_api_key_env
        if is_kimi and llambo_openai_api_key_env == "OPENAI_API_KEY"
        else llambo_openai_api_key_env
    )
    api_key = _resolve_optional_env(api_key_env)
    if not api_key:
        provider_label = "Kimi" if is_kimi else "OpenAI"
        raise ValueError(
            f"LLAMBO {provider_label} backend requires an API key in the user-facing environment. "
            f"Set `{api_key_env}` or choose `--llambo-backend heuristic`."
        )
    kwargs["backend_impl"] = OpenAICompatibleLlamboBackend(
        model=effective_model,
        api_key=api_key,
        base_url=(
            (llambo_openai_base_url or kimi_base_url)
            if is_kimi
            else llambo_openai_base_url or _resolve_optional_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
        ),
        organization=llambo_openai_organization or _resolve_optional_env("OPENAI_ORGANIZATION"),
        project=llambo_openai_project or _resolve_optional_env("OPENAI_PROJECT"),
        timeout_seconds=llambo_openai_timeout_seconds,
        max_retries=llambo_openai_max_retries,
        use_structured_outputs=llambo_openai_use_structured_outputs,
        provider_name="kimi" if is_kimi else "openai",
        include_seed=False if is_kimi else llambo_openai_include_seed,
        include_store=False if is_kimi else llambo_openai_include_store,
    )
    return kwargs


def _build_opro_algorithm_kwargs(
    *,
    opro_backend: str,
    opro_model: str,
    opro_initial_samples: int,
    opro_candidates: int,
    opro_prompt_pairs: int | None,
    opro_openai_api_key_env: str,
    opro_openai_base_url: str | None,
    opro_openai_organization: str | None,
    opro_openai_project: str | None,
    opro_openai_timeout_seconds: float,
    opro_openai_max_retries: int = 3,
    opro_openai_include_seed: bool = True,
    opro_openai_include_store: bool = True,
    kimi_api_key_env: str = DEFAULT_KIMI_API_KEY_ENV,
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL,
    kimi_model: str = DEFAULT_KIMI_MODEL,
) -> dict[str, Any]:
    is_kimi = opro_backend == "kimi"
    effective_model = kimi_model if is_kimi and opro_model == "gpt-4o-mini" else opro_model
    kwargs: dict[str, Any] = {
        "backend": opro_backend,
        "model": effective_model,
        "n_initial_samples": opro_initial_samples,
        "n_candidates": opro_candidates,
        "max_prompt_pairs": opro_prompt_pairs,
    }
    if opro_backend not in {"openai", "kimi"}:
        return kwargs

    api_key_env = (
        kimi_api_key_env
        if is_kimi and opro_openai_api_key_env == "OPENAI_API_KEY"
        else opro_openai_api_key_env
    )
    api_key = _resolve_optional_env(api_key_env)
    if not api_key:
        provider_label = "Kimi" if is_kimi else "OpenAI"
        raise ValueError(
            f"OPRO {provider_label} backend requires an API key in the user-facing environment. "
            f"Set `{api_key_env}` or choose `--opro-backend heuristic`."
        )
    kwargs["backend_impl"] = OpenAICompatibleOproBackend(
        model=effective_model,
        api_key=api_key,
        base_url=(
            (opro_openai_base_url or kimi_base_url)
            if is_kimi
            else opro_openai_base_url or _resolve_optional_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
        ),
        organization=opro_openai_organization or _resolve_optional_env("OPENAI_ORGANIZATION"),
        project=opro_openai_project or _resolve_optional_env("OPENAI_PROJECT"),
        timeout_seconds=opro_openai_timeout_seconds,
        max_retries=opro_openai_max_retries,
        provider_name="kimi" if is_kimi else "openai",
        include_seed=False if is_kimi else opro_openai_include_seed,
        include_store=False if is_kimi else opro_openai_include_store,
    )
    return kwargs


def run_single_experiment(
    *,
    task_name: str,
    algorithm_name: str,
    seed: int,
    max_evaluations: int | None = None,
    task_kwargs: dict[str, Any] | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    resume: bool = False,
    sigma_fraction: float = 0.18,
    popsize: int | None = 2,
    noise_std: float = 0.0,
    surrogate_path: str | Path | None = None,
    knobs_json_path: str | Path | None = None,
    generate_plots: bool = True,
    pfns_device: str | None = None,
    pfns_pool_size: int = 256,
    pfns_model: str = "hebo_plus",
    pfns_acquisition: str = "ei",
    pfns_custom_model_path: str | Path | None = None,
    pfns_tabpfn_n_estimators: int = 8,
    pfns_tabpfn_ignore_pretraining_limits: bool = True,
    pfns_tabpfn_fit_mode: str = "fit_preprocessors",
    gp_ei_pool_size: int | None = None,
    gp_ei_startup_trials: int = 2,
    gp_ei_xi: float = 0.0,
    gp_ei_alpha: float = 1e-6,
    gp_ei_n_restarts_optimizer: int = 0,
    git_bo_subspace_dim: int = 10,
    git_bo_beta: float = 2.33,
    git_bo_n_candidates: int | None = None,
    git_bo_n_estimators: int = 1,
    git_bo_inference_batch_size: int = 128,
    git_bo_device: str = "auto",
    pablo_provider: str = "mock",
    pablo_base_url: str | None = None,
    pablo_api_key_env: str = "PABLO_API_KEY",
    pablo_global_model: str | None = None,
    pablo_worker_model: str | None = None,
    pablo_planner_model: str | None = None,
    pablo_explorer_model: str | None = None,
    pablo_model: str = "gpt-4.1-mini",
    pablo_init_points: int = 4,
    pablo_max_fails: int = 3,
    pablo_num_seeds: int = 2,
    pablo_max_tasks: int = 20,
    pablo_enable_explorer: bool = True,
    pablo_enable_planner: bool = True,
    pablo_enable_worker: bool = True,
    llambo_backend: str = "heuristic",
    llambo_model: str = "gpt-4o-mini",
    llambo_initial_samples: int = 5,
    llambo_candidates: int = 8,
    llambo_templates: int = 2,
    llambo_predictions: int = 6,
    llambo_alpha: float = -0.2,
    llambo_openai_api_key_env: str = "OPENAI_API_KEY",
    llambo_openai_base_url: str | None = None,
    llambo_openai_organization: str | None = None,
    llambo_openai_project: str | None = None,
    llambo_openai_timeout_seconds: float = 30.0,
    llambo_openai_max_retries: int = 3,
    llambo_openai_use_structured_outputs: bool = True,
    llambo_openai_include_seed: bool = True,
    llambo_openai_include_store: bool = True,
    opro_backend: str = "heuristic",
    opro_model: str = "gpt-4o-mini",
    opro_initial_samples: int = 5,
    opro_candidates: int = 8,
    opro_prompt_pairs: int | None = None,
    opro_openai_api_key_env: str = "OPENAI_API_KEY",
    opro_openai_base_url: str | None = None,
    opro_openai_organization: str | None = None,
    opro_openai_project: str | None = None,
    opro_openai_timeout_seconds: float = 30.0,
    opro_openai_max_retries: int = 3,
    opro_openai_include_seed: bool = True,
    opro_openai_include_store: bool = True,
    kimi_api_key_env: str = DEFAULT_KIMI_API_KEY_ENV,
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL,
    kimi_model: str = DEFAULT_KIMI_MODEL,
    agent_timeout_seconds: float = 180.0,
    agent_max_retries: int = 1,
    agent_history_limit: int = 40,
    agent_candidates_per_call: int = 1,
    agent_model: str | None = None,
    agent_provider: str | None = None,
    agent_api_base: str | None = None,
    agent_api_key_env: str | None = None,
    agent_executable: str | None = None,
    agent_initial_random: int = 0,
    agent_tool_mode: str | None = None,
    agent_prompt_style: str = "workspace",
    agent_prompt_profile: str | None = None,
    agent_max_tool_calls: int = 16,
    agent_enable_memory: bool = True,
    agent_enable_code_interpreter: bool = True,
    agent_code_backend: str = "sandboxfusion",
    agent_docker_image: str = "agentic-bbo-analysis-sandbox:v1",
    sandbox_fusion_base_url: str | None = None,
    agent_web_search_provider: str = "disabled",
    agent_web_search_api_key_env: str | None = None,
    agent_allow_fallback: bool = True,
    agent_require_visible_cot: bool = False,
    agent_enable_bbo_skills: bool = False,
    agent_skill_paths: list[str | Path] | None = None,
    agent_enabled_tool_names: list[str] | tuple[str, ...] | None = None,
    agent_optimizer_backend_allowlist: list[str] | tuple[str, ...] = (),
    agent_optimizer_max_calls_per_round: int = 3,
    agent_experiment_condition: str = "default",
    agent_require_analysis_evidence_per_round: bool = False,
    agent_required_tool_names_per_round: list[str] | tuple[str, ...] = (),
    agent_require_candidate_validation_per_round: bool = False,
    agent_require_optimizer_decision_per_round: bool = False,
    skydiscover_interleave_every: int = 5,
    skydiscover_round_iterations: int = 3,
    skydiscover_config_path: str | Path | None = None,
    skydiscover_runner: bool = False,
    skydiscover_search_type: str = "topk",
    skydiscover_model: str | None = None,
    skydiscover_max_meta_history: int = 32,
    molecular_initial_smiles_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_task_kwargs = dict(task_kwargs or {})
    if surrogate_path is not None:
        resolved_task_kwargs["surrogate_path"] = surrogate_path
    if knobs_json_path is not None:
        resolved_task_kwargs["knobs_json_path"] = knobs_json_path
    task = create_task(
        task_name,
        max_evaluations=max_evaluations,
        seed=seed,
        noise_std=noise_std,
        **resolved_task_kwargs,
    )
    _require_algorithm_support(task, algorithm_name)
    if agent_tool_mode is None:
        agent_tool_mode = (
            "no_tool"
            if algorithm_name in NATIVE_HARNESS_ALGORITHM_NAMES
            else "function_calling"
        )
    else:
        agent_tool_mode = normalize_agent_tool_mode(agent_tool_mode)
    run_segments = _runs_subdirectory_name(
        algorithm_name,
        skydiscover_search_type=skydiscover_search_type,
    )
    run_dir = _allocate_run_dir(results_root / task_name / run_segments / f"seed_{seed}", resume=resume)
    results_jsonl = run_dir / "trials.jsonl"

    algorithm_kwargs: dict[str, Any] = {}
    if algorithm_name in {"pycma", "cma_es"}:
        algorithm_kwargs = comparable_baseline_kwargs(
            algorithm_name,
            overrides={
                "sigma_fraction": sigma_fraction,
                "popsize": popsize,
            },
        )
    elif algorithm_name == "pfns4bo":
        algorithm_kwargs = {
            "device": pfns_device,
            "pool_size": pfns_pool_size,
            "model_name": pfns_model,
        }
    elif algorithm_name == "pfns4bo_tabpfn_v2":
        algorithm_kwargs = {
            "device": pfns_device,
            "pool_size": pfns_pool_size,
            "acquisition": pfns_acquisition,
            "n_estimators": pfns_tabpfn_n_estimators,
            "ignore_pretraining_limits": pfns_tabpfn_ignore_pretraining_limits,
            "fit_mode": pfns_tabpfn_fit_mode,
        }
    elif algorithm_name == "pfns4bo_custom":
        algorithm_kwargs = {
            "device": pfns_device,
            "pool_size": pfns_pool_size,
            "acquisition": pfns_acquisition,
            "model_path": pfns_custom_model_path,
        }
    elif algorithm_name in {"gp_ei", "gpei", "gp_bo"}:
        algorithm_kwargs = comparable_baseline_kwargs(
            algorithm_name,
            overrides={
                "pool_size": gp_ei_pool_size,
                "startup_trials": gp_ei_startup_trials,
                "xi": gp_ei_xi,
                "alpha": gp_ei_alpha,
                "n_restarts_optimizer": gp_ei_n_restarts_optimizer,
            },
        )
    elif algorithm_name in {"git_bo", "gitbo"}:
        algorithm_kwargs = {
            "subspace_dim": git_bo_subspace_dim,
            "beta": git_bo_beta,
            "n_candidates": git_bo_n_candidates,
            "n_estimators": git_bo_n_estimators,
            "inference_batch_size": git_bo_inference_batch_size,
            "device": git_bo_device,
        }
    elif algorithm_name in {"graph_ga", "gpbo", "graph_gpbo"}:
        algorithm_kwargs = {
            "initial_smiles_path": molecular_initial_smiles_path,
        }
    elif algorithm_name in {"pablo", "palbo"}:
        effective_pablo_api_key_env = pablo_api_key_env
        effective_pablo_base_url = pablo_base_url
        effective_pablo_model = pablo_model
        if pablo_provider == "kimi":
            if pablo_api_key_env == "PABLO_API_KEY":
                effective_pablo_api_key_env = kimi_api_key_env
            effective_pablo_base_url = normalize_kimi_openai_base_url(pablo_base_url or kimi_base_url)
            if pablo_model == "gpt-4.1-mini":
                effective_pablo_model = kimi_model
        algorithm_kwargs = {
            "provider": pablo_provider,
            "base_url": effective_pablo_base_url,
            "api_key_env": effective_pablo_api_key_env,
            "global_model": pablo_global_model,
            "worker_model": pablo_worker_model,
            "planner_model": pablo_planner_model,
            "explorer_model": pablo_explorer_model,
            "model": effective_pablo_model,
            "init_points": pablo_init_points,
            "max_fails": pablo_max_fails,
            "num_seeds": pablo_num_seeds,
            "max_tasks": pablo_max_tasks,
            "enable_explorer": pablo_enable_explorer,
            "enable_planner": pablo_enable_planner,
            "enable_worker": pablo_enable_worker,
            "run_dir": run_dir,
            "resume": resume,
        }
    elif algorithm_name in {
        "agentic_bo",
        "agentic_nanobot",
        "nanobot",
        "agentic_claude_code",
        "claude_code",
        "claude-code",
        "agentic_codex",
        "codex",
        "agentic_openai_compatible",
        "openai_compatible_agent",
    }:
        algorithm_kwargs = {
            "timeout_seconds": agent_timeout_seconds,
            "max_retries": agent_max_retries,
            "history_limit": agent_history_limit,
            "candidates_per_call": agent_candidates_per_call,
            "model": agent_model,
            "provider": agent_provider,
            "api_base": agent_api_base,
            "api_key_env": agent_api_key_env,
            "executable": agent_executable,
            "initial_random": agent_initial_random,
            "run_dir": run_dir,
            "resume": resume,
            "tool_mode": agent_tool_mode,
            "docker_image": agent_docker_image,
            "prompt_style": agent_prompt_style,
            "prompt_profile": agent_prompt_profile,
            "max_tool_calls": agent_max_tool_calls,
            "enable_memory": agent_enable_memory,
            "enable_code_interpreter": agent_enable_code_interpreter,
            "code_backend": agent_code_backend,
            "sandbox_fusion_base_url": sandbox_fusion_base_url,
            "web_search_provider": agent_web_search_provider,
            "web_search_api_key_env": agent_web_search_api_key_env,
            "allow_fallback": agent_allow_fallback,
            "require_visible_cot": agent_require_visible_cot,
            "experiment_condition": agent_experiment_condition,
            "require_analysis_evidence_per_round": agent_require_analysis_evidence_per_round,
            "required_tool_names_per_round": agent_required_tool_names_per_round,
            "require_candidate_validation_per_round": agent_require_candidate_validation_per_round,
            "require_optimizer_decision_per_round": agent_require_optimizer_decision_per_round,
            "enable_bbo_skills": agent_enable_bbo_skills,
            "skill_paths": agent_skill_paths,
            "enabled_tool_names": agent_enabled_tool_names,
            "optimizer_backend_allowlist": agent_optimizer_backend_allowlist,
            "optimizer_max_calls_per_round": agent_optimizer_max_calls_per_round,
        }
    elif algorithm_name == "llambo":
        algorithm_kwargs = {"run_dir": run_dir}
        algorithm_kwargs.update(_build_llambo_algorithm_kwargs(
            llambo_backend=llambo_backend,
            llambo_model=llambo_model,
            llambo_initial_samples=llambo_initial_samples,
            llambo_candidates=llambo_candidates,
            llambo_templates=llambo_templates,
            llambo_predictions=llambo_predictions,
            llambo_alpha=llambo_alpha,
            llambo_openai_api_key_env=llambo_openai_api_key_env,
            llambo_openai_base_url=llambo_openai_base_url,
            llambo_openai_organization=llambo_openai_organization,
            llambo_openai_project=llambo_openai_project,
            llambo_openai_timeout_seconds=llambo_openai_timeout_seconds,
            llambo_openai_max_retries=llambo_openai_max_retries,
            llambo_openai_use_structured_outputs=llambo_openai_use_structured_outputs,
            llambo_openai_include_seed=llambo_openai_include_seed,
            llambo_openai_include_store=llambo_openai_include_store,
            kimi_api_key_env=kimi_api_key_env,
            kimi_base_url=kimi_base_url,
            kimi_model=kimi_model,
        ))
    elif algorithm_name == "opro":
        algorithm_kwargs = _build_opro_algorithm_kwargs(
            opro_backend=opro_backend,
            opro_model=opro_model,
            opro_initial_samples=opro_initial_samples,
            opro_candidates=opro_candidates,
            opro_prompt_pairs=opro_prompt_pairs,
            opro_openai_api_key_env=opro_openai_api_key_env,
            opro_openai_base_url=opro_openai_base_url,
            opro_openai_organization=opro_openai_organization,
            opro_openai_project=opro_openai_project,
            opro_openai_timeout_seconds=opro_openai_timeout_seconds,
            opro_openai_max_retries=opro_openai_max_retries,
            opro_openai_include_seed=opro_openai_include_seed,
            opro_openai_include_store=opro_openai_include_store,
            kimi_api_key_env=kimi_api_key_env,
            kimi_base_url=kimi_base_url,
            kimi_model=kimi_model,
        )
    elif algorithm_name in {"skydiscover_interleaved", "skydiscover_meta"}:
        algorithm_kwargs = {
            "run_dir": run_dir,
            "interleave_every": skydiscover_interleave_every,
            "skydiscover_round_iterations": skydiscover_round_iterations,
            "skydiscover_config_path": skydiscover_config_path,
            "skydiscover_runner_enabled": skydiscover_runner,
            "skydiscover_search_type": skydiscover_search_type,
            "skydiscover_model": skydiscover_model,
            "max_meta_history": skydiscover_max_meta_history,
        }
    algorithm = create_algorithm(algorithm_name, **algorithm_kwargs)

    logger = JsonlMetricLogger(results_jsonl)
    experiment = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(seed=seed, resume=resume, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    records = logger.load_records()

    serializable_summary = {
        "task_name": summary.task_name,
        "algorithm_name": summary.algorithm_name,
        "seed": summary.seed,
        "n_completed": summary.n_completed,
        "total_eval_time": summary.total_eval_time,
        "best_primary_objective": summary.best_primary_objective,
        "stop_reason": summary.stop_reason,
        "description_fingerprint": summary.description_fingerprint,
        "incumbents": [
            {
                "config": incumbent.config,
                "score": incumbent.score,
                "objectives": incumbent.objectives,
                "trial_id": incumbent.trial_id,
                "metadata": incumbent.metadata,
            }
            for incumbent in summary.incumbents
        ],
        "logger_summary": summary.logger_summary,
        "final_evaluation": (
            None
            if getattr(summary, "final_evaluation", None) is None
            else summary.final_evaluation.to_dict()
        ),
        "run_dir": str(run_dir),
        "results_jsonl": str(results_jsonl),
        "trial_count": len(records),
        "internal_artifacts": getattr(algorithm, "artifact_paths", {}),
        "role_model_routes": getattr(algorithm, "routing_table", {}),
    }
    if algorithm_name in {"skydiscover_interleaved", "skydiscover_meta"}:
        serializable_summary["skydiscover_search_type"] = skydiscover_search_type
        serializable_summary["runs_subdirectory"] = run_segments
    if generate_plots:
        plot_paths = generate_visualizations(
            task=task,
            logger=logger,
            output_dir=run_dir / "plots",
            algorithm_label=algorithm_name,
        )
        serializable_summary["plot_paths"] = [str(p) for p in plot_paths]
    (run_dir / "summary.json").write_text(json.dumps(serializable_summary, indent=2, sort_keys=True), encoding="utf-8")
    return serializable_summary


def run_demo_suite(
    *,
    task_name: str = "bbob_f01_d10",
    seed: int = 7,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    random_evaluations: int = 45,
    pycma_evaluations: int = 36,
    sigma_fraction: float = 0.18,
    popsize: int | None = 2,
    resume: bool = False,
    generate_plots: bool = True,
) -> dict[str, Any]:
    task = create_task(task_name, max_evaluations=random_evaluations, seed=seed)
    _require_algorithm_support(task, "random_search")
    _require_algorithm_support(task, "pycma")
    random_summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=seed,
        max_evaluations=random_evaluations,
        results_root=results_root,
        resume=resume,
        generate_plots=generate_plots,
    )
    pycma_summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="pycma",
        seed=seed,
        max_evaluations=pycma_evaluations,
        results_root=results_root,
        resume=resume,
        sigma_fraction=sigma_fraction,
        popsize=popsize,
        generate_plots=generate_plots,
    )

    comparison_dir = _allocate_run_dir(results_root / task_name / "suite" / f"seed_{seed}", resume=resume)
    if generate_plots:
        comparison_dir_plots = comparison_dir / "plots"
        generate_comparison_plot(
            task=task,
            histories={
                "random_search": JsonlMetricLogger(Path(random_summary["results_jsonl"])).load_records(),
                "pycma": JsonlMetricLogger(Path(pycma_summary["results_jsonl"])).load_records(),
            },
            output_dir=comparison_dir_plots,
        )
        extra = _generate_two_algorithm_suite_plots(
            task=task,
            random_summary=random_summary,
            pycma_summary=pycma_summary,
            output_dir=comparison_dir_plots,
        )
        comparison_plot_paths = [str(comparison_dir_plots / "comparison.png"), *[str(p) for p in extra]]
    suite_summary: dict[str, Any] = {
        "task_name": task_name,
        "seed": seed,
        "random_search": random_summary,
        "pycma": pycma_summary,
        "comparison_dir": str(comparison_dir),
    }
    if generate_plots:
        suite_summary["comparison_plot_paths"] = comparison_plot_paths
    (comparison_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")
    return suite_summary


def generate_visualizations(
    task: Task,
    logger: JsonlMetricLogger,
    output_dir: Path,
    *,
    algorithm_label: str,
) -> list[Path]:
    """Emit one figure per metric: objective trace, distribution, per-trial time, cumulative time, optional 2D landscape."""
    records = logger.load_records()
    if not records:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    objective_name = task.spec.primary_objective.name
    direction = task.spec.primary_objective.direction
    artifacts: list[Path] = [
        OptimizationTracePlotter().plot(
            records,
            objective_name=objective_name,
            direction=direction,
            output_path=output_dir / "trace.png",
            title=f"{display} | {algorithm_label} | {objective_name} trace",
        ).path,
        ObjectiveDistributionPlotter().plot(
            records,
            objective_name=objective_name,
            output_path=output_dir / "distribution.png",
            title=f"{display} | {algorithm_label} | {objective_name} distribution",
        ).path,
        PerTrialEvalTimePlotter().plot(
            records,
            output_path=output_dir / "per_trial_eval_time.png",
            title=f"{display} | {algorithm_label} | eval wall time (per trial)",
        ).path,
        CumulativeEvalTimePlotter().plot(
            records,
            output_path=output_dir / "cumulative_eval_time.png",
            title=f"{display} | {algorithm_label} | cumulative eval time",
        ).path,
    ]
    if int(task.spec.metadata.get("dimension", 0)) == 2 and hasattr(task, "surface_grid"):
        artifacts.append(
            Landscape2DPlotter().plot(
                task,
                records,
                objective_name=objective_name,
                output_path=output_dir / "landscape.png",
                title=f"{display} | {algorithm_label} | sampled 2D landscape",
                resolution=int(task.spec.metadata.get("plot_resolution", 180)),
            ).path
        )
    known = task.spec.metadata.get("known_optimum")
    if known is not None and isinstance(known, (int, float)):
        artifacts.append(
            RegretTracePlotter().plot(
                records,
                objective_name=objective_name,
                direction=direction,
                known_optimum=float(known),
                output_path=output_dir / "regret.png",
                title=f"{display} | {algorithm_label} | regret (known optimum in metadata)",
            ).path
        )
    return artifacts


def generate_comparison_plot(
    *,
    task: Task,
    histories: dict[str, list],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    on = task.spec.primary_objective.name
    artifact = OptimizerComparisonPlotter().plot(
        histories,
        objective_name=on,
        direction=task.spec.primary_objective.direction,
        output_path=output_dir / "comparison.png",
        title=f"{display} | running best {on} (all algorithms)",
    )
    return artifact.path


def _generate_two_algorithm_suite_plots(
    *,
    task: Task,
    random_summary: dict[str, Any],
    pycma_summary: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Bar + cumulative-time comparison for the fixed suite (random_search vs pycma). One metric per figure."""
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    on = task.spec.primary_objective.name
    r_rs = JsonlMetricLogger(Path(random_summary["results_jsonl"])).load_records()
    r_pc = JsonlMetricLogger(Path(pycma_summary["results_jsonl"])).load_records()
    paths: list[Path] = [
        CumulativeEvalTimeComparisonPlotter().plot(
            {"random_search": r_rs, "pycma": r_pc},
            output_path=output_dir / "comparison_cumulative_eval_time.png",
            title=f"{display} | cumulative eval time (all algorithms)",
        ).path,
        ScalarBarPlotter().plot(
            {
                "random_search": float(random_summary["best_primary_objective"]),
                "pycma": float(pycma_summary["best_primary_objective"]),
            },
            ylabel=f"best {on}",
            output_path=output_dir / "bar_best_primary_objective.png",
            title=f"{display} | best {on} (final, one bar per algorithm)",
        ).path,
        ScalarBarPlotter().plot(
            {
                "random_search": float(random_summary["total_eval_time"]),
                "pycma": float(pycma_summary["total_eval_time"]),
            },
            ylabel="Total eval time (s)",
            output_path=output_dir / "bar_total_eval_time.png",
            title=f"{display} | total eval time (sum of trial times)",
        ).path,
    ]
    return paths


def _require_algorithm_support(task: Task, algorithm_name: str) -> None:
    algorithm_spec = ALGORITHM_REGISTRY[algorithm_name]
    try:
        task.spec.search_space.numeric_bounds()
        return
    except TypeError as exc:
        if algorithm_spec.categorical_to_continuous is not None:
            return
        if not algorithm_spec.numeric_only:
            return
        raise ValueError(
            f"Algorithm `{algorithm_name}` only supports fully numeric search spaces; "
            f"task `{task.spec.name}` includes categorical parameters."
        ) from exc


def _allocate_run_dir(base_dir: Path, *, resume: bool) -> Path:
    if resume or not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    counter = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_run_{counter:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = BBOArgumentParser(description="Run tasks for the agentic BBO benchmark core.")
    parser.add_argument("--task", default="bbob_f01_d10", choices=sorted(ALL_TASK_NAMES))
    parser.add_argument(
        "--algorithm",
        default="suite",
        choices=["suite", *sorted(ALGORITHM_REGISTRY)],
        help="Which demo to run. `suite` runs both algorithms and a comparison plot.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--random-evaluations", type=int, default=45)
    parser.add_argument("--pycma-evaluations", type=int, default=36)
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Synthetic-only: add Gaussian noise to objective evaluations.",
    )
    parser.add_argument(
        "--surrogate-path",
        type=Path,
        default=None,
        help="Surrogate-only: override path to .joblib (otherwise uses bundled assets or env var override).",
    )
    parser.add_argument(
        "--knobs-json-path",
        type=Path,
        default=None,
        help="Surrogate-only: override path to knobs_*.json (otherwise uses bundled assets).",
    )
    parser.add_argument("--pfns-device", default=None)
    parser.add_argument("--pfns-pool-size", type=int, default=256)
    parser.add_argument("--pfns-model", default="hebo_plus")
    parser.add_argument("--pfns-acquisition", choices=("ei", "ucb", "mean"), default="ei")
    parser.add_argument("--pfns-custom-model-path", default=None)
    parser.add_argument("--pfns-tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--pfns-tabpfn-fit-mode", default="fit_preprocessors")
    parser.add_argument(
        "--pfns-tabpfn-ignore-pretraining-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gp-ei-pool-size",
        type=int,
        default=None,
        help=(
            "Override GP-EI raw samples. By default GP-EI and TuRBO share "
            "min(5000, max(2048, 200*d)); task benchmark protocols take precedence."
        ),
    )
    parser.add_argument("--gp-ei-startup-trials", type=int, default=2)
    parser.add_argument("--gp-ei-xi", type=float, default=0.0)
    parser.add_argument("--gp-ei-alpha", type=float, default=1e-6)
    parser.add_argument("--gp-ei-n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--git-bo-subspace-dim", type=int, default=10)
    parser.add_argument("--git-bo-beta", type=float, default=2.33)
    parser.add_argument("--git-bo-n-candidates", type=int, default=None)
    parser.add_argument("--git-bo-n-estimators", type=int, default=1)
    parser.add_argument("--git-bo-inference-batch-size", type=int, default=128)
    parser.add_argument("--git-bo-device", default="auto")
    parser.add_argument(
        "--molecular-initial-smiles-path",
        type=Path,
        default=None,
        help="SMILES file used as the explicit initial population for graph_ga/gpbo.",
    )
    parser.add_argument("--kimi-api-key-env", default=DEFAULT_KIMI_API_KEY_ENV)
    parser.add_argument("--kimi-base-url", default=DEFAULT_KIMI_BASE_URL)
    parser.add_argument("--kimi-model", default=DEFAULT_KIMI_MODEL)
    parser.add_argument("--pablo-provider", default="mock", choices=["mock", "openai-compatible", "kimi", "sglang"])
    parser.add_argument("--pablo-base-url", default=None)
    parser.add_argument("--pablo-api-key-env", default="PABLO_API_KEY")
    parser.add_argument("--pablo-global-model", default=None)
    parser.add_argument("--pablo-worker-model", default=None)
    parser.add_argument("--pablo-planner-model", default=None)
    parser.add_argument("--pablo-explorer-model", default=None)
    parser.add_argument("--pablo-model", default="gpt-4.1-mini")
    parser.add_argument("--pablo-init-points", type=int, default=4)
    parser.add_argument("--pablo-max-fails", type=int, default=3)
    parser.add_argument("--pablo-num-seeds", type=int, default=2)
    parser.add_argument("--pablo-max-tasks", type=int, default=20)
    parser.add_argument("--pablo-enable-explorer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pablo-enable-planner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pablo-enable-worker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llambo-backend", choices=["heuristic", "openai", "kimi"], default="heuristic")
    parser.add_argument("--llambo-model", default="gpt-4o-mini")
    parser.add_argument("--llambo-initial-samples", type=int, default=5)
    parser.add_argument("--llambo-candidates", type=int, default=8)
    parser.add_argument("--llambo-templates", type=int, default=2)
    parser.add_argument("--llambo-predictions", type=int, default=6)
    parser.add_argument("--llambo-alpha", type=float, default=-0.2)
    parser.add_argument("--llambo-openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llambo-openai-base-url", default=None)
    parser.add_argument("--llambo-openai-organization", default=None)
    parser.add_argument("--llambo-openai-project", default=None)
    parser.add_argument("--llambo-openai-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--llambo-openai-max-retries", type=int, default=3)
    parser.add_argument(
        "--llambo-openai-include-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send the OpenAI seed field. Disable for compatible endpoints that reject it.",
    )
    parser.add_argument(
        "--llambo-openai-include-store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send store=false. Disable for compatible endpoints that reject it.",
    )
    parser.add_argument(
        "--llambo-openai-use-structured-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use json_schema structured outputs if supported by the endpoint. Disable for older compatible APIs.",
    )
    parser.add_argument("--opro-backend", choices=["heuristic", "openai", "kimi"], default="heuristic")
    parser.add_argument("--opro-model", default="gpt-4o-mini")
    parser.add_argument("--opro-initial-samples", type=int, default=5)
    parser.add_argument("--opro-candidates", type=int, default=8)
    parser.add_argument(
        "--opro-prompt-pairs",
        type=int,
        default=None,
        help="Maximum successful history trials included in OPRO prompts. Default: include all.",
    )
    parser.add_argument("--opro-openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--opro-openai-base-url", default=None)
    parser.add_argument("--opro-openai-organization", default=None)
    parser.add_argument("--opro-openai-project", default=None)
    parser.add_argument("--opro-openai-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--opro-openai-max-retries", type=int, default=3)
    parser.add_argument(
        "--opro-openai-include-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send the OpenAI seed field. Disable for compatible endpoints that reject it.",
    )
    parser.add_argument(
        "--opro-openai-include-store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send store=false. Disable for compatible endpoints that reject it.",
    )
    parser.add_argument("--agent-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--agent-max-retries", type=int, default=1)
    parser.add_argument("--agent-history-limit", type=int, default=40)
    parser.add_argument("--agent-candidates-per-call", type=int, default=1)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--agent-provider", default=None)
    parser.add_argument("--agent-api-base", default=None)
    parser.add_argument("--agent-api-key-env", default=None)
    parser.add_argument(
        "--agent-executable",
        default=None,
        help="Optional harness executable path (for example the Codex CLI binary).",
    )
    parser.add_argument("--agent-initial-random", type=int, default=0)
    parser.add_argument(
        "--agent-tool-mode",
        choices=AGENT_TOOL_MODE_CLI_CHOICES,
        default=None,
        help=(
            "Agent tool mode. Default: no_tool for sealed native Nanobot/Codex/Claude Code harnesses, "
            "function_calling for host-mediated agentic backends."
        ),
    )
    parser.add_argument("--agent-prompt-style", choices=["workspace"], default="workspace")
    parser.add_argument("--agent-max-tool-calls", type=int, default=16)
    parser.add_argument("--agent-enable-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-enable-code-interpreter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-code-backend", choices=["sandboxfusion", "restricted_docker", "local_disabled", "mock"], default="sandboxfusion")
    parser.add_argument("--agent-docker-image", default="agentic-bbo-analysis-sandbox:v1")
    parser.add_argument("--sandbox-fusion-base-url", default=None)
    parser.add_argument(
        "--agent-web-search-provider",
        choices=["disabled", "mock", "serpapi"],
        default="disabled",
    )
    parser.add_argument("--agent-web-search-api-key-env", default=None)
    parser.add_argument("--agent-allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-require-visible-cot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--agent-enable-bbo-skills",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the packaged BBO Nanobot skills into each agent workspace. Disabled by default.",
    )
    parser.add_argument(
        "--agent-skill-path",
        action="append",
        type=Path,
        default=None,
        help="Additional Nanobot skill directory, or directory containing multiple skills, to copy into the agent workspace.",
    )
    parser.add_argument("--skydiscover-interleave-every", type=int, default=5)
    parser.add_argument("--skydiscover-round-iterations", type=int, default=3)
    parser.add_argument("--skydiscover-config", type=Path, default=None)
    parser.add_argument(
        "--skydiscover-runner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run real SkyDiscover Runner (needs `uv sync --extra skydiscover` and LLM credentials).",
    )
    parser.add_argument("--skydiscover-search-type", default="topk")
    parser.add_argument("--skydiscover-model", default=None)
    parser.add_argument("--skydiscover-max-meta-history", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots under run_dir/plots (faster, for CI or headless runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.algorithm == "suite":
        summary = run_demo_suite(
            task_name=args.task,
            seed=args.seed,
            results_root=args.results_root,
            random_evaluations=args.random_evaluations,
            pycma_evaluations=args.pycma_evaluations,
            sigma_fraction=args.sigma_fraction,
            popsize=args.popsize,
            resume=args.resume,
            generate_plots=not args.no_plots,
        )
    else:
        summary = run_single_experiment(
            task_name=args.task,
            algorithm_name=args.algorithm,
            seed=args.seed,
            max_evaluations=args.max_evaluations,
            results_root=args.results_root,
            resume=args.resume,
            sigma_fraction=args.sigma_fraction,
            popsize=args.popsize,
            noise_std=args.noise_std,
            surrogate_path=args.surrogate_path,
            knobs_json_path=args.knobs_json_path,
            generate_plots=not args.no_plots,
            pfns_device=args.pfns_device,
            pfns_pool_size=args.pfns_pool_size,
            pfns_model=args.pfns_model,
            pfns_acquisition=args.pfns_acquisition,
            pfns_custom_model_path=args.pfns_custom_model_path,
            pfns_tabpfn_n_estimators=args.pfns_tabpfn_n_estimators,
            pfns_tabpfn_ignore_pretraining_limits=args.pfns_tabpfn_ignore_pretraining_limits,
            pfns_tabpfn_fit_mode=args.pfns_tabpfn_fit_mode,
            gp_ei_pool_size=args.gp_ei_pool_size,
            gp_ei_startup_trials=args.gp_ei_startup_trials,
            gp_ei_xi=args.gp_ei_xi,
            gp_ei_alpha=args.gp_ei_alpha,
            gp_ei_n_restarts_optimizer=args.gp_ei_n_restarts_optimizer,
            git_bo_subspace_dim=args.git_bo_subspace_dim,
            git_bo_beta=args.git_bo_beta,
            git_bo_n_candidates=args.git_bo_n_candidates,
            git_bo_n_estimators=args.git_bo_n_estimators,
            git_bo_inference_batch_size=args.git_bo_inference_batch_size,
            git_bo_device=args.git_bo_device,
            pablo_provider=args.pablo_provider,
            pablo_base_url=args.pablo_base_url,
            pablo_api_key_env=args.pablo_api_key_env,
            pablo_global_model=args.pablo_global_model,
            pablo_worker_model=args.pablo_worker_model,
            pablo_planner_model=args.pablo_planner_model,
            pablo_explorer_model=args.pablo_explorer_model,
            pablo_model=args.pablo_model,
            pablo_init_points=args.pablo_init_points,
            pablo_max_fails=args.pablo_max_fails,
            pablo_num_seeds=args.pablo_num_seeds,
            pablo_max_tasks=args.pablo_max_tasks,
            pablo_enable_explorer=args.pablo_enable_explorer,
            pablo_enable_planner=args.pablo_enable_planner,
            pablo_enable_worker=args.pablo_enable_worker,
            llambo_backend=args.llambo_backend,
            llambo_model=args.llambo_model,
            llambo_initial_samples=args.llambo_initial_samples,
            llambo_candidates=args.llambo_candidates,
            llambo_templates=args.llambo_templates,
            llambo_predictions=args.llambo_predictions,
            llambo_alpha=args.llambo_alpha,
            llambo_openai_api_key_env=args.llambo_openai_api_key_env,
            llambo_openai_base_url=args.llambo_openai_base_url,
            llambo_openai_organization=args.llambo_openai_organization,
            llambo_openai_project=args.llambo_openai_project,
            llambo_openai_timeout_seconds=args.llambo_openai_timeout_seconds,
            llambo_openai_max_retries=args.llambo_openai_max_retries,
            llambo_openai_use_structured_outputs=args.llambo_openai_use_structured_outputs,
            llambo_openai_include_seed=args.llambo_openai_include_seed,
            llambo_openai_include_store=args.llambo_openai_include_store,
            opro_backend=args.opro_backend,
            opro_model=args.opro_model,
            opro_initial_samples=args.opro_initial_samples,
            opro_candidates=args.opro_candidates,
            opro_prompt_pairs=args.opro_prompt_pairs,
            opro_openai_api_key_env=args.opro_openai_api_key_env,
            opro_openai_base_url=args.opro_openai_base_url,
            opro_openai_organization=args.opro_openai_organization,
            opro_openai_project=args.opro_openai_project,
            opro_openai_timeout_seconds=args.opro_openai_timeout_seconds,
            opro_openai_max_retries=args.opro_openai_max_retries,
            opro_openai_include_seed=args.opro_openai_include_seed,
            opro_openai_include_store=args.opro_openai_include_store,
            kimi_api_key_env=args.kimi_api_key_env,
            kimi_base_url=args.kimi_base_url,
            kimi_model=args.kimi_model,
            agent_timeout_seconds=args.agent_timeout_seconds,
            agent_max_retries=args.agent_max_retries,
            agent_history_limit=args.agent_history_limit,
            agent_candidates_per_call=args.agent_candidates_per_call,
            agent_model=args.agent_model,
            agent_provider=args.agent_provider,
            agent_api_base=args.agent_api_base,
            agent_api_key_env=args.agent_api_key_env,
            agent_executable=args.agent_executable,
            agent_initial_random=args.agent_initial_random,
            agent_tool_mode=args.agent_tool_mode,
            agent_prompt_style=args.agent_prompt_style,
            agent_max_tool_calls=args.agent_max_tool_calls,
            agent_enable_memory=args.agent_enable_memory,
            agent_enable_code_interpreter=args.agent_enable_code_interpreter,
            agent_docker_image=args.agent_docker_image,
            agent_code_backend=args.agent_code_backend,
            sandbox_fusion_base_url=args.sandbox_fusion_base_url,
            agent_web_search_provider=args.agent_web_search_provider,
            agent_web_search_api_key_env=args.agent_web_search_api_key_env,
            agent_allow_fallback=args.agent_allow_fallback,
            agent_require_visible_cot=args.agent_require_visible_cot,
            agent_enable_bbo_skills=args.agent_enable_bbo_skills,
            agent_skill_paths=args.agent_skill_path,
            skydiscover_interleave_every=args.skydiscover_interleave_every,
            skydiscover_round_iterations=args.skydiscover_round_iterations,
            skydiscover_config_path=args.skydiscover_config,
            skydiscover_runner=args.skydiscover_runner,
            skydiscover_search_type=args.skydiscover_search_type,
            skydiscover_model=args.skydiscover_model,
            skydiscover_max_meta_history=args.skydiscover_max_meta_history,
            molecular_initial_smiles_path=args.molecular_initial_smiles_path,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
