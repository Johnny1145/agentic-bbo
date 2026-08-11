"""Unified outer runner for default Nanobot BBO benchmarks.

This module is intentionally policy-light. The default benchmark exposes the
registered task name, task description, and prior knowledge exactly as the core
task defines them. Experiments that restrict those signals should live in
workflow-specific scripts and reuse this module's small helpers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from bbo.algorithms.agentic import (
    AGENT_TOOL_MODE_CLI_CHOICES,
    ClaudeCodeBBOAlgorithm,
    CodexBBOAlgorithm,
    NanobotBBOAlgorithm,
    normalize_agent_tool_mode,
)
from bbo.algorithms.agentic.general_agent_engines import normalize_agent_framework
from bbo.core import (
    ExperimentConfig,
    Experimenter,
    JsonlMetricLogger,
    Task,
    TrialObservation,
    TrialRecord,
)
from bbo.run import DEFAULT_RESULTS_ROOT as CORE_DEFAULT_RESULTS_ROOT
from bbo.run import _allocate_run_dir, _require_algorithm_support, generate_visualizations, run_single_experiment
from bbo.tasks import ALL_TASK_NAMES, TASK_FAMILIES


DEFAULT_RESULTS_ROOT = CORE_DEFAULT_RESULTS_ROOT.parent / "nanobot_benchmark"
NANOBOT_ALGORITHM = "nanobot"
HARNESS_ALGORITHM_NAMES = {
    "nanobot": "nanobot",
    "codex": "codex",
    "claude_code": "claude_code",
}
HARNESS_ALGORITHM_FACTORIES = {
    "nanobot": NanobotBBOAlgorithm,
    "codex": CodexBBOAlgorithm,
    "claude_code": ClaudeCodeBBOAlgorithm,
}
SKILL_MODES = ("no-skill", "skill")
NUMERIC_EVIDENCE_TOOL_NAMES = frozenset(
    {
        "summarize_objective_metrics",
        "get_history_overview",
        "compare_trials",
        "find_nearest_trials",
        "estimate_local_effects",
        "measure_search_coverage",
        "fit_and_check_surrogate",
        "score_virtual_candidates",
        "analyze_history",
        "profile_history_quality",
        "analyze_convergence",
        "rank_parameter_importance",
        "analyze_parameter_interactions",
        "locate_promising_regions",
        "locate_underexplored_regions",
        "recommend_search_regions",
    }
)


@dataclass(frozen=True)
class NanobotTaskOptions:
    """Task-level options exposed by the default outer runner."""

    noise_std: float = 0.0
    surrogate_path: Path | None = None
    knobs_json_path: Path | None = None
    molecular_initial_smiles_path: Path | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "NanobotTaskOptions":
        return cls(
            noise_std=float(args.noise_std),
            surrogate_path=args.surrogate_path,
            knobs_json_path=args.knobs_json_path,
            molecular_initial_smiles_path=args.molecular_initial_smiles_path,
        )


@dataclass(frozen=True)
class HarnessAgentOptions:
    """Agent settings shared by the native-harness CLIs and workflows."""

    framework: str = "nanobot"
    timeout_seconds: float = 180.0
    max_retries: int = 1
    history_limit: int = 40
    candidates_per_call: int = 1
    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    executable: str | None = None
    initial_random: int = 0
    tool_mode: str = "workspace_json"
    prompt_style: str = "workspace"
    prompt_profile: str | None = None
    max_tool_calls: int = 16
    enable_memory: bool = True
    enable_code_interpreter: bool = True
    code_backend: str = "sandboxfusion"
    docker_image: str = "agentic-bbo-analysis-sandbox:v1"
    sandbox_fusion_base_url: str | None = None
    web_search_provider: str = "disabled"
    web_search_api_key_env: str | None = None
    allow_fallback: bool = True
    require_visible_cot: bool = False
    skill_paths: tuple[Path, ...] = ()
    enabled_tool_names: tuple[str, ...] | None = None
    optimizer_backend_allowlist: tuple[str, ...] = ()
    optimizer_max_calls_per_round: int = 3
    experiment_condition: str = "default"
    require_analysis_evidence_per_round: bool = False
    required_tool_names_per_round: tuple[str, ...] = ()
    require_candidate_validation_per_round: bool = False
    require_optimizer_decision_per_round: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "HarnessAgentOptions":
        return cls(
            framework=getattr(args, "harness", "nanobot"),
            timeout_seconds=float(args.agent_timeout_seconds),
            max_retries=int(args.agent_max_retries),
            history_limit=int(args.agent_history_limit),
            candidates_per_call=int(args.agent_candidates_per_call),
            model=args.agent_model,
            provider=args.agent_provider,
            api_base=args.agent_api_base,
            api_key_env=args.agent_api_key_env,
            executable=getattr(args, "agent_executable", None),
            initial_random=int(args.initial_random),
            tool_mode=normalize_agent_tool_mode(args.agent_tool_mode),
            prompt_style=args.agent_prompt_style,
            prompt_profile=getattr(args, "agent_prompt_profile", None),
            max_tool_calls=int(args.agent_max_tool_calls),
            enable_memory=bool(args.agent_enable_memory),
            enable_code_interpreter=bool(args.agent_enable_code_interpreter),
            docker_image=getattr(args, "agent_docker_image", "agentic-bbo-analysis-sandbox:v1"),
            code_backend=args.agent_code_backend,
            sandbox_fusion_base_url=args.sandbox_fusion_base_url,
            web_search_provider=args.agent_web_search_provider,
            web_search_api_key_env=args.agent_web_search_api_key_env,
            allow_fallback=bool(args.agent_allow_fallback),
            require_visible_cot=bool(args.agent_require_visible_cot),
            experiment_condition="default",
            require_analysis_evidence_per_round=False,
            required_tool_names_per_round=(),
            require_candidate_validation_per_round=False,
            require_optimizer_decision_per_round=False,
            skill_paths=tuple(args.agent_skill_path or ()),
            enabled_tool_names=None,
            optimizer_backend_allowlist=(),
            optimizer_max_calls_per_round=3,
        )

    def run_single_kwargs(self, *, skill_mode: str) -> dict[str, Any]:
        self._validate_skill_paths(skill_mode)
        return {
            "agent_timeout_seconds": self.timeout_seconds,
            "agent_max_retries": self.max_retries,
            "agent_history_limit": self.history_limit,
            "agent_candidates_per_call": self.candidates_per_call,
            "agent_model": self.model,
            "agent_provider": self.provider,
            "agent_api_base": self.api_base,
            "agent_api_key_env": self.api_key_env,
            "agent_executable": self.executable,
            "agent_initial_random": self.initial_random,
            "agent_tool_mode": normalize_agent_tool_mode(self.tool_mode),
            "agent_prompt_style": self.prompt_style,
            "agent_prompt_profile": self.prompt_profile,
            "agent_max_tool_calls": self.max_tool_calls,
            "agent_enable_memory": self.enable_memory,
            "agent_docker_image": self.docker_image,
            "agent_enable_code_interpreter": self.enable_code_interpreter,
            "agent_code_backend": self.code_backend,
            "sandbox_fusion_base_url": self.sandbox_fusion_base_url,
            "agent_web_search_provider": self.web_search_provider,
            "agent_web_search_api_key_env": self.web_search_api_key_env,
            "agent_allow_fallback": self.allow_fallback,
            "agent_experiment_condition": self.experiment_condition,
            "agent_require_analysis_evidence_per_round": self.require_analysis_evidence_per_round,
            "agent_required_tool_names_per_round": list(self.required_tool_names_per_round),
            "agent_require_candidate_validation_per_round": self.require_candidate_validation_per_round,
            "agent_require_optimizer_decision_per_round": self.require_optimizer_decision_per_round,
            "agent_require_visible_cot": self.require_visible_cot,
            "agent_enable_bbo_skills": skill_mode == "skill",
            "agent_skill_paths": list(self.skill_paths),
            "agent_enabled_tool_names": self.enabled_tool_names,
            "agent_optimizer_backend_allowlist": list(
                self.optimizer_backend_allowlist
            ),
            "agent_optimizer_max_calls_per_round": self.optimizer_max_calls_per_round,
        }

    def build_algorithm(
        self,
        *,
        run_dir: Path,
        seed: int,
        skill_mode: str,
        algorithm_name: str,
        resume: bool = False,
    ) -> NanobotBBOAlgorithm | CodexBBOAlgorithm | ClaudeCodeBBOAlgorithm:
        del seed
        framework = normalize_agent_framework(self.framework)
        expected_algorithm = HARNESS_ALGORITHM_NAMES.get(framework)
        if expected_algorithm is None:
            raise ValueError(f"Unsupported native harness `{self.framework}`.")
        if normalize_agent_framework(algorithm_name) != framework:
            raise ValueError(
                f"Harness `{framework}` must use algorithm `{expected_algorithm}`, got `{algorithm_name}`."
            )
        self._validate_skill_paths(skill_mode)
        factory = HARNESS_ALGORITHM_FACTORIES[framework]
        return factory(
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            history_limit=self.history_limit,
            candidates_per_call=self.candidates_per_call,
            model=self.model,
            provider=self.provider,
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            executable=self.executable,
            initial_random=self.initial_random,
            run_dir=run_dir,
            resume=resume,
            tool_mode=normalize_agent_tool_mode(self.tool_mode),
            prompt_style=self.prompt_style,
            prompt_profile=self.prompt_profile,
            max_tool_calls=self.max_tool_calls,
            enable_memory=self.enable_memory,
            experiment_condition=self.experiment_condition,
            require_analysis_evidence_per_round=self.require_analysis_evidence_per_round,
            required_tool_names_per_round=self.required_tool_names_per_round,
            require_candidate_validation_per_round=self.require_candidate_validation_per_round,
            docker_image=self.docker_image,
            require_optimizer_decision_per_round=self.require_optimizer_decision_per_round,
            enable_code_interpreter=self.enable_code_interpreter,
            code_backend=self.code_backend,
            sandbox_fusion_base_url=self.sandbox_fusion_base_url,
            web_search_provider=self.web_search_provider,
            web_search_api_key_env=self.web_search_api_key_env,
            allow_fallback=self.allow_fallback,
            require_visible_cot=self.require_visible_cot,
            enable_bbo_skills=skill_mode == "skill",
            skill_paths=list(self.skill_paths),
            enabled_tool_names=self.enabled_tool_names,
            optimizer_backend_allowlist=self.optimizer_backend_allowlist,
            optimizer_max_calls_per_round=self.optimizer_max_calls_per_round,
        )

    def _validate_skill_paths(self, skill_mode: str) -> None:
        if skill_mode == "no-skill" and self.skill_paths:
            raise ValueError("`--agent-skill-path` requires `--skill-mode skill` or `--skill-modes skill`.")
        if skill_mode == "skill" and normalize_agent_tool_mode(self.tool_mode) == "no_tool":
            raise ValueError("`--skill-mode skill` requires `--agent-tool-mode workspace_json` or `function_calling`.")

    @property
    def algorithm_name(self) -> str:
        framework = normalize_agent_framework(self.framework)
        try:
            return HARNESS_ALGORITHM_NAMES[framework]
        except KeyError as exc:
            raise ValueError(f"Unsupported native harness `{self.framework}`.") from exc


# Backwards-compatible public name used by the existing Nanobot entrypoint.
NanobotAgentOptions = HarnessAgentOptions


@dataclass(frozen=True)
class NanobotCase:
    task_name: str
    seed: int
    skill_mode: str

    @property
    def variant(self) -> str:
        return self.skill_mode


def parse_csv_tokens(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw)
    tokens: list[str] = []
    for value in values:
        for token in str(value).split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tuple(tokens)


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in parse_csv_tokens(raw))
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def resolve_task_names(*, tasks: str, task_families: Sequence[str] = ()) -> tuple[str, ...]:
    selected: list[str] = []
    task_tokens = parse_csv_tokens(tasks)
    if not task_tokens:
        raise ValueError("At least one task is required.")
    for token in task_tokens:
        if token == "all":
            selected.extend(ALL_TASK_NAMES)
        elif token.startswith("family:"):
            selected.extend(_family_tasks(token.removeprefix("family:")))
        elif token in ALL_TASK_NAMES:
            selected.append(token)
        else:
            available = ", ".join(ALL_TASK_NAMES)
            raise ValueError(f"Unknown task `{token}`. Available: {available}")
    for family in task_families:
        selected.extend(_family_tasks(family))
    return _unique_preserving_order(selected)


def resolve_skill_modes(raw: str) -> tuple[str, ...]:
    modes: list[str] = []
    for token in parse_csv_tokens(raw):
        if token == "both":
            modes.extend(SKILL_MODES)
        elif token in SKILL_MODES:
            modes.append(token)
        else:
            raise ValueError("Skill modes must be one of: no-skill, skill, both.")
    if not modes:
        raise ValueError("At least one skill mode is required.")
    return _unique_preserving_order(modes)


def planned_cases(*, tasks: Sequence[str], seeds: Sequence[int], skill_modes: Sequence[str]) -> list[NanobotCase]:
    return [
        NanobotCase(task_name=task_name, seed=int(seed), skill_mode=skill_mode)
        for skill_mode in skill_modes
        for task_name in tasks
        for seed in seeds
    ]


def run_default_case(
    case: NanobotCase,
    *,
    results_root: Path,
    max_evaluations: int | None,
    task_options: NanobotTaskOptions,
    agent: HarnessAgentOptions,
    resume: bool,
    generate_plots: bool,
) -> dict[str, Any]:
    summary = run_single_experiment(
        task_name=case.task_name,
        algorithm_name=agent.algorithm_name,
        seed=case.seed,
        max_evaluations=max_evaluations,
        results_root=results_root / "full-prior" / case.variant,
        resume=resume,
        generate_plots=generate_plots,
        noise_std=task_options.noise_std,
        surrogate_path=task_options.surrogate_path,
        knobs_json_path=task_options.knobs_json_path,
        molecular_initial_smiles_path=task_options.molecular_initial_smiles_path,
        **agent.run_single_kwargs(skill_mode=case.skill_mode),
    )
    return augment_summary(summary, case=case, exposure_policy="full-prior")


def run_task_object_case(
    *,
    task: Task,
    output_task_name: str,
    case: NanobotCase,
    results_root: Path,
    agent: HarnessAgentOptions,
    resume: bool,
    generate_plots: bool,
    exposure_policy: str,
    algorithm_name: str = NANOBOT_ALGORITHM,
    experiment_metadata: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
    initial_observations: Sequence[TrialObservation] = (),
    validate_existing_initialization: Callable[[Sequence[TrialRecord]], None] | None = None,
) -> dict[str, Any]:
    _require_algorithm_support(task, algorithm_name)
    run_dir = _allocate_run_dir(
        results_root / exposure_policy / case.variant / output_task_name / algorithm_name / f"seed_{case.seed}",
        resume=resume,
    )
    ensure_run_setting_manifest(run_dir=run_dir, manifest=run_manifest, resume=resume)
    algorithm = agent.build_algorithm(
        run_dir=run_dir,
        seed=case.seed,
        skill_mode=case.skill_mode,
        algorithm_name=algorithm_name,
        resume=resume,
    )
    logger = JsonlMetricLogger(run_dir / "trials.jsonl")
    if initial_observations:
        description = task.get_description()
        logger.bind_run(
            task_spec=task.spec,
            algorithm_name=algorithm.name,
            seed=case.seed,
            description_bundle=description,
        )
        existing_records = logger.load_records()
        if validate_existing_initialization is not None:
            validate_existing_initialization(existing_records)
        existing_initialization = [
            record
            for record in existing_records
            if (
                record.suggestion_metadata.get("phase")
                or record.metadata.get("phase")
            )
            == "initialization"
        ]
        if (
            len(existing_records) > len(existing_initialization)
            and len(existing_initialization) < len(initial_observations)
        ):
            raise RuntimeError(
                "Cannot resume optimization before strict shared initialization "
                "is complete."
            )
        for observation in initial_observations[len(existing_initialization) :]:
            logger.log(observation)
    experiment = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(
            seed=case.seed,
            resume=resume or bool(initial_observations),
            fail_fast_on_sanity=True,
            metadata=dict(experiment_metadata or {}),
        ),
    )
    summary = experiment.run()
    records = logger.load_records()
    best_trial_id = summary.logger_summary.get("best_trial_id")
    best_record = next((record for record in records if record.trial_id == best_trial_id), None)
    serializable_summary: dict[str, Any] = {
        "task_name": summary.task_name,
        "output_task_name": output_task_name,
        "algorithm_name": summary.algorithm_name,
        "seed": summary.seed,
        "n_completed": summary.n_completed,
        "total_eval_time": summary.total_eval_time,
        "best_primary_objective": summary.best_primary_objective,
        "best_trial_id": best_trial_id,
        "best_config": None if best_record is None else dict(best_record.config),
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
        "run_dir": str(run_dir),
        "results_jsonl": str(run_dir / "trials.jsonl"),
        "trial_count": len(records),
        "internal_artifacts": getattr(algorithm, "artifact_paths", {}),
    }
    if generate_plots:
        serializable_summary["plot_paths"] = [
            str(path)
            for path in generate_visualizations(
                task=task,
                logger=logger,
                output_dir=run_dir / "plots",
                algorithm_label=algorithm_name,
            )
        ]
    (run_dir / "summary.json").write_text(
        json.dumps(serializable_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return augment_summary(serializable_summary, case=case, exposure_policy=exposure_policy)


def augment_summary(summary: dict[str, Any], *, case: NanobotCase, exposure_policy: str) -> dict[str, Any]:
    enriched = dict(summary)
    enriched["benchmark_metadata"] = {
        "benchmark_entrypoint": "bbo.benchmark.nanobot",
        "skill_mode": case.skill_mode,
        "variant": case.variant,
        "exposure_policy": exposure_policy,
    }
    run_dir = Path(enriched["run_dir"])
    enriched["tool_usage_summary"] = collect_tool_usage(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_path.write_text(json.dumps(enriched, indent=2, sort_keys=True), encoding="utf-8")
    return enriched


def collect_tool_usage(run_dir: Path) -> dict[str, Any]:
    workspace_counts = _count_workspace_tool_calls(run_dir / "agent_tool_calls.jsonl")
    nanobot_native_counts = _count_nanobot_native_tool_calls(run_dir / "llm_logs")
    structured_native_counts = _count_structured_native_tool_calls(run_dir / "agent_calls.jsonl")
    native_counts = nanobot_native_counts + structured_native_counts
    skill_read_counts = _count_nanobot_skill_reads(run_dir / "llm_logs")
    accepted_skill_counts, accepted_search_intent_counts = _count_accepted_search_actions(run_dir)
    skill_audit_counts = _count_skill_audits(run_dir)
    return {
        "bbo_workspace_tool_calls": sum(workspace_counts.values()),
        "bbo_workspace_tool_counts": dict(sorted(workspace_counts.items())),
        "benchmark_injected_tool_calls": sum(workspace_counts.values()),
        "benchmark_injected_tool_counts": dict(sorted(workspace_counts.items())),
        "bbo_workspace_non_validation_numeric_tool_calls": sum(
            count for name, count in workspace_counts.items() if name in NUMERIC_EVIDENCE_TOOL_NAMES
        ),
        "native_harness_tool_calls": sum(native_counts.values()),
        "native_harness_tool_counts": dict(sorted(native_counts.items())),
        "nanobot_native_tool_calls": sum(nanobot_native_counts.values()),
        "nanobot_native_tool_counts": dict(sorted(nanobot_native_counts.items())),
        "skill_read_counts": dict(sorted(skill_read_counts.items())),
        "accepted_skill_counts": dict(sorted(accepted_skill_counts.items())),
        "accepted_search_intent_counts": dict(sorted(accepted_search_intent_counts.items())),
        "skill_evidence_failure_count": _count_skill_evidence_failures(run_dir / "agent_calls.jsonl"),
        **skill_audit_counts,
    }


def ensure_run_setting_manifest(
    *,
    run_dir: Path,
    manifest: dict[str, Any] | None,
    resume: bool,
) -> None:
    """Persist the exact run setting and reject incompatible resume attempts."""

    if manifest is None:
        return
    path = run_dir / "run_setting.json"
    canonical = json.dumps(manifest, default=str, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        try:
            existing_canonical = json.dumps(
                json.loads(existing),
                default=str,
                indent=2,
                sort_keys=True,
            ) + "\n"
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid existing run setting manifest: {path}") from exc
        if existing_canonical != canonical:
            raise ValueError(
                f"Refusing to resume `{run_dir}` with a different run setting. "
                f"Use a new results directory or restore the original setting in `{path}`."
            )
        return
    if resume and any(
        candidate.exists()
        for candidate in (
            run_dir / "trials.jsonl",
            run_dir / "state.json",
            run_dir / "agent_calls.jsonl",
            run_dir / "summary.json",
        )
    ):
        raise ValueError(
            f"Refusing to resume legacy/incomplete run `{run_dir}` without `{path.name}`. "
            "Use a new results directory."
        )
    path.write_text(canonical, encoding="utf-8")


def run_cases(
    cases: Sequence[NanobotCase],
    *,
    results_root: Path,
    max_evaluations: int | None,
    task_options: NanobotTaskOptions | None = None,
    agent: HarnessAgentOptions,
    resume: bool = False,
    generate_plots: bool = True,
    dry_run: bool = False,
    runner: Callable[[NanobotCase], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    planned = [case.__dict__ for case in cases]
    if dry_run:
        payload = {
            "dry_run": True,
            "harness": normalize_agent_framework(agent.framework),
            "algorithm_name": agent.algorithm_name,
            "planned_cases": planned,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        try:
            if runner is None:
                result = run_default_case(
                    case,
                    results_root=results_root,
                    max_evaluations=max_evaluations,
                    task_options=task_options or NanobotTaskOptions(),
                    agent=agent,
                    resume=resume,
                    generate_plots=generate_plots,
                )
            else:
                result = runner(case)
            results.append(result)
        except Exception as exc:  # pragma: no cover - exercised by real benchmark failures.
            failures.append(
                {
                    "task_name": case.task_name,
                    "seed": case.seed,
                    "skill_mode": case.skill_mode,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    payload = {"dry_run": False, "planned_cases": planned, "results": results, "failures": failures}
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def effective_max_evaluations(
    *,
    max_evaluations: int | None,
    initial_random: int,
    optimizer_budget: int | None,
) -> int | None:
    if max_evaluations is not None and optimizer_budget is not None:
        raise ValueError("Use either `--max-evaluations` or `--optimizer-budget`, not both.")
    if optimizer_budget is None:
        return max_evaluations
    if optimizer_budget <= 0:
        raise ValueError("`--optimizer-budget` must be positive.")
    return int(initial_random) + int(optimizer_budget)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run default full-prior native-harness BBO benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tasks", help="List registered benchmark tasks.")
    list_parser.add_argument("--task-family", choices=["all", *sorted(TASK_FAMILIES)], default="all")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    run_parser = subparsers.add_parser("run", help="Run one task/seed/skill-mode case.")
    run_parser.add_argument("--task", default="bbob_f01_d10", choices=ALL_TASK_NAMES)
    run_parser.add_argument("--seed", type=int, default=1)
    run_parser.add_argument("--skill-mode", choices=SKILL_MODES, default="no-skill")
    _add_common_run_args(run_parser)

    matrix_parser = subparsers.add_parser("matrix", help="Run a task x seed x skill-mode matrix.")
    matrix_parser.add_argument("--tasks", default="bbob_f01_d10", help="Comma list, `all`, or `family:<name>`.")
    matrix_parser.add_argument(
        "--task-family",
        action="append",
        choices=sorted(TASK_FAMILIES),
        default=[],
        help="Add all tasks from a registered family. Can be repeated.",
    )
    matrix_parser.add_argument("--seeds", default="1")
    matrix_parser.add_argument("--skill-modes", default="no-skill", help="Comma list: no-skill, skill, or both.")
    _add_common_run_args(matrix_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "list-tasks":
        tasks = ALL_TASK_NAMES if args.task_family == "all" else TASK_FAMILIES[args.task_family]
        if args.json:
            print(json.dumps({"task_family": args.task_family, "tasks": list(tasks)}, indent=2, sort_keys=True))
        else:
            for task_name in tasks:
                print(task_name)
        return 0

    agent = NanobotAgentOptions.from_namespace(args)
    task_options = NanobotTaskOptions.from_namespace(args)
    max_evaluations = effective_max_evaluations(
        max_evaluations=args.max_evaluations,
        initial_random=args.initial_random,
        optimizer_budget=args.optimizer_budget,
    )
    if args.command == "run":
        cases = [NanobotCase(task_name=args.task, seed=args.seed, skill_mode=args.skill_mode)]
    else:
        cases = planned_cases(
            tasks=resolve_task_names(tasks=args.tasks, task_families=args.task_family),
            seeds=parse_int_csv(args.seeds),
            skill_modes=resolve_skill_modes(args.skill_modes),
        )
    payload = run_cases(
        cases,
        results_root=args.results_root,
        max_evaluations=max_evaluations,
        task_options=task_options,
        agent=agent,
        resume=args.resume,
        generate_plots=not args.no_plots,
        dry_run=args.dry_run,
    )
    return 1 if payload.get("failures") else 0


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--harness",
        choices=tuple(HARNESS_ALGORITHM_NAMES),
        default="nanobot",
        help="Native agent harness. Defaults to Nanobot for backwards compatibility.",
    )
    parser.add_argument("--initial-random", type=int, default=0)
    parser.add_argument("--optimizer-budget", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--surrogate-path", type=Path, default=None)
    parser.add_argument("--knobs-json-path", type=Path, default=None)
    parser.add_argument("--molecular-initial-smiles-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--agent-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--agent-max-retries", type=int, default=1)
    parser.add_argument("--agent-history-limit", type=int, default=40)
    parser.add_argument("--agent-candidates-per-call", type=int, default=1)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--agent-provider", default=None)
    parser.add_argument("--agent-api-base", default=None)
    parser.add_argument("--agent-api-key-env", default=None)
    parser.add_argument("--agent-executable", default=None)
    parser.add_argument("--agent-tool-mode", choices=AGENT_TOOL_MODE_CLI_CHOICES, default="workspace_json")
    parser.add_argument("--agent-prompt-style", choices=["workspace"], default="workspace")
    parser.add_argument("--agent-prompt-profile", default=None)
    parser.add_argument("--agent-max-tool-calls", type=int, default=16)
    parser.add_argument("--agent-enable-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-enable-code-interpreter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-code-backend", choices=["sandboxfusion", "restricted_docker", "local_disabled", "mock"], default="sandboxfusion")
    parser.add_argument("--agent-docker-image", default="agentic-bbo-analysis-sandbox:v1")
    parser.add_argument("--sandbox-fusion-base-url", default=None)
    parser.add_argument("--agent-web-search-provider", choices=["disabled", "mock", "serpapi"], default="disabled")
    parser.add_argument("--agent-web-search-api-key-env", default=None)
    parser.add_argument("--agent-allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-require-visible-cot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agent-skill-path", action="append", type=Path, default=None)


def _family_tasks(family: str) -> tuple[str, ...]:
    if family not in TASK_FAMILIES:
        available = ", ".join(sorted(TASK_FAMILIES))
        raise ValueError(f"Unknown task family `{family}`. Available: {available}")
    return TASK_FAMILIES[family]


def _unique_preserving_order(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _count_workspace_tool_calls(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            counts["<invalid-json>"] += 1
            continue
        counts[str(payload.get("tool_name") or "<unknown>")] += 1
    return counts


def _count_nanobot_native_tool_calls(log_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not log_dir.exists():
        return counts
    for path in sorted(log_dir.glob("**/*_agent-end.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    counts["<unknown>"] += 1
                    continue
                counts[str(function.get("name") or "<unknown>")] += 1
    return counts


def _count_structured_native_tool_calls(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        llm_log = payload.get("llm_log")
        if not isinstance(llm_log, dict):
            continue
        tool_calls = llm_log.get("nativeToolCalls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                counts["<unknown>"] += 1
                continue
            counts[str(tool_call.get("name") or tool_call.get("type") or "<unknown>")] += 1
    return counts


def _count_nanobot_skill_reads(log_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not log_dir.exists():
        return counts
    for path in sorted(log_dir.glob("**/*_agent-end.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                skill = _skill_name_from_read_file_tool_call(tool_call)
                if skill:
                    counts[skill] += 1
    return counts


def _skill_name_from_read_file_tool_call(tool_call: object) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict) or function.get("name") != "read_file":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str):
        return None
    parts = [part for part in raw_path.replace("\\", "/").split("/") if part]
    for index in range(len(parts) - 2):
        if parts[index] == "skills" and parts[index + 2] == "SKILL.md":
            return parts[index + 1]
    return None


def _count_accepted_search_actions(run_dir: Path) -> tuple[Counter[str], Counter[str]]:
    skill_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    used_agent_calls = False
    calls_path = run_dir / "agent_calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            actions = record.get("accepted_search_actions")
            if not isinstance(actions, list):
                continue
            used_agent_calls = True
            for action in actions:
                _count_search_action(action, skill_counts, intent_counts)
    if used_agent_calls:
        return skill_counts, intent_counts
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        return skill_counts, intent_counts
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trial = json.loads(line)
        except json.JSONDecodeError:
            continue
        metadata = trial.get("suggestion_metadata") or {}
        action = metadata.get("search_action") if isinstance(metadata, dict) else None
        if not isinstance(action, dict):
            action = metadata if isinstance(metadata, dict) else {}
        _count_search_action(action, skill_counts, intent_counts)
    return skill_counts, intent_counts


def _count_search_action(action: object, skill_counts: Counter[str], intent_counts: Counter[str]) -> None:
    if not isinstance(action, dict):
        return
    skill = action.get("skill") or action.get("primary_skill")
    if skill and str(skill).lower() not in {"none", "null"}:
        skill_counts[str(skill)] += 1
    intent = action.get("search_intent") or action.get("agent_search_intent")
    if intent:
        intent_counts[str(intent)] += 1


def _count_skill_audits(run_dir: Path) -> dict[str, Any]:
    audit_count = 0
    compliance_counts: Counter[str] = Counter()
    declared_counts: Counter[str] = Counter()
    inferred_counts: Counter[str] = Counter()
    read_counts: Counter[str] = Counter()
    read_but_not_declared = 0
    for audit in _iter_accepted_skill_audits(run_dir):
        audit_count += 1
        compliance = audit.get("compliance")
        if compliance:
            compliance_counts[str(compliance)] += 1
        declared = audit.get("declared_skill")
        if declared:
            declared_counts[str(declared)] += 1
        inferred = audit.get("inferred_skill")
        if inferred:
            inferred_counts[str(inferred)] += 1
        read_skills = audit.get("read_skills")
        if isinstance(read_skills, list):
            for skill in read_skills:
                if skill:
                    read_counts[str(skill)] += 1
            if read_skills and not declared:
                read_but_not_declared += 1
    return {
        "skill_audit_count": audit_count,
        "skill_audit_compliance_counts": dict(sorted(compliance_counts.items())),
        "skill_audit_declared_skill_counts": dict(sorted(declared_counts.items())),
        "skill_audit_inferred_skill_counts": dict(sorted(inferred_counts.items())),
        "skill_audit_read_skill_counts": dict(sorted(read_counts.items())),
        "skill_audit_read_but_not_declared_count": read_but_not_declared,
    }


def _iter_accepted_skill_audits(run_dir: Path) -> Iterable[dict[str, Any]]:
    used_agent_calls = False
    calls_path = run_dir / "agent_calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            actions = record.get("accepted_search_actions")
            if not isinstance(actions, list):
                continue
            used_agent_calls = True
            for action in actions:
                if isinstance(action, dict) and isinstance(action.get("skill_audit"), dict):
                    yield action["skill_audit"]
    if used_agent_calls:
        return
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        return
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trial = json.loads(line)
        except json.JSONDecodeError:
            continue
        metadata = trial.get("suggestion_metadata") or {}
        action = metadata.get("search_action") if isinstance(metadata, dict) else None
        audit = action.get("skill_audit") if isinstance(action, dict) else None
        if isinstance(audit, dict):
            yield audit


def _count_skill_evidence_failures(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        error = str(record.get("validation_error") or "")
        if "required BBO evidence tools" in error or "must not be the primary skill" in error:
            count += 1
    return count


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
