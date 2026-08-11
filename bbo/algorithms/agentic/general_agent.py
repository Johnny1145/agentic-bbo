"""General coding-agent optimizer for black-box optimization tasks."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core import (
    Incumbent,
    ObjectiveDirection,
    SearchSpace,
    load_BBO_manifest,
    TaskDescriptionBundle,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from ...core.algo import Algorithm
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization
from .general_agent_engines import (
    AgentResult,
    AgentWorkCopy,
    GeneralAgentEngine,
    create_general_agent_engine,
    normalize_agent_framework,
)
from .serialization import append_jsonl, dump_json, stable_config_identity, to_jsonable
from .tools import (
    BBOMemoryStore,
    BBOToolCallLogger,
    BBOToolContext,
    BBOToolRegistry,
    BBOWebSourceLogger,
    CodeInterpreterTool,
    DisabledBBOCodeBackend,
    FetchURLTool,
    DockerBBOCodeBackend,
    MockBBOCodeBackend,
    OPTIMIZER_ACTION_TOOLS,
    OPTIMIZER_DECISION_TOOLS,
    SandboxFusionBBOCodeBackend,
    WebSearchTool,
    create_BBO_web_search_provider,
    create_optimizer_tools,
    create_core_BBO_tools,
)
from .tools.core_tools import agent_visible_config, agent_visible_metadata, agent_visible_metrics, agent_visible_payload
from .agent_candidate import GeneralAgentValidationError, ParsedAgentCandidate, _paired_xy_parameter_count, _retry_feedback_block, parse_agent_candidate_payload, search_space_schema
from .agent_skill_audit import BBO_NANOBOT_SKILL_NAMES, NANOBOT_BUILTIN_SKILL_NAMES, BBO_NUMERIC_EVIDENCE_TOOLS, BBO_REGION_EVIDENCE_TOOLS, BBO_REGION_JOINT_SUPPORT_TOOLS, MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES, NON_PROPOSAL_BBO_SKILLS, SKILL_EVIDENCE_TOOL_GROUPS, SKILL_TO_SEARCH_INTENT, _NANOBOT_SKILL_NAME_RE, _search_action_metadata, _declared_agent_skill_names, _nanobot_read_skill_names_for_call, _bbo_workspace_tool_names_for_call, _bbo_tool_names_from_nanobot_session, _build_skill_usage_audit, _format_tool_group

from .prompt_profiles import PromptProfile, WorkflowPromptProfile, resolve_workflow_prompt_profile

DEFAULT_AGENT_TIMEOUT_SECONDS = 300.0
DEFAULT_AGENT_HISTORY_LIMIT = 40
DEFAULT_AGENT_CANDIDATES_PER_CALL = 1
FINAL_CANDIDATE_FILENAME = "final_candidate.json"
AGENT_TOOL_MODES = ("function_calling", "workspace_json", "no_tool")
AGENT_TOOL_MODE_CLI_CHOICES = (
    "function_calling",
    "workspace_json",
    "no_tool",
    "no-tool",
    "no_tools",
    "no-tools",
    "none",
    "disabled",
)


def _call_id_scope(call_id: str | Sequence[str]) -> frozenset[str]:
    if isinstance(call_id, str):
        return frozenset((call_id,))
    return frozenset(str(value) for value in call_id)


@dataclass
class AgentCandidateEntry:
    """Queued candidate ready to be surfaced through ask()."""

    config: dict[str, Any]
    call_id: str
    candidate_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneralAgentConfig:
    """Configuration for the general-agent optimizer."""

    framework: str
    algorithm_name: str
    timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS
    max_retries: int = 1
    history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT
    candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL
    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    executable: str | None = None
    initial_random: int = 0
    run_dir: Path | None = None
    resume: bool = False
    tool_mode: str = "function_calling"
    prompt_style: str = "workspace"
    role_name: str = "proposer"
    prompt_profile: PromptProfile = field(default_factory=lambda: resolve_workflow_prompt_profile("general_bbo", roles={"proposer"}, single_role="proposer").for_role("proposer"))
    max_tool_calls: int = 16
    enable_memory: bool = True
    enable_code_interpreter: bool = True
    docker_image: str = "agentic-bbo-analysis-sandbox:v1"
    code_backend: str = "sandboxfusion"
    sandbox_fusion_base_url: str | None = None
    web_search_provider: str = "disabled"
    web_search_api_key_env: str | None = None
    allow_fallback: bool = True
    require_visible_cot: bool = False
    enable_bbo_skills: bool = False
    skill_paths: tuple[Path, ...] = field(default_factory=tuple)
    enabled_tool_names: tuple[str, ...] | None = None
    optimizer_backend_allowlist: tuple[str, ...] = field(default_factory=tuple)
    optimizer_max_calls_per_round: int = 3
    experiment_condition: str = "default"
    require_analysis_evidence_per_round: bool = False
    required_tool_names_per_round: tuple[str, ...] = field(default_factory=tuple)
    require_candidate_validation_per_round: bool = False
    require_optimizer_decision_per_round: bool = False


class GeneralAgentBBOAlgorithm(Algorithm):
    """Ask/tell wrapper that lets an external general agent propose configs."""

    def __init__(
        self,
        *,
        framework: str,
        algorithm_name: str | None = None,
        engine: GeneralAgentEngine | None = None,
        timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
        max_retries: int = 1,
        history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT,
        candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL,
        model: str | None = None,
        provider: str | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        executable: str | None = None,
        initial_random: int = 0,
        run_dir: Path | str | None = None,
        resume: bool = False,
        tool_mode: str = "function_calling",
        prompt_style: str = "workspace",
        prompt_profile: str | PromptProfile | WorkflowPromptProfile | None = None,
        role_name: str = "proposer",
        max_tool_calls: int = 16,
        enable_memory: bool = True,
        docker_image: str = "agentic-bbo-analysis-sandbox:v1",
        enable_code_interpreter: bool = True,
        code_backend: str = "sandboxfusion",
        sandbox_fusion_base_url: str | None = None,
        web_search_provider: str = "disabled",
        web_search_api_key_env: str | None = None,
        allow_fallback: bool = True,
        require_visible_cot: bool = False,
        enable_bbo_skills: bool = False,
        experiment_condition: str = "default",
        require_analysis_evidence_per_round: bool = False,
        required_tool_names_per_round: Sequence[str] = (),
        require_candidate_validation_per_round: bool = False,
        require_optimizer_decision_per_round: bool = False,
        skill_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        enabled_tool_names: Sequence[str] | None = None,
        optimizer_backend_allowlist: Sequence[str] = (),
        optimizer_max_calls_per_round: int = 3,
    ) -> None:
        normalized = normalize_agent_framework(framework)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if history_limit < 0:
            raise ValueError("history_limit must be non-negative.")
        if candidates_per_call <= 0:
            raise ValueError("candidates_per_call must be positive.")
        if initial_random < 0:
            raise ValueError("initial_random must be non-negative.")
        normalized_tool_mode = normalize_agent_tool_mode(tool_mode)
        normalized_prompt_style = prompt_style.strip().lower().replace("-", "_")
        normalized_role_name = str(role_name).strip()
        if not normalized_role_name:
            raise ValueError("role_name must be non-empty.")
        workflow_prompts = resolve_workflow_prompt_profile(
            "general_bbo" if prompt_profile is None else prompt_profile,
            roles={normalized_role_name}, single_role=normalized_role_name,
        )
        if normalized_prompt_style != "workspace":
            raise ValueError("prompt_style must be `workspace`.")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative.")
        if optimizer_max_calls_per_round < 0:
            raise ValueError("optimizer_max_calls_per_round must be non-negative.")
        normalized_skill_paths = _normalize_skill_paths(skill_paths)
        normalized_tool_names = (
            None
            if enabled_tool_names is None
            else tuple(
                dict.fromkeys(
                    str(name).strip()
                    for name in enabled_tool_names
                    if str(name).strip()
                )
            )
        )
        normalized_optimizer_backends = tuple(
            dict.fromkeys(
                str(name).strip().lower().replace("-", "_")
                for name in optimizer_backend_allowlist
                if str(name).strip()
            )
        )
        normalized_required_tools = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in required_tool_names_per_round
                if str(name).strip()
            )
        )
        if normalized_tool_names is not None:
            unavailable_required = sorted(
                set(normalized_required_tools) - set(normalized_tool_names)
            )
            if unavailable_required:
                raise ValueError(
                    "Required per-round tools must be enabled: "
                    + ", ".join(unavailable_required)
                )
        if normalized_tool_mode == "no_tool" and (enable_bbo_skills or normalized_skill_paths):
            raise ValueError("BBO skills require `tool_mode` to be `workspace_json` or `function_calling`.")
        if normalized_tool_mode == "no_tool" and (
            normalized_tool_names or normalized_optimizer_backends
        ):
            raise ValueError("Tool allowlists require workspace_json or function_calling mode.")

        self.config = GeneralAgentConfig(
            framework=normalized,
            algorithm_name=algorithm_name or f"agentic_{normalized}",
            timeout_seconds=float(timeout_seconds),
            max_retries=int(max_retries),
            history_limit=int(history_limit),
            candidates_per_call=int(candidates_per_call),
            model=model,
            provider=provider,
            api_base=api_base,
            api_key_env=api_key_env,
            executable=executable,
            initial_random=int(initial_random),
            run_dir=None if run_dir is None else Path(run_dir),
            resume=bool(resume),
            tool_mode=normalized_tool_mode,
            prompt_style=normalized_prompt_style,
            role_name=normalized_role_name,
            prompt_profile=workflow_prompts.for_role(normalized_role_name),
            max_tool_calls=int(max_tool_calls),
            docker_image=str(docker_image),
            enable_memory=bool(enable_memory),
            enable_code_interpreter=bool(enable_code_interpreter),
            code_backend=code_backend,
            sandbox_fusion_base_url=sandbox_fusion_base_url,
            web_search_provider=web_search_provider,
            experiment_condition=str(experiment_condition).strip().lower(),
            require_analysis_evidence_per_round=bool(require_analysis_evidence_per_round),
            required_tool_names_per_round=normalized_required_tools,
            require_candidate_validation_per_round=bool(require_candidate_validation_per_round),
            require_optimizer_decision_per_round=bool(require_optimizer_decision_per_round),
            web_search_api_key_env=web_search_api_key_env,
            allow_fallback=bool(allow_fallback),
            require_visible_cot=bool(require_visible_cot),
            enable_bbo_skills=bool(enable_bbo_skills),
            skill_paths=normalized_skill_paths,
            enabled_tool_names=normalized_tool_names,
            optimizer_backend_allowlist=normalized_optimizer_backends,
            optimizer_max_calls_per_round=int(optimizer_max_calls_per_round),
        )
        self._engine = engine or create_general_agent_engine(normalized)
        if (
            normalized in {"nanobot", "codex", "claude_code"}
            and self._engine.name == normalized
            and normalized_tool_mode != "no_tool"
        ):
            raise ValueError(
                "Strict native black-box runs require `tool_mode='no_tool'`; "
                "workspace bridges expose optimizer-side runtime paths."
            )
        self._task_spec: TaskSpec | None = None
        self._description = TaskDescriptionBundle.empty(task_id="unknown")
        self._search_space: SearchSpace | None = None
        self._primary_name: str | None = None
        self._primary_direction = ObjectiveDirection.MINIMIZE
        self._seed = 0
        self._rng = random.Random(0)
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._history: list[TrialObservation] = []
        self._queue: list[AgentCandidateEntry] = []
        self._seen_config_ids: set[str] = set()
        self._best: Incumbent | None = None
        self._call_index = 0
        self._run_dir: Path | None = None
        self._workspace_dir: Path | None = None
        self._state_dir: Path | None = None
        self._memory_dir: Path | None = None
        self._work_copy: AgentWorkCopy | None = None
        self._manifest = None
        self._memory_store: BBOMemoryStore | None = None
        self._tool_registry: BBOToolRegistry | None = None
        self._artifacts: dict[str, str] = {}
        self._loaded_resume_snapshot: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.config.algorithm_name

    @property
    def artifact_paths(self) -> dict[str, str]:
        return dict(self._artifacts)

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        self._task_spec = task_spec
        self._search_space = task_spec.search_space
        self._primary_name = task_spec.primary_objective.name
        self._primary_direction = task_spec.primary_objective.direction
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        description = kwargs.get("task_description")
        self._description = (
            description if isinstance(description, TaskDescriptionBundle) else TaskDescriptionBundle.empty(task_id=task_spec.name)
        )

        self._run_dir = Path(kwargs.get("run_dir") or self.config.run_dir or Path.cwd()).resolve()
        self._workspace_dir = self._run_dir / "agent_workspace"
        self._state_dir = self._run_dir / "agent_state"
        self._memory_dir = self._run_dir / "agent_memory"
        reasoning_dir = self._run_dir / "reasoning_traces"
        log_dir = self._run_dir / "llm_logs"
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = load_BBO_manifest(task_spec)
        self._memory_store = (
            BBOMemoryStore(self._agent_memory_path, self._agent_memory_summary_path)
            if self.config.enable_memory
            else None
        )
        self._tool_registry = self._build_tool_registry()

        config_path = self._build_framework_config(log_dir)
        self._work_copy = AgentWorkCopy(
            state_dir=self._state_dir,
            config_path=config_path,
            project_root=self._workspace_dir,
            workspace_root=self._workspace_dir,
            extra={
                "nanobot_config": {
                    "env": self._agent_env(),
                    "black_box_required": True,
                    "tool_mode": self.config.tool_mode,
                },
                "codex_config": self._codex_config(),
                "claude_config": self._claude_config(),
                "openai_compatible_config": self._openai_compatible_config(),
                "log_dir": log_dir,
                "reasoning_dir": reasoning_dir,
                "reasoning_metadata_path": self._agent_reasoning_metadata_path,
            },
        )
        artifacts = {
            "agent_workspace": str(self._workspace_dir),
            "agent_final_candidate_json": str(
                self._workspace_dir / FINAL_CANDIDATE_FILENAME
            ),
            "agent_state_dir": str(self._state_dir),
            "agent_calls_jsonl": str(self._agent_calls_path),
            "agent_prompts_jsonl": str(self._agent_prompts_path),
            "llm_logs_dir": str(log_dir),
            "agent_llm_logs_dir": str(log_dir),
            "agent_state_json": str(self._agent_state_path),
            "agent_history_jsonl": str(self._workspace_dir / "history.jsonl"),
            "agent_optimization_trace_jsonl": str(self._agent_optimization_trace_path),
            "agent_space_json": str(self._workspace_dir / "space.json"),
            "agent_task_md": str(self._workspace_dir / "task.md"),
            "agent_manifest_json": str(self._workspace_dir / "manifest.json"),
            "agent_sources_jsonl": str(self._agent_sources_path),
            "agent_memory_jsonl": str(self._agent_memory_path),
            "agent_memory_summary_json": str(self._agent_memory_summary_path),
            "agent_reasoning_traces_dir": str(reasoning_dir),
            "agent_reasoning_metadata_jsonl": str(self._agent_reasoning_metadata_path),
        }
        if self._agent_tools_enabled():
            artifacts.update(
                {
                    "agent_tool_specs_json": str(self._agent_tool_specs_path),
                    "agent_workspace_tool_py": str(self._workspace_dir / "bbo_tool.py"),
                    "agent_workspace_bbo_tools_py": str(self._workspace_dir / "bbo_tools.py"),
                    "agent_workspace_tool_config_json": str(self._workspace_dir / "bbo_tool_config.json"),
                    "agent_workspace_tools_md": str(self._workspace_dir / "TOOLS.md"),
                    "agent_workspace_python_environment_md": str(self._workspace_dir / "python_environment.md"),
                    "agent_tool_calls_jsonl": str(self._agent_tool_calls_path),
                }
            )
            if self._optimizer_suggestion_enabled():
                artifacts.update(
                    {
                        "agent_workspace_gp_example_py": str(
                            self._workspace_dir / "examples" / "gp_expected_improvement.py"
                        ),
                        "agent_workspace_gp_entrypoint_py": str(
                            self._workspace_dir / "gp_expected_improvement.py"
                        ),
                    }
                )
        if self._agent_skills_enabled():
            artifacts["agent_workspace_skills_dir"] = str(self._workspace_dir / "skills")
        self._artifacts = artifacts
        log_paths = [
            self._agent_calls_path,
            self._agent_prompts_path,
            self._agent_sources_path,
            self._agent_optimization_trace_path,
            self._agent_reasoning_metadata_path,
        ]
        if self._agent_tools_enabled():
            log_paths.append(self._agent_tool_calls_path)
        for path in log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        if self._tool_registry is not None:
            dump_json(self._agent_tool_specs_path, {"tools": self._agent_tool_specs()})

        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        self._call_index = 0
        self._loaded_resume_snapshot = self._load_resume_snapshot()
        self._write_workspace_context()
        self._persist_state()

    def ask(self) -> TrialSuggestion:
        self._require_ready()
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(len(self._history), algorithm=self.name)
            suggestion.metadata.update(
                {
                    "agent_framework": self.config.framework,
                    "agent_source": "benchmark_initialization",
                }
            )
            self._seen_config_ids.add(stable_config_identity(suggestion.config))
            self._persist_state()
            return suggestion
        if len(self._history) < self.config.initial_random:
            return self._initial_random_suggestion()
        if not self._queue:
            self._fill_queue_from_agent()
        if not self._queue:
            raise RuntimeError(f"{self.name} could not produce any valid candidate configurations.")
        entry = self._queue.pop(0)
        metadata = {
            "agent_framework": self.config.framework,
            "agent_engine": self._engine.name,
            "agent_call_id": entry.call_id,
            "agent_candidate_index": entry.candidate_index,
            "agent_model": self.config.model,
            "agent_provider": self.config.provider,
            **entry.metadata,
        }
        self._persist_state()
        return TrialSuggestion(config=dict(entry.config), metadata=metadata)

    def tell(self, observation: TrialObservation) -> None:
        self._ingest_observation(observation)
        self._write_workspace_context()
        self._persist_state()

    def replay(self, history: list[TrialObservation]) -> None:
        self._require_ready()
        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        for observation in history:
            self._ingest_observation(observation, replay=True)
        self._restore_queue_from_snapshot()
        self._write_workspace_context()
        self._persist_state()

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    def _initial_random_suggestion(self) -> TrialSuggestion:
        search_space = self._require_search_space()
        for _ in range(100):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity not in self._seen_config_ids:
                self._seen_config_ids.add(identity)
                self._persist_state()
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "agent_framework": self.config.framework,
                        "agent_source": "initial_random",
                        **_search_action_metadata(
                            {"search_intent": "initialization", "change_summary": "framework initial random sample"},
                            source="initial_random",
                        ),
                    },
                )
        config = search_space.sample(self._rng)
        self._seen_config_ids.add(stable_config_identity(config))
        self._persist_state()
        return TrialSuggestion(
            config=config,
            metadata={
                "agent_framework": self.config.framework,
                "agent_source": "initial_random",
                **_search_action_metadata(
                    {"search_intent": "initialization", "change_summary": "framework initial random sample"},
                    source="initial_random",
                ),
            },
        )

    def _fill_queue_from_agent(self) -> None:
        search_space = self._require_search_space()
        last_error: str | None = None
        reasoning_requirement_failed = False
        round_call_ids: list[str] = []
        boundary_failed = False
        for attempt_index in range(self.config.max_retries + 1):
            self._write_workspace_context()
            call_id = f"agent_call_{self._call_index:05d}"
            self._call_index += 1
            round_call_ids.append(call_id)
            self._clear_workspace_candidate_file()
            prompt = self._build_agent_prompt(call_id=call_id, attempt_index=attempt_index, last_error=last_error)
            prompt = self.config.prompt_profile.compose(prompt, stage="round")
            append_jsonl(
                self._agent_prompts_path,
                {
                    "call_id": call_id,
                    "attempt_index": attempt_index,
                    "prompt": prompt,
                    "timestamp": time.time(),
                },
            )
            result = self._run_engine(prompt, call_id=call_id)
            call_record = {
                "call_id": call_id,
                "attempt_index": attempt_index,
                "framework": self.config.framework,
                "engine": self._engine.name,
                "status": result.status,
                "returncode": result.returncode,
                "error": result.error,
                "answer": result.answer,
                "llm_log": result.llm_log,
                "timestamp": time.time(),
            }
            if result.status != "success":
                append_jsonl(self._agent_calls_path, call_record)
                if result.returncode == -3:
                    boundary_failed = True
                last_error = result.error or result.answer or result.status
                continue
            reasoning_metadata = self._reasoning_metadata_for_call(call_id)
            if reasoning_metadata:
                call_record["reasoning"] = reasoning_metadata
            if self.config.require_visible_cot and not self._call_has_visible_reasoning(call_id):
                call_record["reasoning_error"] = "Required visible CoT was not captured for this agent call."
                append_jsonl(self._agent_calls_path, call_record)
                last_error = call_record["reasoning_error"]
                reasoning_requirement_failed = True
                continue
            parsed: list[ParsedAgentCandidate] | None = None
            workspace_error: str | None = None
            workspace_candidate = self._read_workspace_candidate_file(call_id)
            if workspace_candidate is not None:
                workspace_candidate_path, workspace_candidate_text = workspace_candidate
                try:
                    parsed = parse_agent_candidate_payload(workspace_candidate_text, search_space)
                except GeneralAgentValidationError as workspace_exc:
                    workspace_error = str(workspace_exc)
                else:
                    call_record["candidate_source"] = "workspace_candidate_file"
                    call_record["candidate_file"] = workspace_candidate_path
            if parsed is None:
                try:
                    parsed = parse_agent_candidate_payload(result.answer, search_space)
                except GeneralAgentValidationError as response_exc:
                    parsed = self._recover_successfully_validated_candidate(
                        round_call_ids, search_space
                    )
                    if parsed is not None:
                        call_record["candidate_source"] = "validated_tool_recovery"
                        call_record["agent_response_error"] = str(response_exc)
                        if workspace_error is not None:
                            call_record["workspace_candidate_error"] = workspace_error
                    else:
                        if workspace_error is None:
                            validation_error = str(response_exc)
                        else:
                            validation_error = (
                                f"workspace candidate file was invalid: {workspace_error}; "
                                f"agent response was also invalid: {response_exc}"
                            )
                        call_record["validation_error"] = validation_error
                        append_jsonl(self._agent_calls_path, call_record)
                        last_error = validation_error
                        continue
                else:
                    call_record["candidate_source"] = "agent_response"
                if workspace_error is not None:
                    call_record["workspace_candidate_error"] = workspace_error
            skill_read_error = self._declared_skill_read_error(call_id, parsed)
            condition_tool_error = self._condition_tool_usage_error(
                round_call_ids, parsed
            )
            if condition_tool_error:
                call_record["validation_error"] = condition_tool_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = condition_tool_error
                continue
            if skill_read_error:
                call_record["validation_error"] = skill_read_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = skill_read_error
                continue
            skill_tool_error = self._declared_skill_tool_usage_error(call_id, parsed)
            if skill_tool_error:
                call_record["validation_error"] = skill_tool_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = skill_tool_error
                continue

            accepted_actions = self._enqueue_candidates(call_id, parsed)
            accepted = len(accepted_actions)
            call_record["accepted_candidates"] = accepted
            if accepted_actions:
                call_record["accepted_search_actions"] = accepted_actions
            append_jsonl(self._agent_calls_path, call_record)
            self._persist_state()
            if accepted > 0:
                return
            last_error = "Agent returned only duplicate candidate configurations."

        if boundary_failed:
            raise RuntimeError(f"{self.name} refused to run without its black-box boundary: {last_error}")

        if reasoning_requirement_failed:
            raise RuntimeError(f"{self.name} failed the visible CoT requirement: {last_error}")

        if not self.config.allow_fallback:
            raise RuntimeError(f"{self.name} failed to produce a valid candidate and fallback is disabled: {last_error}")

        fallback = self._fallback_candidate(last_error or "agent_failed")
        if fallback is not None:
            self._queue.append(fallback)
            self._persist_state()
            append_jsonl(
                self._agent_calls_path,
                {
                    "call_id": fallback.call_id,
                    "framework": self.config.framework,
                    "engine": self._engine.name,
                    "status": "fallback",
                    "reason": last_error,
                    "accepted_candidates": 1,
                    "timestamp": time.time(),
                },
            )
            return
        raise RuntimeError(f"{self.name} failed to produce a valid candidate after retries: {last_error}")

    def _run_engine(self, prompt: str, *, call_id: str) -> AgentResult:
        self._require_ready()
        assert self._work_copy is not None
        tools = None
        tool_executor = None
        if self.config.tool_mode == "function_calling":
            registry = self._require_tool_registry()
            context = self._build_tool_context()
            tools = registry.get_tool_specs()

            async def _execute_tool(tool_name: str, arguments: dict[str, Any], tool_call_id: str | None = None) -> str:
                return await registry.execute_tool(
                    tool_name,
                    arguments,
                    context,
                    call_id=call_id,
                    tool_call_id=tool_call_id,
                )

            tool_executor = _execute_tool
        coro = self._engine.run_agent(
            "",
            prompt,
            self._work_copy,
            agent_id="bbo",
            timeout=self.config.timeout_seconds,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_calls=self.config.max_tool_calls,
            extra_env=self._agent_call_env(call_id),
        )
        return _run_coro_sync(coro)

    def _agent_call_env(self, call_id: str) -> dict[str, str]:
        env = {
            "BBO_AGENT_CALL_ID": call_id,
            "BBO_AGENT_MODEL_REQUESTED": self.config.model or "",
            "BBO_AGENT_PROVIDER": self.config.provider or "",
            "BBO_AGENT_REQUIRE_VISIBLE_COT": "1" if self.config.require_visible_cot else "0",
        }
        if self.config.framework == "nanobot":
            env["BBO_NANOBOT_REASONING_DIR"] = str(self._agent_reasoning_traces_dir)
            env["BBO_NANOBOT_REASONING_METADATA_PATH"] = str(self._agent_reasoning_metadata_path)
        return env

    def _reasoning_metadata_for_call(self, call_id: str) -> dict[str, Any] | None:
        path = self._agent_reasoning_metadata_path
        if not path.exists():
            return None
        latest: dict[str, Any] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("call_id") == call_id:
                latest = record
        return latest

    def _call_has_visible_reasoning(self, call_id: str) -> bool:
        record = self._reasoning_metadata_for_call(call_id)
        return bool(record and record.get("reasoning_visible"))

    def _build_tool_registry(self) -> BBOToolRegistry | None:
        if not self._agent_tools_enabled():
            return None
        tools = create_core_BBO_tools(enable_memory=self.config.enable_memory)
        if self.config.optimizer_backend_allowlist or (
            self.config.enabled_tool_names is not None
            and bool(set(self.config.enabled_tool_names) & OPTIMIZER_ACTION_TOOLS)
        ):
            tools.extend(create_optimizer_tools())
        if self.config.enable_code_interpreter:
            tools.append(CodeInterpreterTool())
        if self._web_tools_enabled():
            tools.extend([WebSearchTool(), FetchURLTool()])
        if self.config.enabled_tool_names is not None:
            by_name = {tool.name: tool for tool in tools}
            unknown = sorted(set(self.config.enabled_tool_names) - set(by_name))
            if unknown:
                raise ValueError(f"Unknown enabled BBO tools: {unknown!r}.")
            tools = [by_name[name] for name in self.config.enabled_tool_names]
        return BBOToolRegistry(tools, logger=BBOToolCallLogger(self._agent_tool_calls_path))

    def _agent_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specs narrowed to the optimizer backends exposed in this arm."""

        if self._tool_registry is None:
            return []
        specs = copy.deepcopy(self._tool_registry.get_tool_specs())
        allowed = list(self.config.optimizer_backend_allowlist)
        if not allowed:
            return specs
        for spec in specs:
            function = spec.get("function") if isinstance(spec, Mapping) else None
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters")
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            if not isinstance(properties, dict):
                continue
            if function.get("name") in {"optimizer_suggest", "optimizer_set_backend"}:
                backend = properties.get("backend")
                if isinstance(backend, dict):
                    backend["enum"] = allowed
            elif function.get("name") == "optimizer_portfolio_suggest":
                backends = properties.get("backends")
                items = backends.get("items") if isinstance(backends, dict) else None
                if isinstance(items, dict):
                    items["enum"] = allowed
        return specs

    def _web_tools_enabled(self) -> bool:
        provider = self.config.web_search_provider.strip().lower().replace("-", "_")
        return provider not in {"", "disabled", "none", "off", "false"}

    def _build_tool_context(self) -> BBOToolContext:
        self._require_ready()
        assert self._workspace_dir is not None
        assert self._state_dir is not None
        assert self._manifest is not None
        return BBOToolContext(
            task_spec=self._require_task_spec(),
            description=self._description,
            manifest=self._manifest,
            workspace_dir=self._workspace_dir,
            state_dir=self._state_dir,
            history=self._history,
            incumbent=self._best,
            memory_store=self._memory_store,
            code_backend=self._build_code_backend(),
            web_search_provider=self._build_web_search_provider(),
            source_logger=BBOWebSourceLogger(self._agent_sources_path),
            seed=self._seed,
            optimizer_backend_allowlist=self.config.optimizer_backend_allowlist,
        )

    def _build_code_backend(self) -> object:
        backend = self.config.code_backend.strip().lower().replace("-", "_")
        if backend == "mock":
            return MockBBOCodeBackend()
        if backend in {"disabled", "local_disabled", "none"}:
            return DisabledBBOCodeBackend()
        if backend in {"docker", "restricted_docker", "local_docker"}:
            if self._workspace_dir is None:
                raise RuntimeError("Agent workspace must exist before building Docker code backend.")
            return DockerBBOCodeBackend(
                workspace_dir=self._workspace_dir, image=self.config.docker_image
            )
        if backend == "sandboxfusion":
            base_url = self.config.sandbox_fusion_base_url or os.environ.get("SANDBOX_FUSION_BASE_URL")
            if not base_url:
                return DisabledBBOCodeBackend()
            return SandboxFusionBBOCodeBackend(base_url=base_url)
        raise ValueError(f"Unknown BBO code backend `{self.config.code_backend}`.")

    def _build_web_search_provider(self) -> object:
        return create_BBO_web_search_provider(
            self.config.web_search_provider,
            api_key_env=self.config.web_search_api_key_env,
        )

    def _require_tool_registry(self) -> BBOToolRegistry:
        if self._tool_registry is None:
            raise RuntimeError("BBO tool registry is not initialized.")
        return self._tool_registry

    def _enqueue_candidates(self, call_id: str, candidates: list[ParsedAgentCandidate]) -> list[dict[str, Any]]:
        accepted_actions: list[dict[str, Any]] = []
        for candidate in candidates:
            identity = stable_config_identity(candidate.config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            metadata = _search_action_metadata(
                candidate.metadata,
                call_id=call_id,
                candidate_index=candidate.candidate_index,
            )
            metadata = self._metadata_with_skill_audit(call_id=call_id, candidate=candidate, metadata=metadata)
            self._queue.append(
                AgentCandidateEntry(
                    config=dict(candidate.config),
                    call_id=call_id,
                    candidate_index=candidate.candidate_index,
                    metadata=metadata,
                )
            )
            action = metadata.get("search_action")
            if isinstance(action, dict):
                accepted_actions.append(agent_visible_payload(action))
            break
        return accepted_actions

    def _metadata_with_skill_audit(
        self,
        *,
        call_id: str,
        candidate: ParsedAgentCandidate,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return metadata
        assert self._run_dir is not None
        read_skills = _nanobot_read_skill_names_for_call(self._run_dir / "llm_logs", call_id)
        used_tools = _bbo_workspace_tool_names_for_call(self._agent_tool_calls_path, call_id)
        if not used_tools:
            used_tools = _bbo_tool_names_from_nanobot_session(self._run_dir / "llm_logs", call_id)
        audit = _build_skill_usage_audit(
            metadata=metadata,
            config=candidate.config,
            read_skills=read_skills,
            used_tools=used_tools,
            history=self._history,
            incumbent=self._best,
        )
        enriched = dict(metadata)
        action = dict(enriched.get("search_action") or {})
        action["skill_audit"] = audit
        enriched["search_action"] = action
        enriched["skill_audit"] = audit
        return enriched

    def _condition_tool_usage_error(
        self,
        call_id: str | Sequence[str],
        candidates: Sequence[ParsedAgentCandidate] | None = None,
    ) -> str | None:
        """Return a retryable error when this condition's round contract was not met."""

        if not (
            self.config.require_analysis_evidence_per_round
            or self.config.required_tool_names_per_round
            or self.config.require_candidate_validation_per_round
            or self.config.require_optimizer_decision_per_round
        ):
            return None
        used = self._successful_tool_names_for_call(call_id)
        missing_required = [
            name
            for name in self.config.required_tool_names_per_round
            if name not in used
        ]
        if missing_required:
            return (
                f"Condition {self.config.experiment_condition!r} requires successful "
                "per-round tool calls: " + ", ".join(missing_required) + "."
            )
        if (
            self.config.require_analysis_evidence_per_round
            and not (used & BBO_NUMERIC_EVIDENCE_TOOLS)
        ):
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful analysis/evidence tool call in every optimization round."
            )
        if (
            self.config.experiment_condition
            in {"t2_guided_analysis", "t3_agentic_search", "t4_soft_portfolio"}
            and not (used & BBO_REGION_EVIDENCE_TOOLS)
        ):
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful search-strategy analysis call in every optimization round: "
                "analyze_search_strategy."
            )
        if (
            candidates
            and self._best is not None
            and (used & BBO_REGION_EVIDENCE_TOOLS)
            and not (used & BBO_REGION_JOINT_SUPPORT_TOOLS)
        ):
            incumbent = agent_visible_config(self._best.config)
            for candidate in candidates:
                visible_candidate = agent_visible_config(candidate.config)
                changed = [
                    name
                    for name in visible_candidate
                    if visible_candidate.get(name) != incumbent.get(name)
                ]
                if len(changed) > MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES:
                    return (
                        "Marginal region evidence may directly change at most "
                        f"{MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES} parameters from the "
                        f"incumbent, but this candidate changes {len(changed)}: "
                        + ", ".join(changed)
                        + ". Keep context-only parameters at incumbent values, or obtain "
                        "successful joint support from analyze_parameter_interactions, "
                        "score_virtual_candidates, optimizer_suggest, "
                        "optimizer_portfolio_suggest, or optimizer_score."
                    )
        if (
            self.config.require_candidate_validation_per_round
            and not (used & {"validate_candidate", "validate_candidates"})
        ):
            return (
                f"Condition {self.config.experiment_condition!r} requires successful "
                "validation of the final formatted candidate in every optimization round."
            )
        decision_count = self._successful_tool_call_count_for_call(call_id, OPTIMIZER_DECISION_TOOLS)
        if self.config.require_optimizer_decision_per_round and decision_count == 0:
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful optimizer_suggest, optimizer_portfolio_suggest, or "
                "optimizer_score candidate-decision call "
                "in every optimization round."
            )
        if decision_count > self.config.optimizer_max_calls_per_round:
            return (
                "Optimizer candidate-decision call count exceeded the per-round cap: "
                f"{decision_count}/{self.config.optimizer_max_calls_per_round}."
            )
        return None

    def _recover_successfully_validated_candidate(
        self, call_id: str | Sequence[str], search_space: SearchSpace
    ) -> list[ParsedAgentCandidate] | None:
        """Recover only the exact config from this call's last successful validator."""

        path = self._agent_tool_calls_path
        if not path.exists():
            return None
        scoped_call_ids = _call_id_scope(call_id)
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if record_call_id in scoped_call_ids and record.get("success") is True:
                records.append(record)
        for record in reversed(records):
            name = str(record.get("tool_name", "")).strip()
            arguments = record.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            config: Mapping[str, Any] | None = None
            if name == "validate_candidate":
                candidate = arguments.get("candidate")
                if isinstance(candidate, Mapping):
                    nested = candidate.get("config")
                    config = nested if isinstance(nested, Mapping) else candidate
            elif name == "validate_candidates":
                candidates = arguments.get("candidates")
                if isinstance(candidates, list) and len(candidates) == 1:
                    candidate = candidates[0]
                    if isinstance(candidate, Mapping):
                        nested = candidate.get("config")
                        config = nested if isinstance(nested, Mapping) else candidate
            if config is None:
                continue
            payload = {"candidates": [{"config": dict(config)}]}
            try:
                return parse_agent_candidate_payload(
                    json.dumps(payload), search_space
                )
            except GeneralAgentValidationError:
                continue
        return None

    def _successful_tool_names_for_call(
        self, call_id: str | Sequence[str]
    ) -> set[str]:
        path = self._agent_tool_calls_path
        if not path.exists():
            return set()
        scoped_call_ids = _call_id_scope(call_id)
        names: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if record_call_id not in scoped_call_ids or record.get("success") is not True:
                continue
            name = str(record.get("tool_name", "")).strip()
            if name:
                names.add(name)
        return names

    def _successful_tool_call_count_for_call(
        self,
        call_id: str | Sequence[str],
        tool_names: set[str] | frozenset[str],
    ) -> int:
        path = self._agent_tool_calls_path
        if not path.exists():
            return 0
        scoped_call_ids = _call_id_scope(call_id)
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id in scoped_call_ids
                and record.get("success") is True
                and record.get("tool_name") in tool_names
            ):
                count += 1
        return count

    def _declared_skill_read_error(self, call_id: str, candidates: list[ParsedAgentCandidate]) -> str | None:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return None
        assert self._workspace_dir is not None
        assert self._run_dir is not None
        declared = _declared_agent_skill_names(candidates)
        if not declared:
            return None
        workspace_skills_dir = self._workspace_dir / "skills"
        required = sorted(skill for skill in declared if (workspace_skills_dir / skill / "SKILL.md").exists())
        if not required:
            return None
        read = _nanobot_read_skill_names_for_call(self._run_dir / "llm_logs", call_id)
        missing = [skill for skill in required if skill not in read]
        if not missing:
            return None
        missing_text = ", ".join(f"`{skill}`" for skill in missing)
        if len(missing) == 1:
            skill = missing[0]
            return (
                f"Agent set search_action.skill to `{skill}` but did not read "
                f"`skills/{skill}/SKILL.md` with the read_file tool in this same attempt. "
                "Follow the skill declaration rules in TOOLS.md, or set search_action.skill to JSON null."
            )
        return (
            f"Agent declared skills {missing_text} but did not read each corresponding "
            "`skills/<skill-name>/SKILL.md` file with the read_file tool in this same attempt. "
            "Follow the skill declaration rules in TOOLS.md, or set search_action.skill to JSON null."
        )

    def _declared_skill_tool_usage_error(self, call_id: str, candidates: list[ParsedAgentCandidate]) -> str | None:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return None
        assert self._workspace_dir is not None
        declared = _declared_agent_skill_names(candidates)
        if not declared:
            return None
        workspace_skills_dir = self._workspace_dir / "skills"
        checked = sorted(skill for skill in declared if (workspace_skills_dir / skill / "SKILL.md").exists())
        if not checked:
            return None
        used_tools = _bbo_workspace_tool_names_for_call(self._agent_tool_calls_path, call_id)
        for skill in checked:
            if skill in NON_PROPOSAL_BBO_SKILLS:
                return (
                    f"Agent set search_action.skill to `{skill}`, but `{skill}` is a memory maintenance "
                    "skill and must not be the primary skill for an evaluator-facing candidate. "
                    "Follow TOOLS.md if memory is useful, then either choose a proposal skill with its evidence tools "
                    "or set search_action.skill to JSON null."
                )
            required_groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill)
            if not required_groups:
                continue
            if skill == "initialize-search" and not self._history:
                required_groups = tuple(group for group in required_groups if group != ("measure_search_coverage",))
            missing = [group for group in required_groups if not any(tool in used_tools for tool in group)]
            if missing:
                missing_text = ", ".join(_format_tool_group(group) for group in missing)
                return (
                    f"Agent declared BBO skill `{skill}` but did not call the required BBO evidence tools "
                    f"in this same attempt. Missing: {missing_text}. Follow the tool protocol in TOOLS.md, "
                    "validate the final candidate, then return the raw JSON; otherwise set search_action.skill "
                    "to JSON null."
                )
        return None

    def _fallback_candidate(self, reason: str) -> AgentCandidateEntry | None:
        search_space = self._require_search_space()
        for index in range(500):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            return AgentCandidateEntry(
                config=config,
                call_id=f"fallback_random_{self._call_index:05d}",
                candidate_index=index,
                metadata={
                    "agent_source": "fallback_random",
                    "agent_fallback_reason": reason,
                    **_search_action_metadata(
                        {"search_intent": "exploration", "change_summary": "fallback random sample after agent failure"},
                        source="fallback_random",
                    ),
                },
            )
        return None

    def _clear_workspace_candidate_file(self) -> None:
        if self._workspace_dir is None:
            return
        path = self._workspace_dir / FINAL_CANDIDATE_FILENAME
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _read_workspace_candidate_file(self, call_id: str) -> tuple[str, str] | None:
        if self._workspace_dir is None:
            return None
        candidates = [
            self._workspace_dir / FINAL_CANDIDATE_FILENAME,
            self._workspace_dir / "scratch" / call_id / FINAL_CANDIDATE_FILENAME,
            self._workspace_dir / "scratch" / call_id / "candidate.json",
            self._workspace_dir / "scratch" / call_id / "candidates.json",
        ]
        for path in candidates:
            if path.is_symlink() or not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                relative_path = str(path.relative_to(self._workspace_dir))
                return relative_path, text
        return None

    def _ingest_observation(self, observation: TrialObservation, *, replay: bool = False) -> None:
        assert self._primary_name is not None
        self._history.append(observation)
        self._seen_config_ids.add(stable_config_identity(observation.suggestion.config))
        if observation.success and self._primary_name in observation.objectives:
            score = float(observation.objectives[self._primary_name])
            incumbent = Incumbent(
                config=dict(observation.suggestion.config),
                score=score,
                objectives=dict(observation.objectives),
                trial_id=observation.suggestion.trial_id,
                metadata={"algorithm": self.name, "agent_framework": self.config.framework},
            )
            if self._best is None:
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MINIMIZE and score < float(self._best.score):
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MAXIMIZE and score > float(self._best.score):
                self._best = incumbent
        if not replay and self._run_dir is not None:
            append_jsonl(
                self._agent_optimization_trace_path,
                {
                    "step": len(self._history),
                    "trial": _observation_summary(observation),
                    "best": None
                    if self._best is None
                    else {
                        "config": agent_visible_config(self._best.config),
                        "score": agent_visible_payload(self._best.score),
                        "objectives": agent_visible_payload(self._best.objectives),
                        "trial_id": self._best.trial_id,
                    },
                    "agent_framework": self.config.framework,
                    "agent_engine": self._engine.name,
                    "timestamp": time.time(),
                },
            )

    def _write_workspace_context(self) -> None:
        self._require_ready()
        assert self._workspace_dir is not None
        task_spec = self._require_task_spec()
        history = self._history[-self.config.history_limit :] if self.config.history_limit else []
        (self._workspace_dir / "task.md").write_text(self._render_task_markdown(), encoding="utf-8")
        dump_json(self._workspace_dir / "space.json", {"parameters": search_space_schema(task_spec.search_space)})
        if self._manifest is not None:
            dump_json(self._workspace_dir / "manifest.json", self._agent_workspace_manifest_payload())
        if self._tool_registry is not None:
            dump_json(self._workspace_dir / "tool_specs.json", {"tools": self._agent_tool_specs()})
        dump_json(
            self._workspace_dir / "objective.json",
            {
                "name": task_spec.primary_objective.name,
                "direction": task_spec.primary_objective.direction.value,
                "all_objectives": [
                    {"name": objective.name, "direction": objective.direction.value} for objective in task_spec.objectives
                ],
            },
        )
        dump_json(
            self._workspace_dir / "incumbent.json",
            {
                "config": None if self._best is None else agent_visible_config(self._best.config),
                "score": None if self._best is None else agent_visible_payload(self._best.score),
                "objectives": {} if self._best is None else agent_visible_payload(self._best.objectives),
                "trial_id": None if self._best is None else self._best.trial_id,
            },
        )
        self._write_history_jsonl(history)
        if self._agent_tools_enabled():
            self._write_workspace_tool_bridge()
            self._write_workspace_python_api()
            if self._optimizer_suggestion_enabled():
                self._write_workspace_gp_example()
            else:
                _remove_path(self._workspace_dir / "gp_expected_improvement.py")
                _remove_path(self._workspace_dir / "examples")
            (self._workspace_dir / "TOOLS.md").write_text(self._render_tools_markdown(), encoding="utf-8")
            (self._workspace_dir / "python_environment.md").write_text(self._render_python_environment(), encoding="utf-8")
        else:
            self._remove_workspace_tool_files()
        self._write_workspace_skills()
        if self._agent_tools_enabled():
            self._write_workspace_audit_script()
        instructions = self.config.prompt_profile.compose(self._render_instructions(), stage="protocol")
        (self._workspace_dir / "instructions.md").write_text(instructions, encoding="utf-8")

    def _agent_workspace_manifest_payload(self) -> dict[str, Any]:
        assert self._manifest is not None
        payload = self._manifest.to_dict()
        tool_policy = dict(payload.get("tool_policy") or {})
        tool_names: list[str] = []
        if self._tool_registry is not None:
            for spec in self._agent_tool_specs():
                function = spec.get("function") if isinstance(spec, Mapping) else None
                name = function.get("name") if isinstance(function, Mapping) else None
                if isinstance(name, str):
                    tool_names.append(name)
        if tool_names or not self._agent_tools_enabled():
            tool_policy["enabled_tools"] = tool_names
        enabled = set(tool_names)
        code_policy = dict(tool_policy.get("code_interpreter") or {})
        code_policy["enabled"] = "code_interpreter" in enabled
        tool_policy["code_interpreter"] = code_policy
        web_policy = dict(tool_policy.get("web_search") or {})
        web_policy["enabled"] = "web_search" in enabled or "fetch_url" in enabled
        tool_policy["web_search"] = web_policy
        payload["tool_policy"] = tool_policy
        payload["harness_policy"] = self._native_harness_policy()
        return payload

    def _write_history_jsonl(self, history: list[TrialObservation]) -> None:
        assert self._workspace_dir is not None
        path = self._workspace_dir / "history.jsonl"
        summarize = (
            _observation_summary
            if self._agent_tools_enabled()
            else _agent_history_summary
        )
        with path.open("w", encoding="utf-8") as handle:
            for observation in history:
                handle.write(
                    json.dumps(
                        to_jsonable(summarize(observation)),
                        sort_keys=True,
                    )
                    + "\n"
                )

    def _write_workspace_tool_bridge(self) -> None:
        assert self._workspace_dir is not None
        cli_source = Path(__file__).with_name("workspace_tool_cli.py").read_text(encoding="utf-8")
        script = "#!/usr/bin/env python3\n" + cli_source
        tool_path = self._workspace_dir / "bbo_tool.py"
        tool_path.write_text(script, encoding="utf-8")
        try:
            tool_path.chmod(0o755)
        except OSError:
            pass
        web_key_env = self.config.web_search_api_key_env
        if not web_key_env and self.config.web_search_provider.strip().lower().replace("-", "_") == "serpapi":
            web_key_env = "SERPAPI_API_KEY"
        web_search_api_key = os.environ.get(web_key_env or "")
        config_path = self._workspace_dir / "bbo_tool_config.json"
        dump_json(
            config_path,
            {
                "workspace_dir": str(self._workspace_dir),
                "state_dir": str(self._state_dir),
                "tool_calls_path": str(self._agent_tool_calls_path),
                "sources_path": str(self._agent_sources_path),
                "memory_path": str(self._agent_memory_path),
                "memory_summary_path": str(self._agent_memory_summary_path),
                "max_tool_calls": self.config.max_tool_calls,
                "enabled_tool_names": (
                    None
                    if self.config.enabled_tool_names is None
                    else list(self.config.enabled_tool_names)
                ),
                "optimizer_backend_allowlist": list(
                    self.config.optimizer_backend_allowlist
                ),
                "optimizer_max_calls_per_round": self.config.optimizer_max_calls_per_round,
                "optimizer_state_path": str(self._state_dir / "optimizer_tool_state.json"),
                "optimizer_python_executable": sys.executable,
                "optimizer_repository_root": str(Path(__file__).resolve().parents[3]),
                "experiment_condition": self.config.experiment_condition,
                "require_analysis_evidence_per_round": self.config.require_analysis_evidence_per_round,
                "required_tool_names_per_round": list(self.config.required_tool_names_per_round),
                "require_candidate_validation_per_round": self.config.require_candidate_validation_per_round,
                "require_optimizer_decision_per_round": self.config.require_optimizer_decision_per_round,
                "optimizer_agent_task_id": self._require_task_spec().name,
                "optimizer_max_evaluations": self._require_task_spec().max_evaluations,
                "optimizer_task_metadata": _optimizer_visible_task_metadata(
                    self._require_task_spec().metadata
                ),
                "seed": self._seed,
                "smiles_pool_path": os.environ.get("BBO_SMILES_POOL_PATH"),
                "code_backend": self.config.code_backend,
                "sandbox_fusion_base_url": self.config.sandbox_fusion_base_url or os.environ.get("SANDBOX_FUSION_BASE_URL"),
                "docker_image": self.config.docker_image,
                "web_search_provider": self.config.web_search_provider,
                "web_search_api_key_env": self.config.web_search_api_key_env,
                "web_search_api_key": web_search_api_key,
                "serpapi_endpoint": os.environ.get("SERPAPI_ENDPOINT"),
            },
        )
        try:
            config_path.chmod(0o600)
        except OSError:
            pass

    def _write_workspace_python_api(self) -> None:
        assert self._workspace_dir is not None
        api_source = Path(__file__).with_name("workspace_python_api.py").read_text(encoding="utf-8")
        api_path = self._workspace_dir / "bbo_tools.py"
        api_path.write_text(api_source, encoding="utf-8")

    def _write_workspace_gp_example(self) -> None:
        assert self._workspace_dir is not None
        examples_dir = self._workspace_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name("gp_expected_improvement_example.py").read_text(encoding="utf-8")
        path = examples_dir / "gp_expected_improvement.py"
        path.write_text(source, encoding="utf-8")
        try:
            path.chmod(0o755)
        except OSError:
            pass
        entrypoint = self._workspace_dir / "gp_expected_improvement.py"
        entrypoint.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                from __future__ import annotations

                import runpy
                from pathlib import Path


                if __name__ == "__main__":
                    workspace = Path(__file__).resolve().parent
                    runpy.run_path(str(workspace / "examples" / "gp_expected_improvement.py"), run_name="__main__")
                """
            ).lstrip(),
            encoding="utf-8",
        )
        try:
            entrypoint.chmod(0o755)
        except OSError:
            pass

    def _write_workspace_skills(self) -> None:
        assert self._workspace_dir is not None
        skill_sources = self._agent_skill_source_dirs()
        if not skill_sources:
            _remove_path(self._workspace_dir / "skills")
            return
        skills_dir = self._workspace_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for source in skill_sources:
            skill_name = _nanobot_skill_name(source)
            target = skills_dir / skill_name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            entries.append(_skill_index_entry(skill_name))
        dump_json(skills_dir / "index.json", {"skills": entries})

    def _agent_skill_source_dirs(self) -> list[Path]:
        sources: list[Path] = []
        if self.config.enable_bbo_skills:
            packaged_root = _packaged_bbo_nanobot_skills_dir()
            sources.extend(packaged_root / name for name in BBO_NANOBOT_SKILL_NAMES)
        for raw_path in self.config.skill_paths:
            sources.extend(_discover_nanobot_skill_dirs(raw_path))
        seen: set[str] = set()
        unique: list[Path] = []
        for source in sources:
            skill_name = _nanobot_skill_name(source)
            if skill_name in seen:
                continue
            seen.add(skill_name)
            unique.append(source)
        return unique

    def _agent_skills_enabled(self) -> bool:
        return bool(self.config.enable_bbo_skills or self.config.skill_paths)

    def _agent_tools_enabled(self) -> bool:
        return self.config.tool_mode != "no_tool"

    def _optimizer_suggestion_enabled(self) -> bool:
        if not self._agent_tools_enabled():
            return False
        if self.config.enabled_tool_names is None:
            return True
        return (
            "optimizer_suggest" in self.config.enabled_tool_names
            and "gp_ei" in self.config.optimizer_backend_allowlist
        )

    def _remove_workspace_tool_files(self) -> None:
        assert self._workspace_dir is not None
        for relative_path in (
            "TOOLS.md",
            "tool_specs.json",
            "bbo_tool.py",
            "bbo_tools.py",
            "bbo_tool_config.json",
            "bbo_workspace_audit.py",
            "bbo_workspace_audit_summary.json",
            "gp_expected_improvement.py",
            "python_environment.md",
            "examples",
        ):
            _remove_path(self._workspace_dir / relative_path)

    def _write_workspace_audit_script(self) -> None:
        assert self._workspace_dir is not None
        audit_path = self._workspace_dir / "bbo_workspace_audit.py"
        audit_path.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                from __future__ import annotations

                import json
                from pathlib import Path
                from typing import Any, Callable

                from bbo_tools import BBO


                def safe_call(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
                    try:
                        return {"ok": True, "result": fn()}
                    except Exception as exc:  # noqa: BLE001
                        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


                def main() -> int:
                    bbo = BBO()
                    summary: dict[str, Any] = {}
                    sample_holder: dict[str, Any] = {}

                    summary["task_context"] = safe_call("task_context", lambda: bbo.task_context())
                    summary["manifest"] = safe_call("manifest", bbo.manifest)
                    summary["search_space"] = safe_call("search_space", bbo.search_space)
                    summary["objective"] = safe_call("objective", bbo.objective)
                    summary["tool_specs"] = safe_call("tool_specs", bbo.tool_specs)
                    summary["history"] = safe_call("history", lambda: bbo.history(limit=20))
                    summary["incumbent"] = safe_call("incumbent", bbo.incumbent)
                    summary["history_overview"] = safe_call("history_overview", bbo.history_overview)
                    summary["objective_metrics"] = safe_call("objective_metrics", bbo.summarize_objective_metrics)
                    summary["coverage"] = safe_call("coverage", bbo.measure_search_coverage)
                    summary["recent_actions"] = safe_call("recent_actions", bbo.recent_search_actions)
                    summary["surrogate_check"] = safe_call("surrogate_check", bbo.fit_and_check_surrogate)

                    def sample_once() -> dict[str, Any]:
                        sample = bbo.sample(n=1, seed=0)
                        sample_holder["sample"] = sample
                        return sample

                    summary["sample"] = safe_call("sample", sample_once)
                    summary["analyze_history"] = safe_call("analyze_history", lambda: bbo.analyze_history(limit=100))
                    summary["memory_write"] = safe_call(
                        "memory_write",
                        lambda: bbo.memory_write(
                            kind="note",
                            content="BBO workspace audit completed.",
                            tags=["audit", "workspace"],
                            source_call_id="bbo_workspace_audit",
                        ),
                    )
                    summary["memory_read"] = safe_call("memory_read", lambda: bbo.memory_read(tags=["audit"], limit=5))
                    summary["code_interpreter"] = safe_call(
                        "code_interpreter",
                        lambda: bbo.code_interpreter("print('bbo workspace audit')", language="python"),
                    )
                    summary["web_search"] = safe_call(
                        "web_search",
                        lambda: bbo.web_search("black-box optimization placement benchmark", limit=1),
                    )
                    summary["fetch_url"] = safe_call(
                        "fetch_url",
                        lambda: bbo.fetch_url("https://example.com", max_chars=500),
                    )

                    def validate_sample() -> dict[str, Any]:
                        sample = sample_holder.get("sample")
                        if not isinstance(sample, dict) or not sample.get("candidates"):
                            sample = bbo.sample(n=1, seed=1)
                        return bbo.validate([sample["candidates"][0]])

                    summary["validate"] = safe_call("validate", validate_sample)
                    summary["validate_candidate"] = safe_call(
                        "validate_candidate",
                        lambda: bbo.validate_candidate(sample_holder.get("sample", {}).get("candidates", [{}])[0]),
                    )

                    path = Path("bbo_workspace_audit_summary.json")
                    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                    ok_count = sum(1 for item in summary.values() if isinstance(item, dict) and item.get("ok"))
                    print(json.dumps({"audit_summary_path": str(path), "ok_count": ok_count, "total": len(summary)}, sort_keys=True))
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).lstrip(),
            encoding="utf-8",
        )
        try:
            audit_path.chmod(0o755)
        except OSError:
            pass

    def _render_python_environment(self) -> str:
        return textwrap.dedent(
            """
            # BBO Python Environment

            The workspace Python API is available with:

            ```python
            from bbo_tools import BBO
            bbo = BBO()
            ```

            Prefer writing small Python scripts that use this API for task inspection,
            history analysis, candidate validation, memory, web research, and code-backed
            analysis. The underlying `bbo_tool.py` CLI remains available as a fallback.

            Recommended libraries for local analysis are the Python standard library,
            `numpy`, `scipy`, `pandas`, and `scikit-learn` when they are available.
            These libraries are optional aids for analysis; they are not required for
            every task or every candidate-generation strategy.

            When `code_interpreter` is configured with SandboxFusion, the SandboxFusion
            image should preinstall `numpy`, `scipy`, `scikit-learn`, `pandas`, and
            `joblib`. Heavy BO stacks such as `torch`, `gpytorch`, and `botorch` are
            optional and not required by the default BBO agent workflow.
            """
        ).strip()

    def _render_task_markdown(self) -> str:
        task_spec = self._require_task_spec()
        if self._description.rendered_context:
            return self._description.rendered_context
        return f"# {task_spec.name}\n\nNo structured task description was available."

    def _render_tools_markdown(self) -> str:
        skill_section = ""
        skill_file_hint = ""
        if self._agent_skills_enabled():
            skill_file_hint = ", and relevant `skills/<skill-name>/SKILL.md` files"
            skill_lines = [
                "## BBO Skills",
                "",
                "BBO skills are instruction documents under `skills/<skill-name>/SKILL.md`.",
                "They are not callable tools, Python functions, or `BBO()` methods.",
                "",
                "- Use at most one primary proposal skill for a candidate.",
                "- Read only the relevant `SKILL.md` files when they would improve the candidate.",
                "- If final `search_action.skill` is non-null, read the exact matching `SKILL.md` with `read_file` in the same attempt.",
                "- Do not rely only on a skill summary or description when declaring skill use.",
                "- If you declare a skill, call that skill's required BBO evidence tools from `skills/index.json` in the same attempt.",
                "- `repair-invalid-candidate` is only a secondary corrective skill after validation fails.",
                "- `distill-search-memory` is a memory maintenance skill and must not be the primary skill for an evaluator-facing candidate.",
                "",
                "Common built-in skill evidence requirements:",
            ]
            for skill_name, groups in SKILL_EVIDENCE_TOOL_GROUPS.items():
                formatted_groups = ", ".join(_format_tool_group(group) for group in groups)
                skill_lines.append(f"- `{skill_name}`: {formatted_groups}")
            skill_section = "\n\n" + "\n".join(skill_lines)
        enabled = (
            None
            if self.config.enabled_tool_names is None
            else set(self.config.enabled_tool_names)
        )
        optimizer_lines: list[str] = []
        if enabled is None or "optimizer_suggest" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_suggest(backend=..., q=1)` returns a candidate menu."
            )
        if enabled is None or "optimizer_score" in enabled:
            optimizer_lines.append("- `bbo.optimizer_score(configs)` scores virtual candidates.")
        if enabled is None or "optimizer_recommend_backends" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_recommend_backends(k=3)` gives an optional, explainable backend shortlist."
            )
        if enabled is None or "optimizer_portfolio_suggest" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_portfolio_suggest(backends=..., q_per_backend=1)` compares registered baselines on the same history and bounds."
            )
        controls = [
            name
            for name in (
                "optimizer_set_backend",
                "optimizer_set_bounds",
                "optimizer_set_acquisition",
                "optimizer_status",
                "optimizer_diagnostics",
                "optimizer_reset_policy",
            )
            if enabled is None or name in enabled
        ]
        if controls:
            optimizer_lines.append(
                "- Available optimizer controls: "
                + ", ".join(f"`bbo.{name}(...)`" for name in controls)
                + "."
            )
        optimizer_api_section = "\n".join(optimizer_lines)
        return textwrap.dedent(
            f"""
            # Tool Usage Notes

            This workspace exposes Nanobot native tools plus a local BBO Python API.

            ## Native Tools

            - Use `read_file` to inspect workspace files such as `task.md`, `space.json`,
              `objective.json`, `history.jsonl`, `incumbent.json`, `instructions.md`,
              and `TOOLS.md`{skill_file_hint}.
            - Use `exec` for short Python snippets that inspect history, call the BBO API,
              or validate candidate JSON.
            - Use relative workspace paths. Do not execute absolute paths when a relative
              path is available.

            ## BBO Python API

            Import the API inside Nanobot's `exec` tool:

            ```python
            from bbo_tools import BBO
            bbo = BBO()
            ```

            BBO is not a native Nanobot function-calling tool. Do not emit
            `<function=BBO>` and do not call invented methods such as
            `BBO().initialize_search()`.

            Useful methods include:

            - `bbo.task_context()`
            - `bbo.search_space()`
            - `bbo.objective()`
            - `bbo.history(mode="recent", limit=20)`
            - `bbo.incumbent()`
            - `bbo.history_overview()`
            - `bbo.summarize_objective_metrics()`
            - `bbo.compare_trials([...])`
            - `bbo.find_nearest_trials(target, k=5)`
            - `bbo.estimate_local_effects(reference, variables=None, local_radius=0.35)`
            - `bbo.measure_search_coverage()`
            - `bbo.sample(n=4, strategy="random")`
            - `bbo.fit_and_check_surrogate()`
            - `bbo.analyze_search_strategy()` for a landscape hypothesis, bias, conservative
              joint subspace, and downstream optimizer policy
            {optimizer_api_section}
            - `bbo.score_virtual_candidates(model_id, candidates)`
            - `bbo.validate_candidate(candidate)`
            - `bbo.validate([candidate])`
            - `bbo.recent_search_actions(limit=8)`
            - `bbo.memory_read()` and `bbo.memory_write(kind=..., content=...)`
            - `bbo.code_interpreter(code)` for restricted offline Python when enabled
            - `bbo.render_search_diagnostics()` for run-local JSON/SVG artifacts

            Use BBO tools selectively when they improve the decision, especially for
            exact trial comparison, objective/metric summaries, nearest trials, local
            effects, search coverage, promising/underexplored region ranking, surrogate
            validation, virtual candidate scoring, and final candidate validation.

            When a decision depends on precise history comparisons, variable
            differences, distances, local effects, coverage, surrogate quality, or
            candidate legality, use a BBO tool for evidence instead of estimating from
            memory.

            Validate the final rounded and formatted candidate with
            `bbo.validate_candidate(...)` or `bbo.validate(...)` when possible. If
            validation fails, repair or replace the candidate and validate again.

            Tool/API calls are append-only logged to `agent_tool_calls.jsonl`. Do not
            call the benchmark evaluator or any operation that consumes real evaluation
            budget.{skill_section}
            """
        ).strip()

    def _condition_tool_guidance(self) -> str:
        """Render explicit, auditable tool-use guidance for this experiment arm."""

        condition = self.config.experiment_condition
        if condition in {"", "default", "t0_bare"}:
            return ""
        guidance = [
            f"Experiment condition: {condition}.",
            "Use only tools listed in tool_specs.json / bbo_tools.py; tool availability is an experimental treatment.",
        ]
        if condition == "t1_analysis_available":
            guidance.append(
                "Search-strategy analysis is available through analyze_search_strategy. It returns a landscape hypothesis, bias, conservative joint subspace, and downstream optimizer policy; use it only when its evidence can improve the decision."
            )
        if self.config.require_analysis_evidence_per_round:
            guidance.append(
                "Before choosing the candidate, call at least one successful analysis/evidence tool and base the decision on its returned evidence."
            )
        if self.config.required_tool_names_per_round:
            required = ", ".join(self.config.required_tool_names_per_round)
            guidance.append(
                "Before choosing the candidate, successfully call every required "
                f"per-round tool: {required}. A failed or merely attempted call does not count."
            )
        if condition in {"t2_guided_analysis", "t3_agentic_search", "t4_soft_portfolio"}:
            guidance.extend(
                [
                    "Every optimization round must call analyze_search_strategy before changing search policy or bounds.",
                    "Treat its landscape and bias fields as hypotheses. Apply recommended_subspace.optimizer_bounds only when recommended_subspace.apply is true; otherwise preserve the original domain.",
                    "Use downstream_policy to choose a downstream optimizer/acquisition, but retain agent ownership and override it when task context or surrogate diagnostics provide stronger evidence.",
                    "State the exploit, explore, or balanced intent and the actionable parameters in search_action.change_summary.",
                    "Never concatenate every marginal row into one joint candidate. A candidate changing more than three parameters from the incumbent requires a successful analyze_parameter_interactions, score_virtual_candidates, optimizer_suggest, optimizer_portfolio_suggest, or optimizer_score call in the same round.",
                    "Validate the final joint candidate after applying these safeguards.",
                ]
            )
        if self.config.require_candidate_validation_per_round:
            guidance.append(
                "After final rounding/formatting, successfully call validate_candidate or validate_candidates on the exact final config."
            )
        if self.config.require_optimizer_decision_per_round:
            allowed = list(self.config.optimizer_backend_allowlist)
            if len(allowed) == 1:
                optimizer_policy = (
                    f"This condition exposes one fixed optimizer backend: {allowed[0]}. "
                    "Do not attempt to select or switch to any other backend. "
                    "You may narrow/reset numeric bounds"
                )
                if allowed[0] == "gp_ei":
                    optimizer_policy += " and select EI, LogEI, or UCB acquisition settings."
                else:
                    optimizer_policy += "."
            else:
                optimizer_policy = (
                    "You control the single-objective search loop and may choose only "
                    f"among these enabled backends: {', '.join(allowed)}. You may persist "
                    "an enabled backend, narrow/reset numeric bounds, and select EI, "
                    "LogEI, or UCB for gp_ei. optimizer_recommend_backends is "
                    "advisory and never switches automatically. "
                    "optimizer_portfolio_suggest compares registered baselines "
                    "on the same history and bounds; you still decide."
                )
            guidance.extend(
                [
                    "Every optimization round must include at least one successful optimizer_suggest, optimizer_portfolio_suggest, or optimizer_score call; "
                    f"at most {self.config.optimizer_max_calls_per_round} such candidate-decision calls are allowed.",
                    optimizer_policy,
                    "All optimizer candidates come from the same registered implementations and benchmark policy as standalone baselines. Menus never evaluate points. Inspect them and submit exactly one final candidate.",
                    "Record search_action.optimizer with relationship=adopt, refine, override, or direct_scored; include the selected backend and candidate identity when applicable.",
                    "Only outer-runner observations update optimizer history. Never invent or tell virtual objective values.",
                ]
            )
        return "\n".join(f"- {item}" for item in guidance)

    def _render_instructions(self) -> str:
        if not self._agent_tools_enabled():
            return self._render_no_tool_instructions()

        task_spec = self._require_task_spec()
        compact_xy_hint = self._compact_xy_output_hint()
        skills_file_line = ""
        skills_workflow_line = ""
        skill_read_line = ""
        skill_strategy_line = "Choose a candidate-generation strategy appropriate for the current task and evidence."
        tools_md_line = "- TOOLS.md: native tool, BBO Python API, and validation guidance."
        search_action_shape = self._candidate_payload_example()
        null_requirement = "- Use JSON null for absent hypothesis values, not the string \"null\"."
        if self._agent_skills_enabled():
            tools_md_line = "- TOOLS.md: native tool, BBO Python API, validation, and skill-use guidance."
            skills_file_line = (
                "\n            - skills/: optional Nanobot BBO skill reference library. "
                "Use it according to `TOOLS.md`."
            )
            skill_read_line = (
                "\n            - Read `TOOLS.md` before using workspace tools, the BBO Python API, or BBO\n"
                "              skills."
            )
            skill_strategy_line = (
                "Choose a candidate-generation strategy appropriate for the current task\n"
                "              and evidence. You may propose directly without any skill when no\n"
                "              specialized skill trigger is clearly satisfied."
            )
            skills_workflow_line = (
                "\n            - Skills are optional references, not mandatory steps. "
                "Decide whether any skill applies from the current task and evidence; "
                "do not read or follow every skill by default. Follow `TOOLS.md` for "
                "skill-read, evidence, validation, and declaration rules."
            )
            null_requirement = "- Use JSON null for absent skill or hypothesis values, not the string \"null\"."
        else:
            skill_read_line = "\n            - Read `TOOLS.md` before using workspace tools or the BBO Python API."
        return textwrap.dedent(
            f"""
            # Agentic BBO Candidate Protocol

            You are proposing configurations for a black-box optimization benchmark.
            Do not call the benchmark evaluator yourself and do not modify benchmark
            result files.

            Files in this workspace:
            - task.md: task background, goal, constraints, and prior knowledge.
            - manifest.json: agent benchmark construction, tool policy, and provenance.
            - space.json: exact parameter schema. Every candidate must include every parameter exactly once.
            - objective.json: primary objective name and optimization direction.
            - history.jsonl: recent evaluated trials.
            - incumbent.json: current best known configuration, if any.
            {tools_md_line}
            - tool_specs.json: available BBO function-calling tools when the backend supports tools.
            - bbo_tools.py: preferred Python API for BBO tools when using shell/file tools.
            - bbo_workspace_audit.py: optional observability script that exercises workspace BBO APIs.
            - examples/: optional candidate-generation or analysis examples. No example is mandatory.
            - python_environment.md: Python and sandbox library guidance.
            - bbo_tool.py: lower-level CLI bridge for BBO tools; use only as a fallback.{skills_file_line}
            - {FINAL_CANDIDATE_FILENAME}: authoritative final candidate handoff. The harness clears it before each attempt.

            Task: {task_spec.name}
            Primary objective: {task_spec.primary_objective.name}
            Direction: {task_spec.primary_objective.direction.value}

            Workspace workflow:
            - Use relative paths from the workspace. Do not execute absolute paths.
            {skill_read_line}
            - Read evaluated history, the incumbent, recent search actions, and the
              remaining budget before choosing this round's one most valuable search action.
            - Choose one search intent for this round: initialization, exploitation,
              directional_extrapolation, hypothesis_test, interaction_test,
              recombination, exploration, stagnation_recovery, surrogate_proposal,
              or repair.
            - {skill_strategy_line}
            - Do not treat any single method or example script as mandatory. Use tools
              to improve judgment, not to follow a fixed recipe.
            - You may create temporary scratch files or short analysis scripts inside
              the workspace when useful, using new relative paths such as
              `candidate.json`, `analysis.py`, or `scratch/candidates.json`.
              Do not overwrite task definitions, history, results, logs, or framework
              state files.{skills_workflow_line}
            - Tool/API calls are append-only logged to agent_tool_calls.jsonl.

            Final candidate handoff:
            - Write the exact top-level payload below to `{FINAL_CANDIDATE_FILENAME}` in the workspace root.
            - Verify JSON syntax with `python -m json.tool {FINAL_CANDIDATE_FILENAME}`.
            - For tool-enabled conditions, validate the exact rounded candidate with
              `validate_candidate` or `validate_candidates`, then immediately write the
              file before any further explanation, analysis, or tool call.
            - Do not announce that you are about to write the file; write it first.
            - Return the same raw JSON in chat for compatibility. The harness prefers
              `{FINAL_CANDIDATE_FILENAME}` and uses chat only as a fallback.

            Exact payload shape:
            {search_action_shape}

            Requirements:
            - Return exactly one candidate configuration. Each real optimization round
              submits one and only one new candidate to the evaluator.
            - Use available tools according to `TOOLS.md` when they improve the
              candidate decision.
            - Validate proposed candidates according to `TOOLS.md` before final output
              when possible.
            - If any script, command, or tool fails, recover with another reasonable
              strategy; never return an error message as the final answer.
            - Temporary candidate JSON files are allowed for checking, but only
              `{FINAL_CANDIDATE_FILENAME}` is the authoritative file handoff.
            - The harness independently rechecks schema, bounds, types, duplicates,
              and condition-specific tool evidence before accepting the candidate.
            - Do not use shell redirection such as `2>/dev/null`; rerun Python scripts only with relative commands.
            - Do not include markdown fences, comments, prose, or partial configurations.
            - Do not force intermediate analysis into JSON; only the final candidate
              payload and persisted action metadata need machine-readable structure.
            - Float and integer values must stay within their declared bounds.
            - Numeric values in final candidate JSON should use at most 4 decimal places.
            - Categorical values must be one of the declared choices.
            {null_requirement}
            {compact_xy_hint}
            """
        ).strip()

    def _build_agent_prompt(self, *, call_id: str, attempt_index: int, last_error: str | None = None) -> str:
        if not self._agent_tools_enabled():
            return self._build_no_tool_agent_prompt(
                call_id=call_id,
                attempt_index=attempt_index,
                last_error=last_error,
            )

        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        retry_feedback = _retry_feedback_block(last_error)
        retry_feedback_section = "" if not retry_feedback else "\n\n" + textwrap.indent(retry_feedback, "            ")
        compact_xy_hint = self._compact_xy_output_hint()
        condition_guidance = self._condition_tool_guidance()
        condition_guidance_section = "" if not condition_guidance else "\n\n" + condition_guidance
        tool_prompt_line = (
            "Read `TOOLS.md` before using workspace tools, BBO Python API helpers, or\n"
            "            BBO skills. Skill use is optional: use a skill only when it clearly helps\n"
            "            this proposal, otherwise set `search_action.skill` to JSON `null`."
            if self._agent_skills_enabled()
            else "Read `TOOLS.md` before using workspace tools or BBO Python API helpers."
        )
        tools_md_scope = (
            "tool protocol, BBO Python API, validation, and skill-use rules"
            if self._agent_skills_enabled()
            else "tool protocol, BBO Python API, and validation rules"
        )
        search_action_fields = self._search_action_prompt_fields()
        candidate_payload_example = self._candidate_payload_example(indent=2)
        return textwrap.dedent(
            f"""
            You are an optimization agent for task `{task_spec.name}`.

            Workspace: `.`
            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            Produce exactly one new candidate configuration.

            Use the workspace as the source of truth:

            * `task.md`: task description, constraints, and prior knowledge
            * `space.json`: parameter names, types, bounds, choices, and precision
            * `objective.json`: objective name and direction
            * `history.jsonl` and `incumbent.json`: evaluated history and current best
            * `instructions.md`: final output and search-action metadata contract
            * `TOOLS.md`: {tools_md_scope}

            Do not call the benchmark evaluator or any operation that consumes real
            evaluation budget.

            {tool_prompt_line}{condition_guidance_section}

            Requirements:

            * Return exactly one candidate configuration.
            * Include every active required parameter exactly once.
            * Preserve the parameter types declared in `space.json`.
            * Respect all bounds, choices, precision rules, conditional rules, and
              constraints.
            * Do not invent objective values or trial IDs.
            * Do not return an exact duplicate of an evaluated configuration.
            * Near-duplicate candidates are allowed only when justified by local refinement
              or a controlled experiment.
            * Numeric precision should follow `space.json`. If no precision is declared,
              use at most 4 decimal places without changing the intended scale.
            * Validate the final formatted candidate according to `TOOLS.md` when
              possible. Validation must occur after rounding, formatting, and any
              repair.
            * If validation fails, repair or replace the candidate and validate again.
            * If validation tooling fails, manually check the candidate against
              `space.json` and the evaluated history.

            Describe the search action using:

            {search_action_fields}

            Only include trial IDs that exist in the workspace history.

            Do not modify protected files:

            * `task.md`
            * `space.json`
            * `objective.json`
            * `history.jsonl`
            * `incumbent.json`
            * `TOOLS.md`
            * `trials.jsonl`
            * `agent_*.jsonl`
            * files under `agent_state/`
            * files under `llm_logs/`
            * files under `reasoning_traces/`

            You may create temporary scratch files or short analysis scripts under a new
            call-specific path such as:

            `scratch/{call_id}/`

            Before finishing, write the exact final payload to
            `{FINAL_CANDIDATE_FILENAME}` in the workspace root and verify it with:

            `python -m json.tool {FINAL_CANDIDATE_FILENAME}`

            Validate the exact rounded configuration with `validate_candidate` or
            `validate_candidates` before writing it. Return the same JSON in chat; the
            harness treats the file as authoritative and chat as a compatibility fallback.

            If a tool or command fails, recover using the workspace files and still return
            one valid candidate.

            Current best primary objective: `{best_score}`
            Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

            Return only valid raw JSON with exactly this shape:

            {candidate_payload_example}

            Replace the example config with the exact active parameters from `space.json`.

            Use native JSON types:

            * numbers as numbers
            * integers as integers
            * booleans as booleans
            * categorical values as strings
            * absent hypotheses as JSON `null`, not the string `"null"`

            {compact_xy_hint}

            Return no Markdown fences, comments, headings, explanations, or additional
            prose.
            """
        ).strip()

    def _render_no_tool_instructions(self) -> str:
        task_spec = self._require_task_spec()
        compact_xy_hint = self._compact_xy_output_hint()
        if self.config.framework == "nanobot":
            native_tool_lines = (
                "- Use `read_file` to inspect the workspace files before proposing.\n"
                "            - Use `exec` only for short local calculations or scratch scripts over\n"
                "              workspace data. Use relative paths from the workspace."
            )
        else:
            native_tool_lines = (
                "- Use the harness's native file-reading tools to inspect the workspace files before proposing.\n"
                "            - Native shell tools may be used for short local calculations or scratch scripts over\n"
                "              workspace data. Use relative paths from the workspace."
            )
        return textwrap.dedent(
            f"""
            # Agentic BBO Candidate Protocol

            You are proposing configurations for a black-box optimization benchmark.
            Do not call the benchmark evaluator yourself and do not modify benchmark
            result files.

            Files in this workspace:
            - task.md: task background, goal, constraints, and prior knowledge.
            - manifest.json: agent benchmark construction and provenance.
            - space.json: exact parameter schema. Every candidate must include every parameter exactly once.
            - objective.json: primary objective name and optimization direction.
            - history.jsonl: recent evaluated trials.
            - incumbent.json: current best known configuration, if any.
            - {FINAL_CANDIDATE_FILENAME}: authoritative final candidate handoff. The harness clears it before each attempt.

            Task: {task_spec.name}
            Primary objective: {task_spec.primary_objective.name}
            Direction: {task_spec.primary_objective.direction.value}

            Workspace workflow:
            {native_tool_lines}
            - Do not call the benchmark evaluator or any operation that consumes real
              evaluation budget.
            - You may create temporary scratch files or short analysis scripts inside
              the workspace when useful, using new relative paths such as
              `candidate.json`, `analysis.py`, or `scratch/candidates.json`.
              Do not overwrite task definitions, history, results, logs, or framework
              state files.

            Final candidate handoff:
            - Write the exact top-level payload below to `{FINAL_CANDIDATE_FILENAME}` in the workspace root.
            - Verify JSON syntax with `python -m json.tool {FINAL_CANDIDATE_FILENAME}`.
            - Manually check the exact rounded candidate against `space.json` and
              `history.jsonl` before writing the file.
            - Return the same raw JSON in chat for compatibility. The harness prefers
              `{FINAL_CANDIDATE_FILENAME}` and uses chat only as a fallback.

            Exact payload shape:
            {self._candidate_payload_example()}

            Requirements:
            - Return exactly one candidate configuration. Each real optimization round
              submits one and only one new candidate to the evaluator.
            - Include every active required parameter exactly once.
            - Respect all bounds, choices, precision rules, conditional rules, and constraints.
            - Do not invent objective values or trial IDs.
            - Do not return an exact duplicate of an evaluated configuration.
            - Numeric values in final candidate JSON should use at most 4 decimal places.
            - Categorical values must be one of the declared choices.
            - Use JSON null for absent hypothesis values, not the string "null".
            - The harness independently rechecks schema, bounds, types, and duplicates.
            {compact_xy_hint}
            """
        ).strip()

    def _build_no_tool_agent_prompt(self, *, call_id: str, attempt_index: int, last_error: str | None = None) -> str:
        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        retry_feedback = _retry_feedback_block(last_error, mention_tool_calls=False)
        retry_feedback_section = "" if not retry_feedback else "\n\n" + textwrap.indent(retry_feedback, "            ")
        compact_xy_hint = self._compact_xy_output_hint()
        native_tool_guidance = (
            "Use `read_file` to inspect these files. You may use `exec` for short\n"
            "                local calculations over workspace data or temporary scratch scripts."
            if self.config.framework == "nanobot"
            else "Use the harness's native file-reading tools to inspect these files. You may use\n"
            "                native shell tools for short local calculations or temporary scratch scripts."
        )
        if self.config.framework in {"nanobot", "codex", "claude_code"}:
            return textwrap.dedent(
                f"""
                You are an optimization agent for task `{task_spec.name}`.

                Workspace: `.`
                Call id: `{call_id}`
                Attempt: `{attempt_index}`

                Produce exactly one new candidate configuration.

                Use the workspace as the source of truth:

                * `task.md`: task description, constraints, and prior knowledge
                * `space.json`: parameter names, types, bounds, choices, and precision
                * `objective.json`: objective name and direction
                * `history.jsonl` and `incumbent.json`: evaluated history and current best
                * `instructions.md`: final output and search-action metadata contract

                {native_tool_guidance}
                Use relative paths from the workspace.

                Do not call the benchmark evaluator or any operation that consumes real
                evaluation budget.

                Requirements:

                * Return exactly one candidate configuration.
                * Include every active required parameter exactly once.
                * Preserve the parameter types declared in `space.json`.
                * Respect all bounds, choices, precision rules, conditional rules, and
                  constraints.
                * Do not invent objective values or trial IDs.
                * Do not return an exact duplicate of an evaluated configuration.
                * Near-duplicate candidates are allowed only when justified by local refinement
                  or a controlled experiment.
                * Numeric precision should follow `space.json`. If no precision is declared,
                  use at most 4 decimal places without changing the intended scale.
                * Manually check the final formatted candidate against `space.json` and
                  the evaluated history.

                Describe the search action using:

                {self._search_action_prompt_fields()}

                Only include trial IDs that exist in the workspace history.

                Do not modify protected files:

                * `task.md`
                * `space.json`
                * `objective.json`
                * `history.jsonl`
                * `incumbent.json`
                * `trials.jsonl`
                * `agent_*.jsonl`
                * files under `agent_state/`
                * files under `llm_logs/`
                * files under `reasoning_traces/`

                You may create temporary scratch files or short analysis scripts under a
                new call-specific path such as:

                `scratch/{call_id}/`

                Before finishing, write the exact final payload to
                `{FINAL_CANDIDATE_FILENAME}` in the workspace root and verify it with:

                `python -m json.tool {FINAL_CANDIDATE_FILENAME}`

                Manually check the exact rounded configuration against `space.json` and
                `history.jsonl` before writing it. Return the same JSON in chat; the
                harness treats the file as authoritative and chat as a compatibility fallback.

                If a command fails, recover using the workspace files and still return
                one valid candidate.

                Current best primary objective: `{best_score}`
                Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

                Return only valid raw JSON with exactly this shape:

                {self._candidate_payload_example(indent=2)}

                Replace the example config with the exact active parameters from
                `space.json`.

                Use native JSON types:

                * numbers as numbers
                * integers as integers
                * booleans as booleans
                * categorical values as strings
                * absent hypotheses as JSON `null`, not the string `"null"`

                {compact_xy_hint}

                Return no Markdown fences, comments, headings, explanations, or
                additional prose.
                """
            ).strip()

        context = self._no_tool_prompt_context()
        return textwrap.dedent(
            f"""
            You are an optimization agent for task `{task_spec.name}`.

            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            Produce exactly one new candidate configuration.

            Task description:
            {context["task_markdown"]}

            Search space JSON:
            {context["space_json"]}

            Objective JSON:
            {context["objective_json"]}

            Recent evaluated history JSONL:
            {context["history_jsonl"]}

            Incumbent JSON:
            {context["incumbent_json"]}

            Do not call the benchmark evaluator or any operation that consumes real
            evaluation budget.

            Requirements:

            * Return exactly one candidate configuration.
            * Include every active required parameter exactly once.
            * Preserve the parameter types declared in the search space JSON.
            * Respect all bounds, choices, precision rules, conditional rules, and
              constraints.
            * Do not invent objective values or trial IDs.
            * Do not return an exact duplicate of an evaluated configuration.
            * Near-duplicate candidates are allowed only when justified by local refinement
              or a controlled experiment.
            * Numeric precision should follow the search space JSON. If no precision is
              declared, use at most 4 decimal places without changing the intended scale.
            * Manually check the final formatted candidate against the search space and
              evaluated history.

            Describe the search action using:

            {self._search_action_prompt_fields()}

            Only include trial IDs that exist in the evaluated history.

            Current best primary objective: `{best_score}`
            Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

            Return only valid raw JSON with exactly this shape:

            {self._candidate_payload_example(indent=2)}

            Replace the example config with the exact active parameters from the search
            space JSON.

            Use native JSON types:

            * numbers as numbers
            * integers as integers
            * booleans as booleans
            * categorical values as strings
            * absent hypotheses as JSON `null`, not the string `"null"`

            {compact_xy_hint}

            Return no Markdown fences, comments, headings, explanations, or additional
            prose.
            """
        ).strip()

    def _no_tool_prompt_context(self) -> dict[str, str]:
        task_spec = self._require_task_spec()
        history = self._history[-self.config.history_limit :] if self.config.history_limit else []
        objective = {
            "name": task_spec.primary_objective.name,
            "direction": task_spec.primary_objective.direction.value,
            "all_objectives": [
                {"name": objective.name, "direction": objective.direction.value} for objective in task_spec.objectives
            ],
        }
        incumbent = {
            "config": None if self._best is None else agent_visible_config(self._best.config),
            "score": None if self._best is None else agent_visible_payload(self._best.score),
            "objectives": {} if self._best is None else agent_visible_payload(self._best.objectives),
            "trial_id": None if self._best is None else self._best.trial_id,
        }
        summarize = (
            _observation_summary
            if self._agent_tools_enabled()
            else _agent_history_summary
        )
        history_jsonl = "\n".join(
            json.dumps(
                to_jsonable(summarize(item)),
                sort_keys=True,
            )
            for item in history
        )
        return {
            "task_markdown": self._render_task_markdown(),
            "space_json": json.dumps({"parameters": search_space_schema(task_spec.search_space)}, indent=2, sort_keys=True),
            "objective_json": json.dumps(objective, indent=2, sort_keys=True),
            "history_jsonl": history_jsonl or "(empty)",
            "incumbent_json": json.dumps(incumbent, indent=2, sort_keys=True),
        }

    def _search_action_prompt_fields(self) -> str:
        lines = []
        if self._agent_skills_enabled():
            lines.append("* `skill`: the primary skill used, or JSON `null` if no skill was used or available")
        lines.extend(
            [
                "* `parent_trials`: trials from which the candidate was directly modified",
                "* `reference_trials`: trials used as evidence for the decision",
                "* `hypothesis`: the specific hypothesis tested by this candidate, or JSON `null`",
                "* `change_summary`: a concise description of how the candidate was produced",
            ]
        )
        if self.config.require_optimizer_decision_per_round:
            lines.append(
                "* `optimizer`: an object with `relationship`, `backend`, "
                "`candidate_identity`, and `considered_backends`; use JSON "
                "`null` for fields that do not apply"
            )
        return "\n".join(lines)

    def _candidate_payload_example(self, *, indent: int | None = None) -> str:
        search_action: dict[str, Any] = {
            "parent_trials": [],
            "reference_trials": [],
            "hypothesis": None,
            "change_summary": "Short description of how the candidate was produced.",
        }
        if self._agent_skills_enabled():
            search_action = {"skill": None, **search_action}
        if self.config.require_optimizer_decision_per_round:
            search_action["optimizer"] = {
                "relationship": "adopt",
                "backend": "gp_ei",
                "candidate_identity": "backend candidate identity or null",
                "considered_backends": ["gp_ei"],
            }
        payload = {
            "candidates": [
                {
                    "config": {"param_name": 0.0},
                    "rationale": "Short evidence-based reason for proposing this candidate.",
                    "search_action": search_action,
                }
            ]
        }
        return json.dumps(payload, indent=indent, sort_keys=False)

    def _compact_xy_output_hint(self) -> str:
        search_space = self._search_space
        if search_space is None:
            return ""
        n_pairs = _paired_xy_parameter_count(search_space)
        if n_pairs <= 0:
            return ""
        return (
            "For this paired-coordinate task, you may use compact coordinate arrays in the final config: "
            f"`{{\"x\": [<exactly {n_pairs} numbers>], \"y\": [<exactly {n_pairs} numbers>]}}`. "
            "The framework expands them to `x_0...` and `y_0...`. This compact form is preferred for BBOPlace."
        )

    def _build_framework_config(self, log_dir: Path) -> Path | None:
        assert self._state_dir is not None
        if self.config.framework == "nanobot":
            config_path = self._state_dir / "config.json"
            provider = self.config.provider
            provider_key = _nanobot_provider_key(provider or "custom")
            model = self.config.model
            cfg: dict[str, Any] = {
                "agents": {
                    "defaults": {
                        "workspace": "/workspace",
                        "provider": provider_key if provider else "auto",
                        "disabled_skills": [] if self._agent_skills_enabled() else list(NANOBOT_BUILTIN_SKILL_NAMES),
                    }
                },
                "providers": {},
                "channels": {
                    "send_progress": False,
                    "send_tool_hints": False,
                },
                "tools": {
                    "restrict_to_workspace": True,
                },
            }
            if not self._agent_tools_enabled():
                cfg["tools"].update(
                    {
                        "exec": {"enable": True},
                        "web": {"enable": False},
                        "my": {"enable": False},
                        "mcp_servers": {},
                    }
                )
            if model:
                cfg["agents"]["defaults"]["model"] = model
            api_key_ref = None
            if self.config.api_key_env:
                api_key_ref = f"${{{self.config.api_key_env}}}"
            elif self._api_key():
                api_key_ref = self._api_key()
            if provider or self.config.api_base or api_key_ref:
                entry: dict[str, str] = {}
                if api_key_ref:
                    entry["api_key"] = api_key_ref
                if self.config.api_base:
                    entry["api_base"] = self.config.api_base
                cfg["providers"][provider_key] = entry
            config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
            return config_path
        if self.config.framework == "codex":
            config_path = self._state_dir / "config.toml"
            provider_name = "bbo_sglang"
            model = self.config.model or "qwen3.5-9b"
            base_url = self.config.api_base or "http://127.0.0.1:18300/v1"
            lines = [
                f"model = {json.dumps(model)}",
                f"model_provider = {json.dumps(provider_name)}",
                "model_context_window = 65536",
                "model_auto_compact_token_limit = 52000",
                "project_root_markers = []",
                "",
                f"[model_providers.{provider_name}]",
                'name = "BBO SGLang Responses API"',
                f"base_url = {json.dumps(base_url.rstrip('/'))}",
                'wire_api = "responses"',
            ]
            if self.config.api_key_env:
                lines.append(f"env_key = {json.dumps(self.config.api_key_env)}")
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return config_path
        if self.config.framework == "claude_code":
            settings_path = self._state_dir / "settings.json"
            if not settings_path.exists():
                settings_path.write_text("{}", encoding="utf-8")
        del log_dir
        return None

    def _agent_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        api_key = self._api_key()
        provider = (self.config.provider or "").lower()
        if api_key:
            if provider == "openai":
                env["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                env["ANTHROPIC_API_KEY"] = api_key
            elif provider == "google":
                env["GOOGLE_API_KEY"] = api_key
            elif self.config.api_key_env:
                env[self.config.api_key_env] = api_key
        if self.config.api_base:
            if provider == "openai":
                env["OPENAI_BASE_URL"] = self.config.api_base
            elif provider == "anthropic":
                env["ANTHROPIC_BASE_URL"] = self.config.api_base
        web_key_env = self.config.web_search_api_key_env
        if not web_key_env and self.config.web_search_provider.strip().lower().replace("-", "_") == "serpapi":
            web_key_env = "SERPAPI_API_KEY"
        if web_key_env and os.environ.get(web_key_env):
            env[web_key_env] = os.environ[web_key_env]
        if os.environ.get("SERPAPI_ENDPOINT"):
            env["SERPAPI_ENDPOINT"] = os.environ["SERPAPI_ENDPOINT"]
        if self.config.framework == "nanobot":
            if not self._agent_tools_enabled():
                env["BBO_NANOBOT_NO_TOOL_MODE"] = "1"
            if not self._agent_skills_enabled():
                env["BBO_NANOBOT_NO_SKILL_MODE"] = "1"
        return env

    def _openai_compatible_config(self) -> dict[str, Any]:
        return {
            "api_key": self._api_key(),
            "api_base": self.config.api_base,
            "model": self.config.model,
        }

    def _codex_config(self) -> dict[str, Any]:
        provider = (self.config.provider or "").lower()
        return {
            "env": self._agent_env(),
            "executable": self.config.executable,
            "sandbox": "workspace-write",
            "approval_policy": "never",
            "black_box_required": True,
            "filesystem_boundary": "outer_bwrap_minimal_root",
            "tool_mode": self.config.tool_mode,
            "model": self.config.model,
            "api_base": self.config.api_base,
            "api_key_env": self.config.api_key_env,
            "wire_api": "responses",
            "responses_api_compat": "sglang" if provider == "sglang" else None,
        }

    def _claude_config(self) -> dict[str, Any]:
        provider = (self.config.provider or "").lower()
        api_key = self._api_key()
        api_base = _anthropic_base_url(self.config.api_base)
        env: dict[str, str] = {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_INSTALLATION_CHECKS": "1",
            "DISABLE_TELEMETRY": "1",
        }
        if self.config.model:
            env.update(
                {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.config.model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": self.config.model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": self.config.model,
                    "ANTHROPIC_SMALL_FAST_MODEL": self.config.model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": self.config.model,
                }
            )
        if provider == "claude":
            if api_key:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = api_key
        elif provider == "anthropic" or not provider:
            if api_base:
                env["ANTHROPIC_BASE_URL"] = api_base
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
        else:
            if api_base:
                env["ANTHROPIC_BASE_URL"] = api_base
            if api_key:
                env["ANTHROPIC_AUTH_TOKEN"] = api_key
        return {
            "env": env,
            "model": self.config.model,
            "executable": self.config.executable,
            "messages_api_compat": "sglang" if provider == "sglang" else None,
            "max_output_tokens": 4096 if provider == "sglang" else None,
            "tools": {"type": "preset", "preset": "claude_code"},
            "system_prompt": {"type": "preset", "preset": "claude_code"},
            "permission_mode": "default",
            "black_box_required": True,
            "tool_mode": self.config.tool_mode,
            "sandbox": {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            },
            "setting_sources": [],
            "skills": [],
        }

    def _native_harness_policy(self) -> dict[str, Any]:
        return {
            "framework": self.config.framework,
            "black_box_boundary": "required",
            "filesystem_isolation": "minimal_root_or_framework_sandbox",
            "evaluator_access": "denied",
            "missing_isolation_behavior": "fail_closed",
            "native_tools_preserved": True,
            "native_tool_policy": "framework_default",
            "benchmark_tools_enabled": self._agent_tools_enabled(),
            "benchmark_skills_enabled": self._agent_skills_enabled(),
            "tool_mode": self.config.tool_mode,
            "external_user_configuration": "isolated" if self.config.framework in {"codex", "claude_code"} else "framework_config",
        }

    def _api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        return os.environ.get(self.config.api_key_env)

    def _restore_queue_from_snapshot(self) -> None:
        if not self.config.resume or not self._loaded_resume_snapshot:
            return
        search_space = self._require_search_space()
        restored: list[AgentCandidateEntry] = []
        for item in self._loaded_resume_snapshot.get("queue", []):
            if not isinstance(item, Mapping):
                continue
            try:
                config = search_space.coerce_config(dict(item.get("config", {})), use_defaults=False)
            except Exception:
                continue
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            restored.append(
                AgentCandidateEntry(
                    config=config,
                    call_id=str(item.get("call_id", "restored")),
                    candidate_index=int(item.get("candidate_index", 0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
            self._seen_config_ids.add(identity)
        self._queue = restored
        self._call_index = max(self._call_index, int(self._loaded_resume_snapshot.get("call_index", 0)))

    def _load_resume_snapshot(self) -> dict[str, Any]:
        if not self.config.resume or not self._agent_state_path.exists():
            return {}
        try:
            data = json.loads(self._agent_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _persist_state(self) -> None:
        if self._run_dir is None:
            return
        dump_json(
            self._agent_state_path,
            {
                "algorithm": self.name,
                "framework": self.config.framework,
                "engine": self._engine.name,
                "call_index": self._call_index,
                "history_size": len(self._history),
                "queue": [to_jsonable(entry) for entry in self._queue],
                "seen_config_ids": sorted(self._seen_config_ids),
                "best_config": None if self._best is None else self._best.config,
                "best_score": None if self._best is None else self._best.score,
                "model": self.config.model,
                "provider": self.config.provider,
                "executable": self.config.executable,
                "tool_mode": self.config.tool_mode,
                "max_tool_calls": self.config.max_tool_calls,
                "enabled_tool_names": (
                    None
                    if self.config.enabled_tool_names is None
                    else list(self.config.enabled_tool_names)
                ),
                "optimizer_backend_allowlist": list(
                    self.config.optimizer_backend_allowlist
                ),
                "optimizer_max_calls_per_round": self.config.optimizer_max_calls_per_round,
                "experiment_condition": self.config.experiment_condition,
                "require_analysis_evidence_per_round": self.config.require_analysis_evidence_per_round,
                "required_tool_names_per_round": list(self.config.required_tool_names_per_round),
                "require_candidate_validation_per_round": self.config.require_candidate_validation_per_round,
                "require_optimizer_decision_per_round": self.config.require_optimizer_decision_per_round,
                "enable_memory": self.config.enable_memory,
                "web_search_provider": self.config.web_search_provider,
                "code_backend": self.config.code_backend,
                "docker_image": self.config.docker_image,
                "allow_fallback": self.config.allow_fallback,
                "require_visible_cot": self.config.require_visible_cot,
                "enable_bbo_skills": self.config.enable_bbo_skills,
                "skill_paths": [str(path) for path in self.config.skill_paths],
                "harness_policy": self._native_harness_policy(),
            },
        )

    @property
    def _agent_calls_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_calls.jsonl"

    @property
    def _agent_prompts_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_prompts.jsonl"

    @property
    def _agent_state_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_state.json"

    @property
    def _agent_optimization_trace_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_optimization_trace.jsonl"

    @property
    def _agent_tool_calls_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_tool_calls.jsonl"

    @property
    def _agent_tool_specs_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_tool_specs.json"

    @property
    def _agent_sources_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_web_sources.jsonl"

    @property
    def _agent_memory_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory.jsonl"

    @property
    def _agent_memory_summary_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory_summary.json"

    @property
    def _agent_reasoning_traces_dir(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "reasoning_traces"

    @property
    def _agent_reasoning_metadata_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_reasoning_metadata.jsonl"

    def _require_ready(self) -> None:
        if self._task_spec is None or self._search_space is None:
            raise RuntimeError(f"{self.__class__.__name__}.setup() must be called before use.")

    def _require_task_spec(self) -> TaskSpec:
        self._require_ready()
        assert self._task_spec is not None
        return self._task_spec

    def _require_search_space(self) -> SearchSpace:
        self._require_ready()
        assert self._search_space is not None
        return self._search_space


class NanobotBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Nanobot."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="nanobot", algorithm_name="agentic_nanobot", **kwargs)


class ClaudeCodeBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Claude Code."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="claude_code", algorithm_name="agentic_claude_code", **kwargs)


class CodexBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Codex CLI."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="codex", algorithm_name="agentic_codex", **kwargs)


class OpenAICompatibleBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by OpenAI-compatible function calling."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="openai_compatible", algorithm_name="agentic_openai_compatible", **kwargs)


def _observation_summary(observation: TrialObservation) -> dict[str, Any]:
    return {
        "trial_id": observation.suggestion.trial_id,
        "config": agent_visible_config(observation.suggestion.config),
        "budget": agent_visible_payload(observation.suggestion.budget),
        "status": observation.status.value,
        "objectives": agent_visible_payload(observation.objectives),
        "metrics": agent_visible_metrics(observation.metrics),
        "elapsed_seconds": agent_visible_payload(observation.elapsed_seconds),
        "error_type": observation.error_type,
        "error_message": observation.error_message,
        "timestamp": agent_visible_payload(observation.timestamp),
        "metadata": agent_visible_metadata(observation.metadata),
        "suggestion_metadata": agent_visible_payload(observation.suggestion.metadata),
        "search_action": agent_visible_payload(observation.suggestion.metadata.get("search_action", {})),
    }


def _agent_history_summary(observation: TrialObservation) -> dict[str, Any]:
    """Return compact optimization evidence without benchmark audit metadata."""

    payload: dict[str, Any] = {
        "trial_id": observation.suggestion.trial_id,
        "config": agent_visible_config(observation.suggestion.config),
        "budget": agent_visible_payload(observation.suggestion.budget),
        "status": observation.status.value,
        "objectives": agent_visible_payload(observation.objectives),
        "metrics": agent_visible_metrics(observation.metrics),
    }
    search_action = agent_visible_payload(
        observation.suggestion.metadata.get("search_action", {})
    )
    if search_action:
        payload["search_action"] = search_action
    if observation.error_type:
        payload["error_type"] = observation.error_type
    if observation.error_message:
        payload["error_message"] = observation.error_message
    return payload


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["result"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive cross-thread propagation.
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box["error"]
    return result_box["result"]


def _normalize_skill_paths(
    skill_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> tuple[Path, ...]:
    if skill_paths is None:
        return ()
    if isinstance(skill_paths, (str, Path)):
        raw_paths = [skill_paths]
    else:
        raw_paths = list(skill_paths)
    return tuple(Path(path).expanduser() for path in raw_paths)


def normalize_agent_tool_mode(raw: str) -> str:
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "no_tools": "no_tool",
        "none": "no_tool",
        "disabled": "no_tool",
        "off": "no_tool",
        "false": "no_tool",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in AGENT_TOOL_MODES:
        choices = ", ".join(AGENT_TOOL_MODE_CLI_CHOICES)
        raise ValueError(f"tool_mode must be one of: {choices}.")
    return normalized


def _optimizer_visible_task_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    visible: dict[str, Any] = {}
    transforms = metadata.get("parameter_transforms")
    if isinstance(transforms, Mapping):
        visible["parameter_transforms"] = dict(transforms)
    protocol = metadata.get("benchmark_protocol")
    if isinstance(protocol, Mapping):
        candidate_budget = protocol.get("candidate_budget")
        if isinstance(candidate_budget, Mapping):
            visible["benchmark_protocol"] = {
                "candidate_budget": dict(candidate_budget)
            }
    return visible


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _packaged_bbo_nanobot_skills_dir() -> Path:
    return Path(__file__).with_name("skills")


def _discover_nanobot_skill_dirs(path: Path) -> list[Path]:
    source = path if path.is_absolute() else path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Nanobot skill path not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Nanobot skill path must be a directory: {source}")
    if (source / "SKILL.md").exists():
        return [source]
    skill_dirs = sorted(child for child in source.iterdir() if child.is_dir() and (child / "SKILL.md").exists())
    if not skill_dirs:
        raise ValueError(f"Nanobot skill path contains no skill directories with SKILL.md: {source}")
    return skill_dirs


def _nanobot_skill_name(skill_dir: Path) -> str:
    source = skill_dir if skill_dir.is_absolute() else skill_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Nanobot skill directory not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Nanobot skill source must be a directory: {source}")
    skill_file = source / "SKILL.md"
    if not skill_file.exists():
        raise ValueError(f"Nanobot skill directory is missing SKILL.md: {source}")
    declared_name = _read_skill_frontmatter_name(skill_file)
    name = declared_name or source.name
    if name != source.name:
        raise ValueError(f"Nanobot skill name `{name}` must match directory name `{source.name}`.")
    if not _NANOBOT_SKILL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Nanobot skill name `{name}` must use lowercase letters, digits, and single hyphens only."
        )
    return name


def _skill_index_entry(skill_name: str) -> dict[str, Any]:
    groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill_name, ())
    return {
        "name": skill_name,
        "search_intent": SKILL_TO_SEARCH_INTENT.get(skill_name),
        "proposal_allowed": skill_name not in NON_PROPOSAL_BBO_SKILLS,
        "evidence_tools": [list(group) for group in groups],
    }


def _read_skill_frontmatter_name(skill_file: Path) -> str | None:
    try:
        lines = skill_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return None
        if stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    return None


def _nanobot_provider_key(provider: str) -> str:
    return {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "gemini",
        "ollama": "ollama",
        "azure": "azure_openai",
        "deepseek": "deepseek",
    }.get(provider, "custom")


def _anthropic_base_url(api_base: str | None) -> str | None:
    if not api_base:
        return None
    normalized = api_base.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


__all__ = [
    "ClaudeCodeBBOAlgorithm",
    "CodexBBOAlgorithm",
    "GeneralAgentBBOAlgorithm",
    "GeneralAgentConfig",
    "GeneralAgentValidationError",
    "NanobotBBOAlgorithm",
    "OpenAICompatibleBBOAlgorithm",
    "AGENT_TOOL_MODE_CLI_CHOICES",
    "AGENT_TOOL_MODES",
    "normalize_agent_tool_mode",
    "parse_agent_candidate_payload",
    "search_space_schema",
]
