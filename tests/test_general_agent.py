from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.agentic import (
    AgentResult,
    GeneralAgentValidationError,
    AgentWorkCopy,
    MockAgentEngine,
    NanobotBBOAlgorithm,
    NanobotEngine,
    parse_agent_candidate_payload,
)
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger, SearchSpace, StringParam
from bbo.core import EvaluationResult, TrialObservation, TrialStatus, TrialSuggestion
from bbo.run import build_arg_parser
from bbo.tasks import create_task
from conftest import create_agent_test_task


class WorkspaceSerpApiMockHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        type(self).requests.append({"path": parsed.path, "params": params})
        body = json.dumps(
            {
                "organic_results": [
                    {
                        "title": "Workspace SERP prior",
                        "link": "https://example.test/workspace-serp",
                        "snippet": "SerpAPI workspace bridge result.",
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReasoningMetadataMockEngine(MockAgentEngine):
    async def run_agent(self, *args, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
        result = await super().run_agent(*args, extra_env=extra_env, **kwargs)
        if extra_env:
            call_id = extra_env["BBO_AGENT_CALL_ID"]
            trace_dir = Path(extra_env["BBO_NANOBOT_REASONING_DIR"])
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{call_id}_reasoning.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "call_id": call_id,
                        "reasoning_visible": True,
                        "reasoning_content": "mock visible reasoning",
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = Path(extra_env["BBO_NANOBOT_REASONING_METADATA_PATH"])
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "call_id": call_id,
                            "reasoning_visible": True,
                            "reasoning_chars": len("mock visible reasoning"),
                            "trace_path": str(trace_path),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return result


def _make_observation(
    trial_id: int,
    config: dict[str, object],
    loss: float,
    *,
    search_intent: str = "exploration",
    skill: str | None = None,
    parent_trials: list[str] | None = None,
    reference_trials: list[str] | None = None,
    hypothesis: str | None = None,
    repaired: bool = False,
) -> TrialObservation:
    action = {
        "search_intent": search_intent,
        "skill": skill,
        "parent_trials": parent_trials or [],
        "reference_trials": reference_trials or [],
        "hypothesis": hypothesis,
        "modified_variables": ["x1"],
        "repaired": repaired,
    }
    suggestion = TrialSuggestion(
        trial_id=trial_id,
        config=dict(config),
        metadata={
            "search_intent": search_intent,
            "search_action": action,
        },
    )
    return TrialObservation.from_evaluation(
        suggestion,
        EvaluationResult(status=TrialStatus.SUCCESS, objectives={"loss": loss}),
    )


def _call_workspace_tool(artifacts: dict[str, str], tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, artifacts["agent_workspace_tool_py"], tool_name, json.dumps(arguments)],
        cwd=artifacts["agent_workspace"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True, payload
    return payload["result"]


def _write_nanobot_skill_read_log(work_copy: AgentWorkCopy, call_id: str, skill_name: str) -> None:
    log_dir = Path(work_copy.extra["log_dir"]) / call_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "mock_agent-end.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": f"skills/{skill_name}/SKILL.md"}),
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_general_agent_algorithms_are_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")
    web_search_action = next(action for action in parser._actions if action.dest == "agent_web_search_provider")

    assert "agentic_nanobot" in ALGORITHM_REGISTRY
    assert "nanobot" in ALGORITHM_REGISTRY
    assert "agentic_claude_code" in ALGORITHM_REGISTRY
    assert "claude_code" in ALGORITHM_REGISTRY
    assert "claude-code" in ALGORITHM_REGISTRY
    assert "agentic_openai_compatible" in ALGORITHM_REGISTRY
    assert "openai_compatible_agent" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["nanobot"].family == "agentic"
    assert ALGORITHM_REGISTRY["claude_code"].family == "agentic"
    assert ALGORITHM_REGISTRY["agentic_openai_compatible"].family == "agentic"
    assert "nanobot" in algorithm_action.choices
    assert "claude_code" in algorithm_action.choices
    assert "agentic_openai_compatible" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "claude-code"]).algorithm == "claude-code"
    assert parser.parse_args(["--algorithm", "agentic_openai_compatible"]).agent_tool_mode == "function_calling"
    assert parser.parse_args(["--algorithm", "nanobot"]).agent_candidates_per_call == 1
    prompt_style_action = next(action for action in parser._actions if action.dest == "agent_prompt_style")
    assert set(prompt_style_action.choices) == {"workspace"}
    assert set(web_search_action.choices) == {"disabled", "mock", "serpapi"}
    assert parser.parse_args(["--agent-web-search-provider", "serpapi"]).agent_web_search_provider == "serpapi"
    assert parser.parse_args(["--algorithm", "nanobot", "--agent-require-visible-cot"]).agent_require_visible_cot is True
    assert parser.parse_args(["--algorithm", "nanobot"]).agent_tool_mode == "workspace_json"
    assert (
        parser.parse_args(["--algorithm", "nanobot", "--agent-tool-mode", "function_calling"]).agent_tool_mode
        == "function_calling"
    )
    assert parser.parse_args(["--algorithm", "nanobot", "--agent-tool-mode", "no-tool"]).agent_tool_mode == "no_tool"
    assert parser.parse_args(["--algorithm", "nanobot", "--agent-tool-mode", "no-tools"]).agent_tool_mode == "no_tool"
    assert parser.parse_args(["--algorithm", "nanobot", "--agent-tool-mode", "none"]).agent_tool_mode == "no_tool"
    assert parser.parse_args(["--algorithm", "agentic_openai_compatible"]).agent_tool_mode == "function_calling"
    assert parser.parse_args(["--algorithm", "nanobot"]).agent_enable_bbo_skills is False
    assert parser.parse_args(["--algorithm", "nanobot"]).agent_skill_path is None
    skill_args = parser.parse_args(
        [
            "--algorithm",
            "nanobot",
            "--agent-enable-bbo-skills",
            "--agent-skill-path",
            "skills/custom-one",
            "--agent-skill-path",
            "skills/custom-two",
        ]
    )
    assert skill_args.agent_enable_bbo_skills is True
    assert skill_args.agent_skill_path == [Path("skills/custom-one"), Path("skills/custom-two")]


def test_workspace_prompt_uses_general_strategy_instead_of_mandatory_gp(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    prompt = algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)
    workspace = Path(algorithm.artifact_paths["agent_workspace"])
    instructions = (workspace / "instructions.md").read_text(encoding="utf-8")
    tools_text = (workspace / "TOOLS.md").read_text(encoding="utf-8")

    assert "gp_expected_improvement.py" not in prompt
    assert "gp_expected_improvement.py" not in instructions
    assert "Use the workspace as the source of truth" in prompt
    assert "Read `TOOLS.md` before using workspace tools or BBO Python API helpers." in prompt
    assert "Produce exactly one new candidate configuration." in prompt
    assert "scratch/agent_call_00000/" in prompt
    assert "validate_candidate" in prompt
    assert "final_candidate.json" in prompt
    assert "harness treats the file as authoritative" in prompt
    assert "search_action" in prompt
    assert "You may create temporary scratch files or short analysis scripts under a new" in prompt
    assert "Do not modify protected files" in prompt
    assert "`history.jsonl`" in prompt
    assert "`TOOLS.md`" in prompt
    assert "Return only valid raw JSON with exactly this shape" in prompt
    assert "from bbo_tools import BBO" not in prompt
    assert "BBO().validate_candidate" not in prompt
    assert '"skill"' not in prompt
    assert "skill" not in prompt.lower()
    assert "skill" not in instructions.lower()
    assert "skill" not in tools_text.lower()
    assert "If you set `search_action.skill` to a non-null skill name" not in prompt
    assert "BBO skills are instruction documents under `skills/<skill-name>/SKILL.md`" not in prompt
    assert "from bbo_tools import BBO" in tools_text
    assert "bbo.validate_candidate(candidate)" in tools_text
    assert "do not call invented methods such as" in tools_text


def test_nanobot_no_tool_mode_omits_bbo_tool_surface_but_keeps_basic_workspace_access(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="no-tool",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    prompt = algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)
    workspace = Path(algorithm.artifact_paths["agent_workspace"])
    instructions = (workspace / "instructions.md").read_text(encoding="utf-8")
    config = json.loads((Path(algorithm.artifact_paths["agent_state_dir"]) / "config.json").read_text(encoding="utf-8"))
    env = algorithm._agent_env()

    assert "read_file" in prompt
    assert "`exec`" in prompt
    assert "skill" not in prompt.lower()
    assert "TOOLS.md" not in prompt
    assert "bbo_tools.py" not in prompt
    assert "validate_candidate" not in prompt
    assert "final_candidate.json" in prompt
    assert '"skill"' not in prompt
    assert "`task.md`" in prompt
    assert "`space.json`" in prompt
    assert "`history.jsonl`" in prompt
    assert "read_file" in instructions
    assert "`exec`" in instructions
    assert "skill" not in instructions.lower()
    assert "TOOLS.md" not in instructions
    assert "bbo_tools.py" not in instructions
    assert "validate_candidate" not in instructions

    forbidden_workspace_paths = [
        "TOOLS.md",
        "tool_specs.json",
        "bbo_tool.py",
        "bbo_tools.py",
        "bbo_tool_config.json",
        "bbo_workspace_audit.py",
        "gp_expected_improvement.py",
        "python_environment.md",
        "examples",
        "skills",
    ]
    for relative_path in forbidden_workspace_paths:
        assert not (workspace / relative_path).exists()
    forbidden_artifacts = {
        "agent_tool_specs_json",
        "agent_workspace_tool_py",
        "agent_workspace_bbo_tools_py",
        "agent_workspace_tool_config_json",
        "agent_workspace_tools_md",
        "agent_workspace_gp_example_py",
        "agent_workspace_gp_entrypoint_py",
        "agent_workspace_python_environment_md",
        "agent_workspace_skills_dir",
        "agent_tool_calls_jsonl",
    }
    assert forbidden_artifacts.isdisjoint(algorithm.artifact_paths)
    assert env["BBO_NANOBOT_NO_TOOL_MODE"] == "1"
    assert "BBO_NANOBOT_PARSE_TEXT_TOOL_CALLS" not in env
    assert config["agents"]["defaults"].get("max_tool_iterations") != 0
    assert config["tools"]["exec"]["enable"] is True
    assert config["tools"]["web"]["enable"] is False
    assert config["tools"]["my"]["enable"] is False


def test_nanobot_no_tool_mode_rejects_bbo_skills(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BBO skills require"):
        NanobotBBOAlgorithm(
            engine=MockAgentEngine(seed=13),
            run_dir=tmp_path / "agent_run",
            tool_mode="no_tool",
            enable_bbo_skills=True,
        )


def test_workspace_prompt_includes_retry_feedback_after_failed_attempt(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    first_prompt = algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)
    retry_prompt = algorithm._build_agent_prompt(
        call_id="agent_call_00001",
        attempt_index=1,
        last_error="Provider response is not valid JSON: <tool_call>" + ("x" * 1000),
    )

    assert "Previous attempt failed and was not accepted" not in first_prompt
    assert "Previous attempt failed and was not accepted" in retry_prompt
    assert "Provider response is not valid JSON" in retry_prompt
    assert "Do not return tool-call XML" in retry_prompt
    assert "Rewrite `final_candidate.json`" in retry_prompt
    assert "python -m json.tool final_candidate.json" in retry_prompt
    assert "x" * 900 not in retry_prompt



def test_workspace_final_candidate_file_takes_precedence_over_chat_response(tmp_path: Path) -> None:
    class FileFirstEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            workspace = work_copy.workspace_root or work_copy.project_root
            file_payload = {"candidates": [{"config": {"x1": 0.25, "x2": 0.75}}]}
            (workspace / "final_candidate.json").write_text(json.dumps(file_payload), encoding="utf-8")
            chat_payload = {"candidates": [{"config": {"x1": 0.8, "x2": 0.8}}]}
            return AgentResult(status="success", answer=json.dumps(chat_payload))

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=FileFirstEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="no_tool",
        max_retries=0,
        allow_fallback=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.25, "x2": 0.75}
    row = json.loads(Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()[-1])
    assert row["candidate_source"] == "workspace_candidate_file"
    assert row["candidate_file"] == "final_candidate.json"


def test_workspace_final_candidate_file_recovers_from_malformed_chat_response(tmp_path: Path) -> None:
    class FileRecoveryEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            workspace = work_copy.workspace_root or work_copy.project_root
            payload = {"candidates": [{"config": {"x1": 0.3, "x2": 0.7}}]}
            (workspace / "final_candidate.json").write_text(json.dumps(payload), encoding="utf-8")
            return AgentResult(status="success", answer="Now let me validate the final answer...")

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=FileRecoveryEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="no_tool",
        max_retries=0,
        allow_fallback=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.3, "x2": 0.7}



def test_malformed_response_recovers_last_successfully_validated_candidate(tmp_path: Path) -> None:
    class TruncatedAfterValidationEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, work_copy, kwargs
            return AgentResult(
                status="success",
                answer="Validation succeeded. Now I will write final_candidate.json:",
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=TruncatedAfterValidationEngine(seed=13),
        run_dir=tmp_path / "validated_recovery",
        tool_mode="workspace_json",
        enabled_tool_names=("validate_candidate",),
        max_retries=0,
        allow_fallback=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    calls_path = Path(algorithm.artifact_paths["agent_tool_calls_jsonl"])
    calls_path.write_text(
        json.dumps(
            {
                "agent_call_id": "agent_call_00000",
                "tool_name": "validate_candidate",
                "arguments": {"candidate": {"x1": 0.35, "x2": 0.65}},
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.35, "x2": 0.65}
    row = json.loads(calls_path.with_name("agent_calls.jsonl").read_text().splitlines()[-1])
    assert row["candidate_source"] == "validated_tool_recovery"
    assert "Provider response is not valid JSON" in row["agent_response_error"]


def test_validated_recovery_rejects_unvalidated_or_multi_candidate_records(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "no_unsafe_recovery",
        tool_mode="workspace_json",
        enabled_tool_names=("validate_candidates",),
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    calls_path = Path(algorithm.artifact_paths["agent_tool_calls_jsonl"])
    calls_path.write_text(
        json.dumps(
            {
                "agent_call_id": "agent_call_test",
                "tool_name": "validate_candidates",
                "arguments": {
                    "candidates": [
                        {"x1": 0.2, "x2": 0.8},
                        {"x1": 0.8, "x2": 0.2},
                    ]
                },
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert algorithm._recover_successfully_validated_candidate(
        "agent_call_test", task.spec.search_space
    ) is None

def test_workspace_final_candidate_file_is_cleared_before_retry(tmp_path: Path) -> None:
    class StaleFileEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            workspace = work_copy.workspace_root or work_copy.project_root
            self.calls += 1
            if self.calls == 1:
                duplicate = {"candidates": [{"config": {"x1": 0.2, "x2": 0.2}}]}
                (workspace / "final_candidate.json").write_text(json.dumps(duplicate), encoding="utf-8")
                return AgentResult(status="success", answer="incomplete response")
            fresh = {"candidates": [{"config": {"x1": 0.8, "x2": 0.8}}]}
            return AgentResult(status="success", answer=json.dumps(fresh))

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=StaleFileEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="no_tool",
        max_retries=1,
        allow_fallback=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.2, "x2": 0.2}, 10.0))

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.8, "x2": 0.8}
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["candidate_source"] == "workspace_candidate_file"
    assert rows[0]["accepted_candidates"] == 0
    assert rows[1]["candidate_source"] == "agent_response"
    assert rows[1]["accepted_candidates"] == 1


def test_nanobot_bbo_skills_are_opt_in_and_copied_to_workspace(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    default_algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "default_agent_run",
        tool_mode="workspace_json",
    )
    default_algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    default_workspace = Path(default_algorithm.artifact_paths["agent_workspace"])
    assert not (default_workspace / "skills").exists()

    skilled_algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "skilled_agent_run",
        tool_mode="workspace_json",
        enable_bbo_skills=True,
    )
    skilled_algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    skills_dir = Path(skilled_algorithm.artifact_paths["agent_workspace_skills_dir"])
    workspace = Path(skilled_algorithm.artifact_paths["agent_workspace"])
    instructions = (workspace / "instructions.md").read_text(encoding="utf-8")
    tools_text = (workspace / "TOOLS.md").read_text(encoding="utf-8")
    assert "Skills are optional references, not mandatory steps." in instructions
    assert "Follow `TOOLS.md` for skill-read, evidence, validation, and declaration rules." in instructions
    assert "Skills are instruction documents, not callable tools or Python functions" not in instructions
    assert "BBO skills are instruction documents under `skills/<skill-name>/SKILL.md`" in tools_text

    expected = {
        "initialize-search",
        "refine-incumbent",
        "follow-promising-direction",
        "isolate-variable-effect",
        "probe-variable-interaction",
        "recombine-complementary-elites",
        "escape-search-stagnation",
        "surrogate-guided-proposal",
        "repair-invalid-candidate",
        "distill-search-memory",
    }
    assert {path.name for path in skills_dir.iterdir() if path.is_dir()} == expected
    skill_index = json.loads((skills_dir / "index.json").read_text(encoding="utf-8"))
    index_by_name = {item["name"]: item for item in skill_index["skills"]}
    assert set(index_by_name) == expected
    assert index_by_name["surrogate-guided-proposal"]["evidence_tools"] == [
        ["fit_and_check_surrogate"],
        ["score_virtual_candidates"],
        ["validate_candidate", "validate_candidates"],
    ]
    assert index_by_name["distill-search-memory"]["proposal_allowed"] is False
    counterexample_skills = {
        "refine-incumbent",
        "follow-promising-direction",
        "escape-search-stagnation",
        "surrogate-guided-proposal",
        "repair-invalid-candidate",
    }
    for skill_name in expected:
        skill_md = skills_dir / skill_name / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
        assert f"name: {skill_name}" in text
        assert "always: true" not in text
        assert "[TODO" not in text
        assert "## Use When" in text
        assert "## Do Not Use When" in text
        assert "## Positive Example" in text
        if skill_name in counterexample_skills:
            assert "## Counterexample" in text

    prompt = skilled_algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)
    instructions = (workspace / "instructions.md").read_text(encoding="utf-8")
    tools_text = (workspace / "TOOLS.md").read_text(encoding="utf-8")
    assert "Read `TOOLS.md` before using workspace tools, BBO Python API helpers, or" in prompt
    assert "Skill use is optional: use a skill only when it clearly helps" in prompt
    assert "skills/index.json" not in prompt
    assert "summarize_objective_metrics" not in prompt
    assert "search_action.skill` to a non-null skill name" not in prompt
    assert "Do not rely only on a skill summary or description" in tools_text
    assert "load every skill" not in prompt
    assert "repair-invalid-candidate" not in prompt
    assert "skills/index.json" in tools_text
    assert "summarize_objective_metrics" in tools_text
    assert "repair-invalid-candidate" in tools_text
    assert "skills/: optional Nanobot BBO skill reference library" in instructions
    assert "skills/index.json" not in instructions
    assert "Skills are optional references, not mandatory steps" in instructions
    assert "final search_action.skill is non-null" not in instructions
    assert "use the BBO skills as a staged workflow" not in instructions


def test_declared_bbo_skill_requires_reading_skill_md_before_acceptance(tmp_path: Path) -> None:
    class SkillClaimThenDirectEngine(MockAgentEngine):
        async def run_agent(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            self.calls += 1
            if self.calls == 1:
                return AgentResult(
                    status="success",
                    answer=json.dumps(
                        {
                            "candidates": [
                                {
                                    "config": {"x1": 0.5, "x2": 0.5},
                                    "rationale": "claimed local refinement",
                                    "search_action": {
                                        "skill": "refine-incumbent",
                                        "parent_trials": [],
                                        "reference_trials": [],
                                        "hypothesis": None,
                                        "change_summary": "claim without reading skill file",
                                    },
                                }
                            ]
                        }
                    ),
                )
            return AgentResult(
                status="success",
                answer=json.dumps(
                    {
                        "candidates": [
                            {
                                "config": {"x1": 0.7, "x2": 0.7},
                                "rationale": "direct candidate after correction",
                                "search_action": {
                                    "skill": None,
                                    "parent_trials": [],
                                    "reference_trials": [],
                                    "hypothesis": None,
                                    "change_summary": "direct candidate without skill",
                                },
                            }
                        ]
                    }
                ),
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    engine = SkillClaimThenDirectEngine(seed=13)
    algorithm = NanobotBBOAlgorithm(
        engine=engine,
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=1,
        allow_fallback=False,
        enable_bbo_skills=True,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.7, "x2": 0.7}
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["validation_error"].startswith("Agent set search_action.skill to `refine-incumbent`")
    assert "skills/refine-incumbent/SKILL.md" in rows[0]["validation_error"]
    assert rows[1]["accepted_candidates"] == 1


def test_declared_bbo_skill_requires_required_evidence_tools(tmp_path: Path) -> None:
    class SkillReadWithoutEvidenceThenDirectEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, *, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            self.calls += 1
            call_id = (extra_env or {})["BBO_AGENT_CALL_ID"]
            if self.calls == 1:
                _write_nanobot_skill_read_log(work_copy, call_id, "refine-incumbent")
                return AgentResult(
                    status="success",
                    answer=json.dumps(
                        {
                            "candidates": [
                                {
                                    "config": {"x1": 0.5, "x2": 0.5},
                                    "rationale": "claimed local refinement",
                                    "search_action": {
                                        "skill": "refine-incumbent",
                                        "parent_trials": [],
                                        "reference_trials": [],
                                        "hypothesis": None,
                                        "change_summary": "claim with skill read but no evidence tools",
                                    },
                                }
                            ]
                        }
                    ),
                )
            return AgentResult(
                status="success",
                answer=json.dumps(
                    {
                        "candidates": [
                            {
                                "config": {"x1": 0.7, "x2": 0.7},
                                "rationale": "direct candidate after correction",
                                "search_action": {
                                    "skill": None,
                                    "parent_trials": [],
                                    "reference_trials": [],
                                    "hypothesis": None,
                                    "change_summary": "direct candidate without skill",
                                },
                            }
                        ]
                    }
                ),
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=SkillReadWithoutEvidenceThenDirectEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=1,
        allow_fallback=False,
        enable_bbo_skills=True,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.7, "x2": 0.7}
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["validation_error"].startswith("Agent declared BBO skill `refine-incumbent`")
    assert "required BBO evidence tools" in rows[0]["validation_error"]
    assert rows[1]["accepted_candidates"] == 1


def test_declared_bbo_skill_with_required_evidence_tools_is_accepted(tmp_path: Path) -> None:
    class SkilledEvidenceEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, *, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            self.calls += 1
            call_id = (extra_env or {})["BBO_AGENT_CALL_ID"]
            _write_nanobot_skill_read_log(work_copy, call_id, "refine-incumbent")
            workspace = work_copy.workspace_root or work_copy.project_root
            candidate = {"config": {"x1": 0.5, "x2": 0.5}}
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from bbo_tools import BBO; "
                        "bbo = BBO(); "
                        f"candidate = {candidate!r}; "
                        "bbo.history_overview(); "
                        "bbo.find_nearest_trials(candidate, k=1); "
                        "bbo.estimate_local_effects(candidate, variables=['x1'], local_radius=1.0); "
                        "bbo.validate_candidate(candidate)"
                    ),
                ],
                cwd=workspace,
                env={**os.environ, **(extra_env or {})},
                text=True,
                capture_output=True,
                check=False,
            )
            assert probe.returncode == 0, probe.stderr
            return AgentResult(
                status="success",
                answer=json.dumps(
                    {
                        "candidates": [
                            {
                                "config": candidate["config"],
                                "rationale": "local refinement with required evidence",
                                "search_action": {
                                    "skill": "refine-incumbent",
                                    "parent_trials": [],
                                    "reference_trials": [],
                                    "hypothesis": None,
                                    "change_summary": "used required refine evidence tools",
                                },
                            }
                        ]
                    }
                ),
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=SkilledEvidenceEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=0,
        allow_fallback=False,
        enable_bbo_skills=True,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.5, "x2": 0.5}
    assert suggestion.metadata["search_action"]["skill"] == "refine-incumbent"
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["accepted_candidates"] == 1
    assert rows[0]["accepted_search_actions"][0]["skill"] == "refine-incumbent"
    assert rows[0]["accepted_search_actions"][0]["skill_audit"]["declared_skill"] == "refine-incumbent"
    assert rows[0]["accepted_search_actions"][0]["skill_audit"]["inferred_skill"] == "refine-incumbent"
    assert rows[0]["accepted_search_actions"][0]["skill_audit"]["compliance"] == "declared_and_supported"
    tool_records = [
        json.loads(line)
        for line in Path(algorithm.artifact_paths["agent_tool_calls_jsonl"]).read_text().splitlines()
        if line.strip()
    ]
    assert {record["tool_name"] for record in tool_records} >= {
        "get_history_overview",
        "find_nearest_trials",
        "estimate_local_effects",
        "validate_candidate",
    }


def test_skill_read_and_evidence_are_audited_when_agent_leaves_skill_null(tmp_path: Path) -> None:
    class SkillReadEvidenceButNullEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, *, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            call_id = (extra_env or {})["BBO_AGENT_CALL_ID"]
            _write_nanobot_skill_read_log(work_copy, call_id, "isolate-variable-effect")
            workspace = work_copy.workspace_root or work_copy.project_root
            candidate = {"x1": 0.25, "x2": 0.3}
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from bbo_tools import BBO; "
                        "bbo = BBO(); "
                        f"candidate = {candidate!r}; "
                        "bbo.compare_trials([0, 1]); "
                        "bbo.find_nearest_trials('trial_0', k=1); "
                        "bbo.validate_candidate(candidate)"
                    ),
                ],
                cwd=workspace,
                env={**os.environ, **(extra_env or {})},
                text=True,
                capture_output=True,
                check=False,
            )
            assert probe.returncode == 0, probe.stderr
            return AgentResult(
                status="success",
                answer=json.dumps(
                    {
                        "candidates": [
                            {
                                "config": candidate,
                                "rationale": "controlled one-variable change from the incumbent",
                                "search_action": {
                                    "skill": None,
                                    "parent_trials": [0],
                                    "reference_trials": [0, 1],
                                    "hypothesis": "Changing x1 alone tests the local variable effect.",
                                    "change_summary": "Modified only x1 from trial 0.",
                                },
                            }
                        ]
                    }
                ),
            )

    task = create_agent_test_task(max_evaluations=6, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=SkillReadEvidenceButNullEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=0,
        allow_fallback=False,
        enable_bbo_skills=True,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.2, "x2": 0.3}, 1.0))
    algorithm.tell(_make_observation(1, {"x1": 0.8, "x2": 0.3}, 2.0))

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.25, "x2": 0.3}
    action = suggestion.metadata["search_action"]
    audit = action["skill_audit"]
    assert action["skill"] is None
    assert audit["declared_skill"] is None
    assert audit["read_skills"] == ["isolate-variable-effect"]
    assert audit["inferred_skill"] == "isolate-variable-effect"
    assert audit["candidate_pattern"] == "single_variable_change_from_incumbent"
    assert audit["candidate_pattern_evidence"]["modified_variables"] == ["x1"]
    assert audit["compliance"] == "read_and_inferred_but_not_declared"
    assert audit["skill_evidence"]["isolate-variable-effect"]["read_skill_md"] is True
    assert audit["skill_evidence"]["isolate-variable-effect"]["missing_tool_groups"] == []
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["accepted_search_actions"][0]["skill_audit"] == audit


def test_non_proposal_bbo_skill_cannot_be_final_candidate_skill(tmp_path: Path) -> None:
    class MemorySkillThenDirectEngine(MockAgentEngine):
        async def run_agent(self, session_id, message, work_copy, *, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
            del session_id, message, kwargs
            self.calls += 1
            call_id = (extra_env or {})["BBO_AGENT_CALL_ID"]
            if self.calls == 1:
                _write_nanobot_skill_read_log(work_copy, call_id, "distill-search-memory")
                return AgentResult(
                    status="success",
                    answer=json.dumps(
                        {
                            "candidates": [
                                {
                                    "config": {"x1": 0.5, "x2": 0.5},
                                    "rationale": "incorrect memory skill candidate",
                                    "search_action": {
                                        "skill": "distill-search-memory",
                                        "parent_trials": [],
                                        "reference_trials": [],
                                        "hypothesis": None,
                                        "change_summary": "incorrectly used memory skill as proposal",
                                    },
                                }
                            ]
                        }
                    ),
                )
            return AgentResult(
                status="success",
                answer=json.dumps({"candidates": [{"config": {"x1": 0.8, "x2": 0.8}, "rationale": "direct"}]}),
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MemorySkillThenDirectEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=1,
        allow_fallback=False,
        enable_bbo_skills=True,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.8, "x2": 0.8}
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert "memory maintenance skill" in rows[0]["validation_error"]


def test_nanobot_custom_skill_path_is_copied_to_workspace(tmp_path: Path) -> None:
    source_root = tmp_path / "source_skills"
    custom_skill = source_root / "custom-bbo-prior"
    custom_skill.mkdir(parents=True)
    (custom_skill / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: custom-bbo-prior",
                "description: Apply a custom BBO prior. Use when a benchmark run needs a provided task-specific search heuristic.",
                "---",
                "",
                "# Custom BBO Prior",
                "",
                "Use this custom prior only when it matches the current task evidence.",
            ]
        ),
        encoding="utf-8",
    )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        skill_paths=[source_root],
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    copied = Path(algorithm.artifact_paths["agent_workspace_skills_dir"]) / "custom-bbo-prior" / "SKILL.md"
    assert copied.exists()
    assert "custom BBO prior" in copied.read_text(encoding="utf-8")


def test_declared_custom_skill_requires_reading_skill_md_before_acceptance(tmp_path: Path) -> None:
    source_root = tmp_path / "source_skills"
    custom_skill = source_root / "custom-bbo-prior"
    custom_skill.mkdir(parents=True)
    (custom_skill / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: custom-bbo-prior",
                "description: Apply a custom BBO prior.",
                "---",
                "",
                "# Custom BBO Prior",
            ]
        ),
        encoding="utf-8",
    )

    class CustomSkillClaimThenDirectEngine(MockAgentEngine):
        async def run_agent(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            self.calls += 1
            if self.calls == 1:
                return AgentResult(
                    status="success",
                    answer=json.dumps(
                        {
                            "candidates": [
                                {
                                    "config": {"x1": 0.5, "x2": 0.5},
                                    "rationale": "claimed custom skill",
                                    "search_action": {
                                        "skill": "custom-bbo-prior",
                                        "parent_trials": [],
                                        "reference_trials": [],
                                        "hypothesis": None,
                                        "change_summary": "claim without reading custom skill file",
                                    },
                                }
                            ]
                        }
                    ),
                )
            return AgentResult(
                status="success",
                answer=json.dumps(
                    {
                        "candidates": [
                            {
                                "config": {"x1": 0.7, "x2": 0.7},
                                "rationale": "direct candidate after correction",
                                "search_action": {
                                    "skill": None,
                                    "parent_trials": [],
                                    "reference_trials": [],
                                    "hypothesis": None,
                                    "change_summary": "direct candidate without skill",
                                },
                            }
                        ]
                    }
                ),
            )

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=CustomSkillClaimThenDirectEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        max_retries=1,
        allow_fallback=False,
        skill_paths=[source_root],
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 0.7, "x2": 0.7}
    rows = [json.loads(line) for line in Path(algorithm.artifact_paths["agent_calls_jsonl"]).read_text().splitlines()]
    assert rows[0]["validation_error"].startswith("Agent set search_action.skill to `custom-bbo-prior`")
    assert "skills/custom-bbo-prior/SKILL.md" in rows[0]["validation_error"]
    assert rows[1]["accepted_candidates"] == 1


def test_nanobot_provider_config_uses_api_key_env_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-key")
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        provider="deepseek",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    config = json.loads((Path(algorithm.artifact_paths["agent_state_dir"]) / "config.json").read_text(encoding="utf-8"))

    assert config["agents"]["defaults"]["provider"] == "deepseek"
    assert config["providers"]["deepseek"]["api_key"] == "${DEEPSEEK_API_KEY}"
    assert "secret-test-key" not in json.dumps(config)


def test_direct_json_prompt_style_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt_style must be `workspace`"):
        NanobotBBOAlgorithm(
            engine=MockAgentEngine(seed=13),
            run_dir=tmp_path / "agent_run",
            prompt_style="direct_json",
        )


def test_workspace_prompt_includes_retry_feedback_after_direct_json_removal(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=13),
        run_dir=tmp_path / "agent_run",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    retry_prompt = algorithm._build_agent_prompt(
        call_id="agent_call_00001",
        attempt_index=1,
        last_error="Agent response must contain exactly one top-level key: `candidates`.",
    )

    assert "Previous attempt failed and was not accepted" in retry_prompt
    assert "Agent response must contain exactly one top-level key" in retry_prompt
    assert "BBO_CONTEXT_JSON" not in retry_prompt
    assert "Return only valid raw JSON" in retry_prompt


def test_parse_agent_candidate_payload_accepts_config_wrappers() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {"config": {"x1": 0.5, "x2": 0.5}, "rationale": "center probe"},
                {"config": {"x1": 0.1, "x2": 0.9}},
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}, {"x1": 0.1, "x2": 0.9}]
    assert candidates[0].metadata["rationale"] == "center probe"


def test_parse_agent_candidate_payload_accepts_cli_preamble() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = """
    Using config: /tmp/config.json

    nanobot
    {"candidates": [{"config": {"x1": 0.5, "x2": 0.5}}]}
    """

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}]


def test_parse_agent_candidate_payload_repairs_wrapped_string_newlines() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = '''nanobot
    {
      "candidates": [
        {
          "config": {"x1": 0.5, "x2": 0.5},
          "rationale": "line one
line two"
        }
      ]
    }
    '''

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert candidates[0].config == {"x1": 0.5, "x2": 0.5}
    assert "line two" in candidates[0].metadata["rationale"]


def test_parse_agent_candidate_payload_skips_invalid_candidates() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {"config": {"x1": 0.5, "x2": 0.5}},
                {"config": {"x1": 11.0, "x2": 0.5}},
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}]


def test_parse_agent_candidate_payload_accepts_compact_xy_arrays_for_bboplace() -> None:
    task = create_task("bboplace_bench", max_evaluations=1, seed=7)
    xs = [float(index) for index in range(32)]
    ys = [float(32 + index) for index in range(32)]
    payload = json.dumps(
        {
            "candidates": [
                {
                    "x": xs,
                    "y": ys,
                    "rationale": "compact macro placement coordinates",
                }
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert candidates[0].config["x_0"] == 0.0
    assert candidates[0].config["x_31"] == 31.0
    assert candidates[0].config["y_0"] == 32.0
    assert candidates[0].config["y_31"] == 63.0
    assert candidates[0].metadata["rationale"] == "compact macro placement coordinates"


def test_parse_agent_candidate_payload_trims_extra_compact_xy_values_for_bboplace() -> None:
    task = create_task("bboplace_bench", max_evaluations=1, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {
                    "x": [-5.0, *[float(index) for index in range(1, 35)]],
                    "y": [999.0, *[float(100 + index) for index in range(1, 34)]],
                    "rationale": "local model emitted extra coordinates",
                }
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert candidates[0].config["x_0"] == 0.0
    assert candidates[0].config["x_31"] == 31.0
    assert candidates[0].config["y_0"] == 224.0
    assert candidates[0].config["y_31"] == 131.0
    assert "x_32" not in candidates[0].config


def test_parse_agent_candidate_payload_pads_nearly_complete_compact_xy_values_for_bboplace() -> None:
    task = create_task("bboplace_bench", max_evaluations=1, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {
                    "x": [float(index) for index in range(31)],
                    "y": [float(100 + index) for index in range(32)],
                    "rationale": "local model omitted one x coordinate",
                }
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert candidates[0].config["x_30"] == 30.0
    assert candidates[0].config["x_31"] == task.spec.search_space["x_31"].effective_default()
    assert candidates[0].config["y_31"] == 131.0


def test_parse_agent_candidate_payload_accepts_markdown_wrapped_json() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)

    candidates = parse_agent_candidate_payload(
        '```json\n{"candidates": [{"config": {"x1": 0.5, "x2": 0.5}}]}\n```',
        task.spec.search_space,
    )

    assert candidates[0].config == {"x1": 0.5, "x2": 0.5}


def test_parse_agent_candidate_payload_accepts_bbo_text_tool_call_candidate() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    raw = """
    I will validate this candidate.
    <tool_call>
    <function=BBO>
    <parameter=action>
    validate_candidate
    </parameter>
    <parameter=candidate>
    {"x1": 0.5, "x2": 0.5}
    </parameter>
    </function>
    </tool_call>
    """

    candidates = parse_agent_candidate_payload(raw, task.spec.search_space)

    assert candidates[0].config == {"x1": 0.5, "x2": 0.5}


def test_parse_agent_candidate_payload_accepts_inline_bbo_validate_call() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    raw = """
    <tool_call>
    <function=BBO().validate_candidate({"x1": 5.0, "x2": -5.0})
    </parameter>
    </function>
    </tool_call>
    """

    candidates = parse_agent_candidate_payload(raw, task.spec.search_space)

    assert candidates[0].config == {"x1": 5.0, "x2": -5.0}


def test_parse_agent_candidate_payload_accepts_single_parameter_text_tool_call_value() -> None:
    search_space = SearchSpace([StringParam(name="smiles", min_length=1, max_length=512)])
    raw = """
    <tool_call>
    <function=BBO>
    <parameter=validate_candidate>
    CC1(C)C(=O)C2CC1CC2
    </parameter>
    </function>
    </tool_call>
    """

    candidates = parse_agent_candidate_payload(raw, search_space)

    assert candidates[0].config == {"smiles": "CC1(C)C(=O)C2CC1CC2"}


def test_parse_agent_candidate_payload_accepts_quoted_key_value_fragment() -> None:
    search_space = SearchSpace([StringParam(name="smiles", min_length=1, max_length=512)])
    raw = 'BBO().validate_candidate("smiles": "CC1(C)C(=O)C2CC1CC2")'

    candidates = parse_agent_candidate_payload(raw, search_space)

    assert candidates[0].config == {"smiles": "CC1(C)C(=O)C2CC1CC2"}


def test_parse_agent_candidate_payload_accepts_key_value_argument_pair() -> None:
    search_space = SearchSpace([StringParam(name="smiles", min_length=1, max_length=512)])
    raw = 'BBO().validate_candidate("smiles", "CC1(C)C(=O)C2CC1CC2")'

    candidates = parse_agent_candidate_payload(raw, search_space)

    assert candidates[0].config == {"smiles": "CC1(C)C(=O)C2CC1CC2"}


def test_parse_agent_candidate_payload_accepts_named_numeric_prose_candidate() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    raw = """
    The incumbent is already at the origin, so I will propose a different
    candidate for exploration.

    Candidate:
    - x1: 3.0
    - x2: -3.0
    """

    candidates = parse_agent_candidate_payload(raw, task.spec.search_space)

    assert candidates[0].config == {"x1": 3.0, "x2": -3.0}


def test_parse_agent_candidate_payload_rejects_invalid_configs() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)

    with pytest.raises(GeneralAgentValidationError, match="expects"):
        parse_agent_candidate_payload(
            json.dumps({"candidates": [{"config": {"x1": 11.0, "x2": 0.5}}]}),
            task.spec.search_space,
        )


def test_nanobot_bbo_algorithm_with_mock_engine_writes_artifacts(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=3)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=11),
        run_dir=tmp_path / "agent_run",
        timeout_seconds=5.0,
        candidates_per_call=3,
    )
    logger = JsonlMetricLogger(tmp_path / "trials.jsonl")

    summary = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(seed=3, resume=False, fail_fast_on_sanity=True),
    ).run()

    assert summary.n_completed == 4
    assert summary.best_primary_objective is not None
    artifacts = algorithm.artifact_paths
    assert Path(artifacts["agent_workspace"]).exists()
    assert Path(artifacts["agent_calls_jsonl"]).exists()
    assert Path(artifacts["llm_logs_dir"]).exists()
    assert Path(artifacts["agent_llm_logs_dir"]) == Path(artifacts["llm_logs_dir"])
    assert Path(artifacts["agent_state_json"]).exists()
    assert Path(artifacts["agent_manifest_json"]).exists()
    assert Path(artifacts["agent_tool_specs_json"]).exists()
    assert Path(artifacts["agent_workspace_tool_py"]).exists()
    assert Path(artifacts["agent_workspace_bbo_tools_py"]).exists()
    assert Path(artifacts["agent_workspace_tool_config_json"]).exists()
    assert Path(artifacts["agent_workspace_gp_example_py"]).exists()
    assert Path(artifacts["agent_workspace_gp_entrypoint_py"]).exists()
    assert Path(artifacts["agent_workspace_python_environment_md"]).exists()
    assert Path(artifacts["agent_workspace_tools_md"]).exists()
    assert Path(artifacts["agent_tool_calls_jsonl"]).exists()
    assert Path(artifacts["agent_memory_jsonl"]).parent.exists()
    assert Path(artifacts["agent_reasoning_traces_dir"]).exists()
    assert Path(artifacts["agent_reasoning_metadata_jsonl"]).exists()
    assert (Path(artifacts["agent_workspace"]) / "instructions.md").exists()
    assert (Path(artifacts["agent_workspace"]) / "TOOLS.md").exists()
    assert (Path(artifacts["agent_workspace"]) / "manifest.json").exists()
    assert (Path(artifacts["agent_workspace"]) / "tool_specs.json").exists()
    assert Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").strip()
    records = logger.load_records()
    assert records[0].suggestion_metadata["agent_framework"] == "nanobot"
    assert records[0].suggestion_metadata["agent_engine"] == "mock"


def test_workspace_manifest_reflects_runtime_tool_policy(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=3)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=11),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        enable_code_interpreter=False,
        web_search_provider="disabled",
    )
    algorithm.setup(task.spec, seed=3, task_description=task.get_description())
    algorithm.ask()

    workspace = Path(algorithm.artifact_paths["agent_workspace"])
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    tool_specs = json.loads((workspace / "tool_specs.json").read_text(encoding="utf-8"))
    spec_names = [tool["function"]["name"] for tool in tool_specs["tools"]]

    assert manifest["tool_policy"]["enabled_tools"] == spec_names
    assert "summarize_objective_metrics" in manifest["tool_policy"]["enabled_tools"]
    assert "code_interpreter" not in manifest["tool_policy"]["enabled_tools"]
    assert "web_search" not in manifest["tool_policy"]["enabled_tools"]
    assert manifest["tool_policy"]["code_interpreter"]["enabled"] is False
    assert manifest["tool_policy"]["web_search"]["enabled"] is False


def test_workspace_tool_bridge_calls_every_advertised_tool(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        code_backend="mock",
        web_search_provider="mock",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.0, "x2": 0.0}, 30.0, search_intent="initialization"))
    algorithm.tell(
        _make_observation(
            1,
            {"x1": 1.0, "x2": 2.0},
            20.0,
            search_intent="hypothesis_test",
            skill="isolate-variable-effect",
            parent_trials=["trial_0"],
            reference_trials=["trial_0"],
            hypothesis="x1 increase improves loss",
        )
    )
    artifacts = algorithm.artifact_paths
    workspace = Path(artifacts["agent_workspace"])
    tool_script = Path(artifacts["agent_workspace_tool_py"])
    assert "from bbo." not in tool_script.read_text(encoding="utf-8")
    assert "from bbo." not in Path(artifacts["agent_workspace_bbo_tools_py"]).read_text(encoding="utf-8")
    calls = {
        "get_task_context": {"max_chars_per_section": 500},
        "get_search_space": {},
        "get_trial_history": {"mode": "all", "limit": 5},
        "get_incumbent": {},
        "get_history_overview": {"recent_limit": 2},
        "compare_trials": {"trial_ids": ["trial_0", "trial_1"]},
        "find_nearest_trials": {"target": "trial_1", "k": 1},
        "estimate_local_effects": {"reference": "trial_1", "variables": ["x1"], "local_radius": 1.0},
        "measure_search_coverage": {"recent_limit": 2},
        "summarize_objective_metrics": {"recent_limit": 2},
        "fit_and_check_surrogate": {"min_observations": 6},
        "score_virtual_candidates": {"model_id": "missing", "candidates": [{"config": {"x1": 0.5, "x2": 0.5}}]},
        "validate_candidate": {"candidate": {"config": {"x1": 0.0, "x2": 0.0}}},
        "validate_candidates": {"candidates": [{"config": {"x1": 0.0, "x2": 0.0}}]},
        "get_recent_search_actions": {"limit": 2},
        "sample_candidates": {"n": 2, "seed": 5},
        "analyze_history": {},
        "memory_write": {"kind": "note", "content": "workspace bridge probe", "tags": ["healthcheck"]},
        "memory_read": {"tags": ["healthcheck"]},
        "code_interpreter": {"code": "print(1)", "language": "python"},
        "web_search": {"query": "branin optimization prior", "limit": 1},
        "fetch_url": {"url": "https://example.com", "max_chars": 200},
    }

    for tool_name, arguments in calls.items():
        completed = subprocess.run(
            [sys.executable, str(tool_script), tool_name, json.dumps(arguments)],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True

    for tool_name in ("get_space", "get_history", "get_objective", "get_tool_specs", "get_manifest"):
        completed = subprocess.run(
            [sys.executable, str(tool_script), tool_name, "{}"],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True

    records = [
        json.loads(line)
        for line in Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {record["tool_name"] for record in records} >= set(calls)
    assert all(record["success"] is True for record in records)


def test_history_analysis_tools_return_structured_evidence(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=10, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.0, "x2": 0.0}, 30.0, search_intent="initialization"))
    algorithm.tell(
        _make_observation(
            1,
            {"x1": 1.0, "x2": 0.0},
            20.0,
            search_intent="hypothesis_test",
            skill="isolate-variable-effect",
            parent_trials=["trial_0"],
            hypothesis="x1 caused the improvement",
        )
    )
    algorithm.tell(_make_observation(2, {"x1": 1.0, "x2": 2.0}, 18.0, search_intent="exploitation", parent_trials=["trial_1"]))
    artifacts = algorithm.artifact_paths

    overview = _call_workspace_tool(artifacts, "get_history_overview", {"recent_limit": 3})
    assert overview["evaluated_count"] == 3
    assert overview["last_best_trial_id"] == 2

    comparison = _call_workspace_tool(artifacts, "compare_trials", {"trial_ids": ["trial_0", "trial_1"]})
    changed = comparison["comparisons"][0]["changed_variables"]
    assert changed == [{"name": "x1", "old": 0.0, "new": 1.0, "normalized_delta": 0.1}]

    nearest = _call_workspace_tool(artifacts, "find_nearest_trials", {"target": "trial_2", "k": 1})
    assert nearest["neighbors"][0]["trial_id"] == 1
    assert nearest["neighbors"][0]["distance"] > 0

    coverage = _call_workspace_tool(artifacts, "measure_search_coverage", {"recent_limit": 3})
    assert coverage["evaluated_count"] == 3
    assert "x1" in coverage["numeric_coverage"]

    metrics = _call_workspace_tool(artifacts, "summarize_objective_metrics", {"recent_limit": 3})
    assert metrics["primary_objective"] == "loss"
    assert metrics["objective"]["best"] == 18.0
    assert metrics["budget_consumed"] is False

    actions = _call_workspace_tool(artifacts, "get_recent_search_actions", {"limit": 3})
    assert actions["actions"][1]["skill"] == "isolate-variable-effect"
    assert actions["actions"][1]["search_intent"] == "hypothesis_test"

    regions = _call_workspace_tool(
        artifacts,
        "recommend_search_regions",
        {"limit": 10, "bins": 3, "mode": "auto"},
    )
    assert regions["recommended_mode"] in {"exploit", "explore", "balanced"}
    assert 1 <= regions["max_actionable_parameters"] <= 2
    assert set(regions["actionable_regions"]) | set(regions["context_only_parameters"]) == {"x1", "x2"}
    assert len(regions["actionable_regions"]) <= 3
    assert regions["budget_consumed"] is False
    assert "never concatenate" in regions["caution"]

    strategy = _call_workspace_tool(artifacts, "analyze_search_strategy", {})
    assert strategy["evidence_sufficient"] is False
    assert strategy["recommended_subspace"]["apply"] is False
    assert strategy["recommended_subspace"]["optimizer_bounds"] == {}
    assert strategy["downstream_policy"]["mode"] == "explore"

    duplicate = _call_workspace_tool(artifacts, "validate_candidate", {"candidate": {"config": {"x1": 1.0, "x2": 2.0}}})
    assert duplicate["valid"] is False
    assert any(item["rule"] == "duplicate" for item in duplicate["violations"])



def test_search_strategy_returns_legal_conservative_optimizer_subspace(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=20, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "strategy_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    points = [
        (-4.5, -4.0), (-3.5, 3.0), (-2.5, -1.0), (-1.5, 2.0),
        (-0.8, -0.5), (-0.4, 0.2), (0.0, 0.0), (0.4, -0.3),
        (1.0, 0.7), (-1.2, 1.0), (-0.6, 0.8), (0.6, -0.8),
        (1.2, 1.1), (2.5, -2.0), (3.5, 2.5), (4.5, -3.5),
    ]
    for trial_id, (x1, x2) in enumerate(points):
        algorithm.tell(_make_observation(trial_id, {"x1": x1, "x2": x2}, x1 * x1 + x2 * x2))

    strategy = _call_workspace_tool(
        algorithm.artifact_paths,
        "analyze_search_strategy",
        {"elite_fraction": 0.3, "min_width_fraction": 0.35},
    )

    assert strategy["evidence_sufficient"] is True
    assert strategy["recommended_subspace"]["sample_supported"] is True
    assert strategy["recommended_subspace"]["apply"] is False
    assert strategy["recommended_subspace"]["cv_supported"] is False
    assert strategy["recommended_subspace"]["optimizer_bounds"] == {}
    bounds = strategy["recommended_subspace"]["candidate_bounds"]
    assert set(bounds) == {"x1", "x2"}
    assert all(-5.0 <= low < high <= 5.0 for low, high in bounds.values())
    assert all(high - low >= 3.5 - 1e-6 for low, high in bounds.values())
    assert strategy["recommended_subspace"]["reversible"] is True
    validation = strategy["landscape"]["model_validation"]
    assert validation["status"] == "ok"
    assert set(validation["models"]) == {"linear", "quadratic", "global_gp"}
    assert all(isinstance(item["cv_r2"], float) for item in validation["models"].values())
    assert validation["local_scope_count"] >= 6
    assert isinstance(validation["global_gp_on_local_r2"], float)
    assert isinstance(validation["local_gp_r2"], float)
    assert isinstance(validation["local_gp_gain_over_global"], float)
    assert strategy["landscape"]["shape_hypothesis"] in {
        "approximately_monotonic_or_unresolved",
        "curved_or_unimodal",
        "coupled_or_multimodal",
    }
    assert strategy["downstream_policy"]["strategy"] in {
        "global_bo", "focused_global_bo", "local_bo"
    }
    assert strategy["downstream_policy"]["acquisition"] in {"logei", "ucb"}
    assert strategy["budget_consumed"] is False

def test_surrogate_rejects_short_history_and_scores_virtual_candidates_without_budget(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=20, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": -5.0, "x2": 0.0}, 5.0))
    algorithm.tell(_make_observation(1, {"x1": -3.0, "x2": 1.0}, 3.0))

    short = _call_workspace_tool(algorithm.artifact_paths, "fit_and_check_surrogate", {"min_observations": 6})
    assert short["usable_signal"] is False
    assert short["selected_model_id"] is None

    for trial_id in range(2, 8):
        x1 = -5.0 + trial_id
        algorithm.tell(_make_observation(trial_id, {"x1": x1, "x2": float(trial_id % 3)}, x1 + 10.0))

    fitted = _call_workspace_tool(algorithm.artifact_paths, "fit_and_check_surrogate", {"min_observations": 6})
    if fitted["usable_signal"]:
        scored = _call_workspace_tool(
            algorithm.artifact_paths,
            "score_virtual_candidates",
            {
                "model_id": fitted["selected_model_id"],
                "candidates": [{"config": {"x1": -4.5, "x2": 2.0}}],
            },
        )
        assert scored["budget_consumed"] is False
        assert scored["scores"][0]["scored"] is True
        assert _call_workspace_tool(algorithm.artifact_paths, "get_history_overview", {})["evaluated_count"] == 8
    else:
        assert "warnings" in fitted


def test_agent_accepts_only_one_candidate_per_real_round(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=7)
    engine = MockAgentEngine(seed=13)
    algorithm = NanobotBBOAlgorithm(
        engine=engine,
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        candidates_per_call=3,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    task.spec.search_space.validate_config(suggestion.config)
    assert suggestion.metadata["search_intent"] == "exploration"
    assert suggestion.metadata["search_action"]["search_intent"] == "exploration"
    assert algorithm._queue == []


def test_duplicate_candidate_rejection_uses_next_valid_candidate(tmp_path: Path) -> None:
    class StaticEngine(MockAgentEngine):
        async def run_agent(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            payload = {
                "candidates": [
                    {"config": {"x1": 0.0, "x2": 0.0}, "search_intent": "exploration"},
                    {"config": {"x1": 1.0, "x2": 1.0}, "search_intent": "hypothesis_test"},
                ]
            }
            from bbo.algorithms.agentic.general_agent_engines import AgentResult

            return AgentResult(status="success", answer=json.dumps(payload))

    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = NanobotBBOAlgorithm(
        engine=StaticEngine(seed=13),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.0, "x2": 0.0}, 30.0))

    suggestion = algorithm.ask()

    assert suggestion.config == {"x1": 1.0, "x2": 1.0}
    assert suggestion.metadata["search_intent"] == "hypothesis_test"


def test_repair_metadata_preserves_original_search_intent() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {
                    "config": {"x1": 0.5, "x2": 0.5},
                    "rationale": "repair after local refinement",
                    "search_intent": "exploitation",
                    "search_action": {
                        "skill": "repair-invalid-candidate",
                        "original_search_intent": "exploitation",
                        "repair": {"field": "x1", "before": 11.0, "after": 0.5},
                    },
                }
            ]
        }
    )

    parsed = parse_agent_candidate_payload(payload, task.spec.search_space)
    from bbo.algorithms.agentic.general_agent import _search_action_metadata

    metadata = _search_action_metadata(parsed[0].metadata, call_id="agent_call_00000", candidate_index=0)
    assert metadata["search_intent"] == "exploitation"
    assert metadata["search_action"]["skill"] == "repair-invalid-candidate"
    assert metadata["search_action"]["original_search_intent"] == "exploitation"


def test_stagnation_and_memory_skills_have_required_guardrails() -> None:
    root = Path(__file__).resolve().parents[1] / "bbo" / "algorithms" / "agentic" / "skills"
    stagnation = (root / "escape-search-stagnation" / "SKILL.md").read_text(encoding="utf-8")
    memory = (root / "distill-search-memory" / "SKILL.md").read_text(encoding="utf-8")

    assert "History is still short" in stagnation
    assert "## Counterexample" in stagnation
    assert "Do not delete key trial references" in memory
    assert "does not directly propose evaluator candidates" in memory


def test_workspace_tool_bridge_supports_serpapi_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    WorkspaceSerpApiMockHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceSerpApiMockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
        monkeypatch.setenv("SERPAPI_ENDPOINT", f"http://127.0.0.1:{server.server_port}/search.json")
        task = create_agent_test_task(max_evaluations=4, seed=5)
        algorithm = NanobotBBOAlgorithm(
            engine=MockAgentEngine(seed=23),
            run_dir=tmp_path / "agent_run",
            tool_mode="workspace_json",
            web_search_provider="serpapi",
        )
        algorithm.setup(task.spec, seed=5, task_description=task.get_description())
        artifacts = algorithm.artifact_paths
        workspace = Path(artifacts["agent_workspace"])
        completed = subprocess.run(
            [
                sys.executable,
                str(artifacts["agent_workspace_tool_py"]),
                "web_search",
                json.dumps({"query": "branin optimization prior", "limit": 2}),
            ],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["result"]["results"][0]["title"] == "Workspace SERP prior"
    assert WorkspaceSerpApiMockHandler.requests == [
        {
            "path": "/search.json",
            "params": {
                "engine": "google",
                "q": "branin optimization prior",
                "api_key": "test-key",
                "num": "2",
            },
        }
    ]
    source_records = Path(artifacts["agent_sources_jsonl"]).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(source_records[0])["kind"] == "search_result"


def test_workspace_python_api_and_gp_example_run_in_workspace(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        code_backend="mock",
        web_search_provider="mock",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    algorithm.tell(_make_observation(0, {"x1": 0.0, "x2": 0.0}, 30.0, search_intent="initialization"))
    algorithm.tell(_make_observation(1, {"x1": 1.0, "x2": 2.0}, 20.0, search_intent="exploitation", parent_trials=["trial_0"]))
    artifacts = algorithm.artifact_paths
    workspace = Path(artifacts["agent_workspace"])

    api_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from bbo_tools import BBO; "
                "bbo = BBO(); "
                "payload = {"
                "'dimension': bbo.search_space()['dimension'], "
                "'history_total': bbo.history(mode='all')['total'], "
                "'overview_count': bbo.history_overview()['evaluated_count'], "
                "'metric_count': bbo.summarize_objective_metrics()['success_count'], "
                "'nearest': len(bbo.find_nearest_trials('trial_1', k=1)['neighbors']), "
                "'coverage_count': bbo.measure_search_coverage()['evaluated_count'], "
                "'recent_actions': bbo.recent_search_actions(limit=2)['count'], "
                "'single_valid': bbo.validate_candidate({'config': {'x1': 0.5, 'x2': 0.5}})['valid'], "
                "'valid_count': bbo.validate([{'config': {'x1': 0.0, 'x2': 0.0}}])['valid_count'], "
                "'code_backend': bbo.code_interpreter('print(1)')['backend'], "
                "'search_enabled': bbo.web_search('branin optimization', limit=1)['enabled']"
                "}; "
                "print(json.dumps(payload, sort_keys=True))"
            ),
        ],
        cwd=workspace,
        check=False,
        text=True,
        capture_output=True,
    )
    assert api_probe.returncode == 0, api_probe.stderr
    api_payload = json.loads(api_probe.stdout)
    assert api_payload["dimension"] == 2
    assert api_payload["history_total"] == 2
    assert api_payload["overview_count"] == 2
    assert api_payload["metric_count"] == 2
    assert api_payload["nearest"] == 1
    assert api_payload["coverage_count"] == 2
    assert api_payload["recent_actions"] == 2
    assert api_payload["single_valid"] is True
    assert api_payload["valid_count"] == 1
    assert api_payload["code_backend"] == "mock"
    assert api_payload["search_enabled"] is True

    gp_probe = subprocess.run(
        [sys.executable, "gp_expected_improvement.py"],
        cwd=workspace,
        check=False,
        text=True,
        capture_output=True,
    )
    assert gp_probe.returncode == 0, gp_probe.stderr
    gp_payload = json.loads(gp_probe.stdout)
    assert set(gp_payload) == {"candidates"}
    assert 1 <= len(gp_payload["candidates"]) <= 4
    for item in gp_payload["candidates"]:
        task.spec.search_space.validate_config(item["config"])

    records = [
        json.loads(line)
        for line in Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record["interface"] == "workspace_python_api" for record in records)
    assert any(record["tool_name"] == "validate_candidates" for record in records)


def test_general_agent_visible_cot_requirement_uses_reasoning_metadata(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=4, seed=8)
    missing = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=31),
        run_dir=tmp_path / "missing_reasoning",
        tool_mode="workspace_json",
        require_visible_cot=True,
        max_retries=0,
    )
    missing.setup(task.spec, seed=8, task_description=task.get_description())
    with pytest.raises(RuntimeError, match="visible CoT"):
        missing.ask()

    present = NanobotBBOAlgorithm(
        engine=ReasoningMetadataMockEngine(seed=32),
        run_dir=tmp_path / "present_reasoning",
        tool_mode="workspace_json",
        require_visible_cot=True,
        max_retries=0,
    )
    present.setup(task.spec, seed=8, task_description=task.get_description())
    suggestion = present.ask()
    task.spec.search_space.validate_config(suggestion.config)
    metadata_records = [
        json.loads(line)
        for line in Path(present.artifact_paths["agent_reasoning_metadata_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metadata_records
    assert metadata_records[-1]["reasoning_visible"] is True
    call_records = [
        json.loads(line)
        for line in Path(present.artifact_paths["agent_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert call_records[-1]["reasoning"]["reasoning_visible"] is True


def test_nanobot_engine_uses_agent_call_id_as_cli_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bbo.algorithms.agentic import general_agent_engines

    captured: dict[str, object] = {}
    raw_answer = '{"candidates": [{"config": {"x1": 0.25, "x2": 0.75}}]}'
    log_dir = tmp_path / "llm_logs"
    session_dir = log_dir / "agent_call_00007"
    session_dir.mkdir(parents=True)
    (session_dir / "2026-06-18T00-00-00-000Z_agent-end.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": raw_answer},
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"candidates": [{"config": {"x1": 0\\n.0, "x2": 0.0}}]}', b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(general_agent_engines.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    work_copy = AgentWorkCopy(
        state_dir=tmp_path / "state",
        config_path=tmp_path / "config.json",
        project_root=tmp_path,
        workspace_root=tmp_path,
        extra={"log_dir": log_dir},
    )

    result = asyncio.run(
        NanobotEngine().run_agent(
            "",
            "prompt",
            work_copy,
            extra_env={"BBO_AGENT_CALL_ID": "agent_call_00007"},
        )
    )

    assert result.status == "success"
    assert result.answer == raw_answer
    cmd = list(captured["cmd"])
    session_index = cmd.index("-s")
    assert cmd[session_index + 1] == "agent_call_00007"


def test_nanobot_engine_enforces_workspace_tool_call_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bbo.algorithms.agentic import general_agent_engines

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool_calls_path = tmp_path / "agent_tool_calls.jsonl"
    (workspace / "bbo_tool_config.json").write_text(
        json.dumps({"tool_calls_path": str(tool_calls_path)}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None

        def __init__(self) -> None:
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            with tool_calls_path.open("a", encoding="utf-8") as handle:
                for index in range(3):
                    handle.write(json.dumps({"tool_name": "sample_candidates", "index": index}) + "\n")
            while not self.killed:
                await asyncio.sleep(0.01)
            self.returncode = -9
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        del cmd, kwargs
        proc = FakeProcess()
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(general_agent_engines.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    work_copy = AgentWorkCopy(
        state_dir=tmp_path / "state",
        config_path=None,
        project_root=workspace,
        workspace_root=workspace,
    )

    result = asyncio.run(NanobotEngine().run_agent("", "prompt", work_copy, max_tool_calls=2, timeout=2))

    assert result.status == "failed"
    assert result.returncode == -9
    assert result.error == "Exceeded max BBO workspace tool calls (2) in one agent invocation."
    assert captured["proc"].killed is True


def test_nanobot_engine_timeout_reports_retry_feedback_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bbo.algorithms.agentic import general_agent_engines

    class FakeProcess:
        returncode: int | None = None

        def __init__(self) -> None:
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            while not self.killed:
                await asyncio.sleep(0.01)
            self.returncode = -9
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    captured: dict[str, FakeProcess] = {}

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        del cmd, kwargs
        proc = FakeProcess()
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(general_agent_engines.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    work_copy = AgentWorkCopy(
        state_dir=tmp_path / "state",
        config_path=None,
        project_root=tmp_path,
        workspace_root=tmp_path,
    )

    result = asyncio.run(NanobotEngine().run_agent("", "prompt", work_copy, timeout=0.01))

    assert result.status == "timeout"
    assert result.returncode == -1
    assert "timed out after 0.01s" in (result.error or "")
    assert "thinking/tool use took too long" in (result.error or "")
    assert "required raw JSON object" in (result.error or "")
    assert captured["proc"].killed is True


def test_general_agent_replay_resume_extends_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "agent_run"
    results_path = tmp_path / "trials.jsonl"

    first_task = create_agent_test_task(max_evaluations=2, seed=4)
    first_algorithm = NanobotBBOAlgorithm(engine=MockAgentEngine(seed=17), run_dir=run_dir, resume=False)
    first_logger = JsonlMetricLogger(results_path)
    Experimenter(
        task=first_task,
        algorithm=first_algorithm,
        logger_backend=first_logger,
        config=ExperimentConfig(seed=4, resume=False, fail_fast_on_sanity=True),
    ).run()

    second_task = create_agent_test_task(max_evaluations=3, seed=4)
    second_algorithm = NanobotBBOAlgorithm(engine=MockAgentEngine(seed=17), run_dir=run_dir, resume=True)
    second_logger = JsonlMetricLogger(results_path)
    summary = Experimenter(
        task=second_task,
        algorithm=second_algorithm,
        logger_backend=second_logger,
        config=ExperimentConfig(seed=4, resume=True, fail_fast_on_sanity=True),
    ).run()

    assert summary.n_completed == 3
    assert len(second_logger.load_records()) == 3
    state = json.loads((run_dir / "agent_state.json").read_text(encoding="utf-8"))
    assert state["history_size"] == 3
