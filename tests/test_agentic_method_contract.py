from __future__ import annotations

import json
from pathlib import Path

from bbo.algorithms.agentic import (
    AGENTIC_METHOD_REGISTRY,
    AgenticBOAlgorithm,
    EventedAlgorithm,
    GeneralAgentBBOAlgorithm,
    MockAgentEngine,
    AgentResult,
    GeneralAgentEngine,
    StatefulOptimizerBackend,
    create_agentic_method,
)
from bbo.core import EvaluationResult, TrialObservation, TrialStatus, TrialSuggestion
class ToolUsingMockEngine(GeneralAgentEngine):
    @property
    def name(self) -> str:
        return "tool-using-mock"

    async def run_agent(
        self, session_id, message, work_copy, *, tools=None, tool_executor=None, **kwargs
    ):
        del session_id, message, work_copy, tools, kwargs
        raw = await tool_executor(
            "optimizer_suggest", {"backend": "gp_ei", "q": 1}, "mock_gp_suggest"
        )
        payload = json.loads(raw)
        candidate = payload["result"]["candidate"]
        return AgentResult(
            status="success",
            answer=json.dumps({"candidates": [{"config": candidate, "rationale": "GP proposal"}]}),
        )

from conftest import create_agent_test_task


def _observe(task, suggestion, trial_id: int) -> TrialObservation:
    committed = TrialSuggestion(
        config=dict(suggestion.config),
        trial_id=trial_id,
        budget=suggestion.budget,
        metadata=dict(suggestion.metadata),
    )
    result = task.evaluate(committed)
    return TrialObservation.from_evaluation(committed, result)


def test_three_methods_have_declarative_capabilities() -> None:
    assert set(AGENTIC_METHOD_REGISTRY) == {"agentic_bo", "llambo", "pablo"}
    assert AGENTIC_METHOD_REGISTRY["agentic_bo"].supports_optimizer_tools
    assert AGENTIC_METHOD_REGISTRY["pablo"].supports_multiple_roles
    assert [role.name for role in AGENTIC_METHOD_REGISTRY["pablo"].roles] == [
        "planner", "explorer", "worker"
    ]
    assert AGENTIC_METHOD_REGISTRY["llambo"].capabilities == {
        "probe",
        "propose",
        "commit",
    }


def test_llambo_and_pablo_use_the_same_event_decorator(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=3)

    for method in ("llambo", "pablo"):
        run_dir = tmp_path / method
        kwargs = {"provider": "mock"} if method == "pablo" else {"backend": "heuristic"}
        algorithm = create_agentic_method(method, run_dir=run_dir, **kwargs)
        assert isinstance(algorithm, EventedAlgorithm)

        algorithm.setup(task.spec, seed=7, task_description=task.get_description())
        suggestion = algorithm.ask()
        task.spec.search_space.validate_config(suggestion.config)
        algorithm.tell(_observe(task, suggestion, 0))

        event_path = Path(algorithm.artifact_paths["deliberation_events_jsonl"])
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        kinds = [event["kind"] for event in events]
        assert kinds[0] == "context"
        assert "propose" in kinds
        assert "commit" in kinds
        assert kinds[-1] == "observation"
        assert all(event["method"] == method for event in events)


def test_agentic_bo_is_a_first_class_method_with_fixed_gp_surface(tmp_path: Path) -> None:
    algorithm = create_agentic_method(
        "agentic_bo",
        run_dir=tmp_path,
        engine=MockAgentEngine(seed=3),
        framework="nanobot",
        tool_mode="workspace_json",
        enable_code_interpreter=False,
        enable_memory=False,
    )
    assert isinstance(algorithm.algorithm, GeneralAgentBBOAlgorithm)
    config = algorithm.algorithm.config
    assert config.algorithm_name == "agentic_bo"
    assert config.optimizer_backend_allowlist == ("gp_ei",)
    assert config.require_optimizer_decision_per_round
    assert "optimizer_suggest" in config.enabled_tool_names
    assert "optimizer_set_bounds" in config.enabled_tool_names
    assert "optimizer_set_acquisition" in config.enabled_tool_names


def test_agentic_optimizer_defaults_to_logei(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=3)
    backend = StatefulOptimizerBackend(
        allowlist=("gp_ei",),
        state_path=tmp_path / "optimizer_state.json",
    )

    status = backend.execute(
        "optimizer_status",
        task_spec=task.spec,
        history=[],
        seed=11,
        incumbent=None,
        arguments={},
    )

    assert status["acquisition"] == {"name": "logei", "parameters": {}}


def test_optimizer_backend_never_consumes_evaluation_budget(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=3)
    backend = StatefulOptimizerBackend(
        allowlist=("random",),
        state_path=tmp_path / "optimizer_state.json",
    )
    payload = backend.execute(
        "optimizer_suggest",
        task_spec=task.spec,
        history=[],
        seed=11,
        incumbent=None,
        arguments={"backend": "random", "q": 1},
    )
    assert payload["evaluator_called"] is False
    assert payload["budget_consumed"] is False
    task.spec.search_space.validate_config(payload["candidate"])
def test_agentic_bo_smoke_emits_tool_and_commit_events(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=2)
    algorithm = create_agentic_method(
        "agentic_bo",
        run_dir=tmp_path,
        engine=ToolUsingMockEngine(),
        framework="nanobot",
        initial_random=1,
        enable_code_interpreter=False,
        enable_memory=False,
        allow_fallback=False,
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    first = algorithm.ask()
    algorithm.tell(_observe(task, first, 0))
    second = algorithm.ask()
    algorithm.tell(_observe(task, second, 1))

    events = [
        json.loads(line)
        for line in Path(algorithm.artifact_paths["deliberation_events_jsonl"]).read_text().splitlines()
    ]
    second_round = [event for event in events if event["evaluation_index"] == 1]
    assert any(event["kind"] == "propose" and event["payload"].get("tool_name") == "optimizer_suggest" for event in second_round)
    assert any(event["kind"] == "commit" for event in second_round)
    assert len([event for event in events if event["kind"] == "observation"]) == 2
