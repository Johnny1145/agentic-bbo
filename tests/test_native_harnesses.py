from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from bbo.algorithms.agentic import (
    AgentWorkCopy,
    ClaudeCodeBBOAlgorithm,
    ClaudeCodeEngine,
    CodexBBOAlgorithm,
    CodexEngine,
    MockAgentEngine,
)
from bbo.algorithms.agentic.claude_messages_compat import (
    anthropic_to_sglang_chat_request,
    openai_to_anthropic_sse,
)
from bbo.algorithms.agentic.codex_responses_compat import (
    chat_to_responses_sse,
    responses_to_sglang_chat_request,
)
from bbo.benchmark.nanobot import build_arg_parser, collect_tool_usage, ensure_run_setting_manifest
from bbo.tasks import create_task


def _sse_event(event_type: str, data: dict[str, object]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _sse_json(event: bytes) -> dict[str, object]:
    data_line = next(line for line in event.decode().splitlines() if line.startswith("data:"))
    return json.loads(data_line[5:].strip())


def test_sglang_messages_compat_splits_text_after_tool_use() -> None:
    events = [
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "Read",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": "done"},
                        "finish_reason": None,
                    }
                ]
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
        ).encode()
        + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    translated = [
        _sse_json(event)
        for event in openai_to_anthropic_sse(events, model="qwen3.5-9b")
    ]
    normalized = [
        event
        for event in translated
        if event["type"].startswith("content_block")
    ]

    assert [event["type"] for event in normalized] == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert normalized[0]["index"] == 0
    assert normalized[1]["index"] == 0
    assert normalized[3]["content_block"] == {"type": "text", "text": ""}
    assert normalized[3]["index"] == 1
    assert normalized[4]["index"] == 1
    assert normalized[5]["index"] == 1
    assert translated[-2]["usage"] == {"output_tokens": 4}
    assert translated[-1] == {"type": "message_stop"}


def test_sglang_messages_compat_translates_tools_and_disables_thinking() -> None:
    request = {
        "model": "qwen3.5-9b",
        "max_tokens": 1024,
        "stream": True,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "read files"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Read",
                        "input": {"file_path": "task.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "task",
                    }
                ],
            },
        ],
    }

    prepared = json.loads(
        anthropic_to_sglang_chat_request(
            json.dumps(request).encode(),
            max_output_tokens=512,
        )
    )

    assert prepared["messages"][0] == {"role": "user", "content": "read files"}
    assert prepared["messages"][1]["tool_calls"][0]["function"]["name"] == "Read"
    assert prepared["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "task",
    }
    assert prepared["separate_reasoning"] is False
    assert prepared["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }
    assert prepared["max_tokens"] == 512
    assert prepared["stream_options"] == {"include_usage": True}


def test_sglang_responses_compat_translates_native_tools_and_history() -> None:
    request = {
        "model": "qwen3.5-9b",
        "instructions": "Use native tools.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "read space"}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"sed -n 1,20p space.json"}',
                "call_id": "call-1",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"parameters":[]}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command.",
                "parameters": {"type": "object"},
            },
            {
                "type": "namespace",
                "name": "native",
                "tools": [
                    {
                        "type": "function",
                        "name": "inspect",
                        "description": "Inspect.",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ],
        "stream": True,
        "parallel_tool_calls": False,
    }

    translated = json.loads(
        responses_to_sglang_chat_request(json.dumps(request).encode())
    )

    assert translated["messages"][-2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert translated["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"parameters":[]}',
    }
    assert [tool["function"]["name"] for tool in translated["tools"]] == [
        "exec_command",
        "native.inspect",
    ]
    assert translated["separate_reasoning"] is False
    assert translated["chat_template_kwargs"]["enable_thinking"] is False


def test_sglang_responses_compat_emits_codex_function_call_events() -> None:
    chunks = [
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"pwd"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }
        ).encode()
        + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    request = {
        "model": "qwen3.5-9b",
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }

    events = [
        _sse_json(event)
        for event in chat_to_responses_sse(chunks, request=request)
    ]

    done = next(event for event in events if event["type"] == "response.output_item.done")
    assert done["item"]["type"] == "function_call"
    assert done["item"]["name"] == "exec_command"
    assert done["item"]["arguments"] == '{"cmd":"pwd"}'
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["usage"]["total_tokens"] == 25


def test_codex_no_tool_workspace_uses_isolated_responses_config(tmp_path: Path) -> None:
    task = create_task("bbob_f01_d10", max_evaluations=2, seed=1)
    algorithm = CodexBBOAlgorithm(
        engine=MockAgentEngine(seed=3),
        run_dir=tmp_path / "run",
        model="qwen3.5-9b",
        provider="sglang",
        api_base="http://127.0.0.1:18301/v1",
        api_key_env="LOCAL_LLM_API_KEY",
        tool_mode="no_tool",
        enable_bbo_skills=False,
    )
    algorithm.setup(task.spec, seed=1, task_description=task.get_description())

    workspace = Path(algorithm.artifact_paths["agent_workspace"])
    state_dir = Path(algorithm.artifact_paths["agent_state_dir"])
    config = (state_dir / "config.toml").read_text(encoding="utf-8")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    prompt = algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)

    assert 'wire_api = "responses"' in config
    assert 'base_url = "http://127.0.0.1:18301/v1"' in config
    assert 'env_key = "LOCAL_LLM_API_KEY"' in config
    assert manifest["harness_policy"]["native_tools_preserved"] is True
    assert manifest["harness_policy"]["benchmark_tools_enabled"] is False
    assert manifest["harness_policy"]["benchmark_skills_enabled"] is False
    assert algorithm._work_copy is not None
    assert algorithm._work_copy.extra["codex_config"]["responses_api_compat"] == "sglang"
    assert algorithm._work_copy.extra["codex_config"]["sandbox"] == "danger-full-access"
    assert "native file-reading tools" in prompt
    assert "`task.md`" in prompt
    assert "Search space JSON:" not in prompt
    assert not (workspace / "bbo_tool.py").exists()
    assert not (workspace / "bbo_tools.py").exists()
    assert not (workspace / "skills").exists()


def test_claude_no_tool_workspace_isolates_settings_and_keeps_native_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "EMPTY")
    task = create_task("bbob_f01_d10", max_evaluations=2, seed=1)
    algorithm = ClaudeCodeBBOAlgorithm(
        engine=MockAgentEngine(seed=3),
        run_dir=tmp_path / "run",
        model="qwen3.5-9b",
        provider="sglang",
        api_base="http://127.0.0.1:18301/v1",
        api_key_env="LOCAL_LLM_API_KEY",
        tool_mode="no_tool",
        enable_bbo_skills=False,
    )
    algorithm.setup(task.spec, seed=1, task_description=task.get_description())

    workspace = Path(algorithm.artifact_paths["agent_workspace"])
    work_copy = algorithm._work_copy
    assert work_copy is not None
    config = work_copy.extra["claude_config"]
    prompt = algorithm._build_agent_prompt(call_id="agent_call_00000", attempt_index=0)
    assert config["tools"] == {"type": "preset", "preset": "claude_code"}
    assert config["setting_sources"] == []
    assert config["skills"] == []
    assert config["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18301"
    assert config["env"]["ANTHROPIC_AUTH_TOKEN"] == "EMPTY"
    assert config["env"]["ANTHROPIC_API_KEY"] == ""
    assert config["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert config["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.5-9b"
    assert config["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "qwen3.5-9b"
    assert config["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "qwen3.5-9b"
    assert config["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "qwen3.5-9b"
    assert config["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen3.5-9b"
    assert config["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert config["messages_api_compat"] == "sglang"
    assert config["max_output_tokens"] == 4096
    assert "native file-reading tools" in prompt
    assert "`history.jsonl`" in prompt
    assert (work_copy.state_dir / "settings.json").read_text(encoding="utf-8") == "{}"
    assert not (workspace / "bbo_tool.py").exists()
    assert not (workspace / "skills").exists()


def test_codex_engine_invokes_isolated_cli_and_parses_native_tool_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bbo.algorithms.agentic import general_agent_engines

    executable = tmp_path / "codex"
    executable.touch()
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    workspace.mkdir()
    state.mkdir()
    captured: dict[str, object] = {}
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {"id": "cmd-1", "type": "command_execution", "command": "ls", "status": "completed"},
        },
        {
            "type": "item.completed",
            "item": {"id": "msg-1", "type": "agent_message", "text": '{"candidates": []}'},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}},
    ]

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return ("\n".join(json.dumps(event) for event in events).encode(), b"")

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(general_agent_engines.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    result = asyncio.run(
        CodexEngine().run_agent(
            "",
            "return JSON",
            AgentWorkCopy(
                state_dir=state,
                config_path=state / "config.toml",
                project_root=workspace,
                workspace_root=workspace,
                extra={"codex_config": {"executable": str(executable)}},
            ),
            timeout=1,
        )
    )

    assert result.status == "success"
    assert result.answer == '{"candidates": []}'
    assert result.llm_log is not None
    assert result.llm_log["sessionId"] == "thread-1"
    assert result.llm_log["nativeToolCalls"] == [
        {"id": "cmd-1", "type": "command_execution", "command": "ls", "status": "completed"}
    ]
    cmd = list(captured["cmd"])
    assert cmd[0] == str(executable)
    assert ["--strict-config", "-C", str(workspace)] == cmd[1:4]
    assert "exec" in cmd
    assert "--json" in cmd
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cwd"] == str(workspace)
    assert kwargs["env"]["CODEX_HOME"] == str(state)
    assert kwargs["start_new_session"] is True


def test_claude_engine_uses_native_preset_without_user_settings_or_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    module = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)
            captured["options"] = kwargs
            self.resume = None

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class ToolUseBlock:
        def __init__(self, name: str) -> None:
            self.id = "tool-1"
            self.name = name
            self.input = {"path": "task.md"}

    class ToolResultBlock:
        pass

    class ThinkingBlock:
        pass

    class AssistantMessage:
        def __init__(self) -> None:
            self.content = [ToolUseBlock("Read"), TextBlock('{"candidates": []}')]
            self.model = "qwen3.5-9b"
            self.parent_tool_use_id = None
            self.error = None

    class UserMessage:
        pass

    class ResultMessage:
        def __init__(self) -> None:
            self.is_error = False
            self.result = ""
            self.subtype = "success"
            self.session_id = "session-1"
            self.duration_ms = 20
            self.num_turns = 1
            self.usage = {"input_tokens": 8}
            self.total_cost_usd = 0
            self.model_usage = {}

    class CLINotFoundError(Exception):
        pass

    class ProcessError(Exception):
        pass

    async def query(*, prompt: str, options: object):
        captured["prompt"] = prompt
        captured["query_options"] = options
        yield AssistantMessage()
        yield ResultMessage()

    for name, value in locals().copy().items():
        if name in {
            "ClaudeAgentOptions",
            "TextBlock",
            "ToolUseBlock",
            "ToolResultBlock",
            "ThinkingBlock",
            "AssistantMessage",
            "UserMessage",
            "ResultMessage",
            "CLINotFoundError",
            "ProcessError",
            "query",
        }:
            setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    result = asyncio.run(
        ClaudeCodeEngine().run_agent(
            "",
            "return JSON",
            AgentWorkCopy(
                state_dir=tmp_path / "state",
                config_path=None,
                project_root=tmp_path,
                workspace_root=tmp_path,
                extra={
                    "claude_config": {
                        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:18301"},
                        "model": "qwen3.5-9b",
                        "tools": {"type": "preset", "preset": "claude_code"},
                        "system_prompt": {"type": "preset", "preset": "claude_code"},
                        "setting_sources": [],
                        "skills": [],
                    }
                },
            ),
            max_tool_calls=7,
        )
    )

    assert result.status == "success"
    assert result.answer == '{"candidates": []}'
    options = captured["options"]
    assert isinstance(options, dict)
    assert options["tools"] == {"type": "preset", "preset": "claude_code"}
    assert options["setting_sources"] == []
    assert options["skills"] == []
    assert options["strict_mcp_config"] is True
    assert options["mcp_servers"] == {}
    assert options["plugins"] == []
    assert options["max_turns"] == 7
    assert result.llm_log is not None
    assert result.llm_log["nativeToolCalls"][0]["name"] == "Read"


def test_tool_usage_combines_structured_native_calls_without_bbo_injection(tmp_path: Path) -> None:
    (tmp_path / "agent_calls.jsonl").write_text(
        json.dumps(
            {
                "llm_log": {
                    "nativeToolCalls": [
                        {"type": "command_execution", "command": "ls"},
                        {"type": "toolCall", "name": "Read"},
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    usage = collect_tool_usage(tmp_path)

    assert usage["benchmark_injected_tool_calls"] == 0
    assert usage["native_harness_tool_calls"] == 2
    assert usage["native_harness_tool_counts"] == {"Read": 1, "command_execution": 1}


def test_tool_usage_counts_regional_analysis_as_numeric_evidence(tmp_path: Path) -> None:
    records = [
        {"tool_name": "recommend_search_regions", "success": True},
        {"tool_name": "locate_promising_regions", "success": True},
        {"tool_name": "locate_underexplored_regions", "success": True},
        {"tool_name": "validate_candidate", "success": True},
    ]
    (tmp_path / "agent_tool_calls.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    usage = collect_tool_usage(tmp_path)

    assert usage["bbo_workspace_tool_calls"] == 4
    assert usage["bbo_workspace_non_validation_numeric_tool_calls"] == 3


def test_run_setting_manifest_rejects_incompatible_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ensure_run_setting_manifest(run_dir=run_dir, manifest={"model": "a"}, resume=False)
    ensure_run_setting_manifest(run_dir=run_dir, manifest={"model": "a"}, resume=True)
    with pytest.raises(ValueError, match="different run setting"):
        ensure_run_setting_manifest(run_dir=run_dir, manifest={"model": "b"}, resume=True)


def test_default_benchmark_cli_exposes_native_harness_selection() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "run",
            "--harness",
            "codex",
            "--agent-executable",
            "/opt/codex",
            "--dry-run",
        ]
    )
    assert args.harness == "codex"
    assert args.agent_executable == "/opt/codex"
