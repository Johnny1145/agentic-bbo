"""ClawArena-inspired agent engines for general-agent BBO methods."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


BBOToolExecutor = Callable[[str, dict[str, Any], str | None], Awaitable[str]]


@dataclass
class AgentResult:
    """Result of one external agent invocation."""

    status: Literal["success", "failed", "timeout"]
    answer: str
    error: str | None = None
    returncode: int | None = None
    raw: Any = None
    llm_log: dict[str, Any] | None = None


@dataclass
class AgentWorkCopy:
    """Workspace and framework state handed to one agent engine."""

    state_dir: Path
    config_path: Path | None
    project_root: Path
    workspace_root: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class GeneralAgentEngine(ABC):
    """Minimal async agent execution interface borrowed from ClawArena."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name surfaced in logs."""

    @abstractmethod
    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        """Execute a single agent call."""


class NanobotEngine(GeneralAgentEngine):
    """Nanobot engine using the ClawArena compatibility runner."""

    @property
    def name(self) -> str:
        return "nanobot"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del tool_executor
        if tools:
            return AgentResult(
                status="failed",
                answer="",
                error="NanobotEngine does not support injected BBO function-calling tools in this runtime.",
                returncode=-2,
            )
        cfg = work_copy.extra.get("nanobot_config", {})
        workspace_path = _resolve_workspace(work_copy, agent_id or "")
        call_session_id = session_id or (extra_env or {}).get("BBO_AGENT_CALL_ID", "")
        cmd = [sys.executable, "-m", "bbo.algorithms.agentic.nanobot_runner", "agent", "-m", message, "--no-markdown"]
        if call_session_id:
            cmd.extend(["-s", call_session_id])
        if workspace_path:
            cmd.extend(["-w", str(workspace_path)])
        if work_copy.config_path:
            cmd.extend(["-c", str(work_copy.config_path)])

        env = {
            **os.environ,
            **(cfg.get("env") or {}),
            **(extra_env or {}),
            "BBO_NANOBOT_NO_MAX_TOKENS": "1",
        }
        env.setdefault("BBO_NANOBOT_PARSE_TEXT_TOOL_CALLS", "1")
        if max_tool_calls > 0:
            env["BBO_WORKSPACE_MAX_TOOL_CALLS"] = str(max_tool_calls)
        if log_dir := work_copy.extra.get("log_dir"):
            env["BBO_NANOBOT_LOG_DIR"] = str(log_dir)

        tool_calls_path = _workspace_tool_calls_path(workspace_path)
        tool_calls_baseline = _count_nonempty_lines(tool_calls_path) if max_tool_calls > 0 else 0
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=str(workspace_path) if workspace_path else str(work_copy.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        try:
            stdout, stderr = await _await_process_with_tool_limit(
                proc,
                communicate_task,
                timeout=timeout,
                tool_calls_path=tool_calls_path,
                tool_calls_baseline=tool_calls_baseline,
                max_tool_calls=max_tool_calls,
            )
        except WorkspaceToolCallLimitExceeded as exc:
            return AgentResult(
                status="failed",
                answer="",
                error=str(exc),
                returncode=getattr(proc, "returncode", None) or -9,
            )
        except asyncio.TimeoutError:
            _kill_process(proc)
            await communicate_task
            return AgentResult(
                status="timeout",
                answer="",
                error=_agent_timeout_error(timeout),
                returncode=-1,
            )
        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()
        error_text = None
        if proc.returncode != 0:
            error_text = "\n".join(part for part in (stderr_text, stdout_text) if part) or None
        log_answer = None
        if log_dir and call_session_id:
            log_answer = _latest_nanobot_log_answer(Path(log_dir), call_session_id)
        return AgentResult(
            status="success" if proc.returncode == 0 else "failed",
            answer=log_answer or stdout_text,
            error=error_text,
            returncode=proc.returncode,
        )


class CodexEngine(GeneralAgentEngine):
    """Codex CLI engine backed by a per-run isolated ``CODEX_HOME``."""

    @property
    def name(self) -> str:
        return "codex"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del session_id, tool_executor, max_tool_calls
        if tools:
            return AgentResult(
                status="failed",
                answer="",
                error="CodexEngine does not support injected BBO function-calling tools in this runtime.",
                returncode=-2,
            )

        cfg = work_copy.extra.get("codex_config", {})
        executable = str(cfg.get("executable") or os.environ.get("BBO_CODEX_BIN") or "codex")
        resolved_executable = executable if Path(executable).is_file() else shutil.which(executable)
        if not resolved_executable:
            return AgentResult(
                status="failed",
                answer="",
                error=(
                    f"Codex backend could not find `{executable}`. Install the Codex CLI or set "
                    "`--agent-executable`/`BBO_CODEX_BIN`."
                ),
                returncode=127,
            )

        workspace_path = _resolve_workspace(work_copy, agent_id or "") or work_copy.project_root
        cmd = [
            str(resolved_executable),
            "--strict-config",
            "-C",
            str(workspace_path),
            "-s",
            str(cfg.get("sandbox") or "workspace-write"),
            "-a",
            str(cfg.get("approval_policy") or "never"),
        ]
        responses_proxy = None
        if cfg.get("responses_api_compat") == "sglang":
            from .codex_responses_compat import SGLangResponsesCompatibilityProxy

            upstream_base_url = cfg.get("api_base")
            if not upstream_base_url:
                return AgentResult(
                    status="failed",
                    answer="",
                    error="Codex SGLang compatibility mode requires an API base URL.",
                )
            responses_proxy = SGLangResponsesCompatibilityProxy(upstream_base_url)
            responses_proxy.start()
            cmd.extend(
                [
                    "-c",
                    (
                        "model_providers.bbo_sglang.base_url="
                        f"{json.dumps(responses_proxy.base_url)}"
                    ),
                ]
            )
        cmd.extend(
            [
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            message,
            ]
        )
        env = {
            **os.environ,
            **(cfg.get("env") or {}),
            **(extra_env or {}),
            "CODEX_HOME": str(work_copy.state_dir),
            "NO_COLOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                if timeout is None:
                    stdout, stderr = await proc.communicate()
                else:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                _kill_process(proc, process_group=True)
                await proc.communicate()
                return AgentResult(
                    status="timeout",
                    answer="",
                    error=_agent_timeout_error(timeout),
                    returncode=-1,
                )
        finally:
            if responses_proxy is not None:
                await asyncio.to_thread(responses_proxy.close)

        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()
        events, invalid_lines = _parse_codex_jsonl(stdout_text)
        answer = _codex_final_answer(events)
        llm_log = _build_codex_llm_log(
            events=events,
            invalid_lines=invalid_lines,
            stderr=stderr_text,
            agent_id=agent_id,
        )
        if proc.returncode == 0 and answer:
            return AgentResult(
                status="success",
                answer=answer,
                returncode=0,
                raw=events,
                llm_log=llm_log,
            )
        error = _codex_error(events) or stderr_text or (
            "Codex exited successfully but did not emit a final agent message."
            if proc.returncode == 0
            else stdout_text
        )
        return AgentResult(
            status="failed",
            answer=answer,
            error=error,
            returncode=proc.returncode,
            raw=events,
            llm_log=llm_log,
        )


class ClaudeCodeEngine(GeneralAgentEngine):
    """Claude Code engine using ``claude_agent_sdk`` when available."""

    @property
    def name(self) -> str:
        return "claude_code"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del tool_executor
        if tools:
            return AgentResult(
                status="failed",
                answer="",
                error="ClaudeCodeEngine does not support injected BBO function-calling tools in this runtime.",
                returncode=-2,
            )
        try:
            from claude_agent_sdk import (  # type: ignore
                AssistantMessage,
                ClaudeAgentOptions,
                CLINotFoundError,
                ProcessError,
                ResultMessage,
                TextBlock,
                UserMessage,
                query,
            )
        except ImportError as exc:
            return AgentResult(
                status="failed",
                answer="",
                error=(
                    "Claude Code backend requires `claude-agent-sdk`. "
                    "Install the agent dependency before using `claude_code`."
                ),
                raw=exc,
            )

        cc = work_copy.extra.get("claude_config", {})
        workspace_path = _resolve_workspace(work_copy, agent_id or "")
        stderr_lines: list[str] = []
        claude_env = {
            "CLAUDE_CONFIG_DIR": str(work_copy.state_dir),
            **(cc.get("env") or {}),
            **(extra_env or {}),
        }
        messages_proxy = None
        if cc.get("messages_api_compat") == "sglang":
            from .claude_messages_compat import SGLangMessagesCompatibilityProxy

            upstream_base_url = claude_env.get("ANTHROPIC_BASE_URL")
            if not upstream_base_url:
                return AgentResult(
                    status="failed",
                    answer="",
                    error="Claude Code SGLang compatibility mode requires ANTHROPIC_BASE_URL.",
                )
            messages_proxy = SGLangMessagesCompatibilityProxy(
                upstream_base_url,
                max_output_tokens=cc.get("max_output_tokens"),
            )
            messages_proxy.start()
            claude_env["ANTHROPIC_BASE_URL"] = messages_proxy.base_url

        def _collect_stderr(line: str) -> None:
            stderr_lines.append(line)

        opts = ClaudeAgentOptions(
            cwd=str(workspace_path) if workspace_path else str(work_copy.project_root),
            env=claude_env,
            tools=cc.get("tools", {"type": "preset", "preset": "claude_code"}),
            system_prompt=cc.get("system_prompt", {"type": "preset", "preset": "claude_code"}),
            permission_mode=cc.get("permission_mode", "bypassPermissions"),
            allowed_tools=cc.get("allowed_tools", []),
            disallowed_tools=cc.get("disallowed_tools", []),
            model=cc.get("model"),
            max_turns=cc.get("max_turns") or (max_tool_calls if max_tool_calls > 0 else None),
            cli_path=cc.get("executable"),
            setting_sources=cc.get("setting_sources", []),
            skills=cc.get("skills", []),
            strict_mcp_config=True,
            mcp_servers={},
            plugins=[],
            sandbox=cc.get(
                "sandbox",
                {
                    "enabled": bool(shutil.which("bwrap") and shutil.which("socat")),
                    "autoAllowBashIfSandboxed": True,
                    "allowUnsandboxedCommands": False,
                },
            ),
            stderr=_collect_stderr,
        )
        if session_id:
            opts.resume = session_id

        async def _query_once() -> AgentResult:
            answer_parts: list[str] = []
            messages: list[dict[str, Any]] = []
            result_msg: ResultMessage | None = None
            async for msg in query(prompt=message, options=opts):
                if isinstance(msg, AssistantMessage):
                    messages.append(_serialize_assistant_message(msg))
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            answer_parts.append(block.text)
                elif isinstance(msg, UserMessage):
                    messages.append(_serialize_claude_user_message(msg))
                elif isinstance(msg, ResultMessage):
                    result_msg = msg

            answer_text = "\n".join(part for part in answer_parts if part).strip()
            llm_log = _build_claude_llm_log(messages, result_msg, session_id, agent_id)
            if result_msg is not None and result_msg.is_error:
                return AgentResult(
                    status="failed",
                    answer=answer_text,
                    error=str(result_msg.result),
                    raw=result_msg,
                    llm_log=llm_log,
                )
            return AgentResult(status="success", answer=answer_text, raw=result_msg, llm_log=llm_log)

        try:
            if timeout is None:
                return await _query_once()
            return await asyncio.wait_for(_query_once(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return AgentResult(status="timeout", answer="", error=_agent_timeout_error(timeout), returncode=-1)
        except CLINotFoundError as exc:
            return AgentResult(status="failed", answer="", error=str(exc))
        except ProcessError as exc:
            stderr_text = "\n".join(stderr_lines) if stderr_lines else exc.stderr
            return AgentResult(
                status="failed",
                answer="",
                error=f"exit={exc.exit_code}: {stderr_text}",
                returncode=exc.exit_code,
            )
        except Exception as exc:  # pragma: no cover - depends on external agent behavior.
            stderr_text = "\n".join(stderr_lines)
            detail = f"{exc}"
            if stderr_text:
                detail = f"{detail}\nstderr: {stderr_text}"
            return AgentResult(status="failed", answer="", error=detail)
        finally:
            if messages_proxy is not None:
                await asyncio.to_thread(messages_proxy.close)


class OpenAICompatibleToolEngine(GeneralAgentEngine):
    """OpenAI-compatible chat-completions engine with BBO function-calling support."""

    @property
    def name(self) -> str:
        return "openai_compatible"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del session_id, agent_id, extra_env
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency.
            return AgentResult(
                status="failed",
                answer="",
                error="OpenAI-compatible BBO engine requires the optional `pablo` extra (`openai>=1.0`).",
                raw=exc,
            )
        cfg = work_copy.extra.get("openai_compatible_config", {})
        api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return AgentResult(status="failed", answer="", error="OpenAI-compatible BBO engine requires an API key.")
        model = cfg.get("model") or "gpt-4.1-mini"
        client = AsyncOpenAI(api_key=api_key, base_url=cfg.get("api_base"))
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
        tool_calls_used = 0

        async def _query_once() -> AgentResult:
            nonlocal tool_calls_used
            while True:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                try:
                    response = await client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except TypeError:
                    response = await client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                tool_calls = list(getattr(msg, "tool_calls", None) or [])
                content = getattr(msg, "content", None) or ""
                if not tool_calls:
                    return AgentResult(status="success", answer=str(content), raw=response)
                if tool_executor is None:
                    return AgentResult(status="failed", answer=str(content), error="Model requested tools but no BBO tool executor was provided.")
                if tool_calls_used >= max_tool_calls:
                    return AgentResult(status="failed", answer=str(content), error=f"Exceeded max BBO tool calls ({max_tool_calls}).")
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    if tool_calls_used >= max_tool_calls:
                        break
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = await tool_executor(call.function.name, arguments, call.id)
                    tool_calls_used += 1
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        try:
            if timeout is None:
                return await _query_once()
            return await asyncio.wait_for(_query_once(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return AgentResult(status="timeout", answer="", error=_agent_timeout_error(timeout), returncode=-1)
        except Exception as exc:  # pragma: no cover - provider-specific failures.
            return AgentResult(status="failed", answer="", error=str(exc))


class MockAgentEngine(GeneralAgentEngine):
    """Deterministic local agent used by tests and offline examples."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del session_id, message, agent_id, timeout, extra_env, max_tool_calls
        import json

        if tools and tool_executor is not None:
            sample_raw = await tool_executor(
                "sample_candidates",
                {"n": 4, "seed": self.seed + self.calls, "strategy": "random"},
                "mock_sample_candidates",
            )
            self.calls += 1
            try:
                sample_payload = json.loads(sample_raw)
                candidates = [
                    {"config": item["config"], "rationale": "mock tool sample"}
                    for item in sample_payload["result"]["candidates"]
                ]
            except Exception:
                candidates = []
            if candidates:
                return AgentResult(status="success", answer=json.dumps({"candidates": candidates}, sort_keys=True))

        space_path = (work_copy.workspace_root or work_copy.project_root) / "space.json"
        payload = json.loads(space_path.read_text(encoding="utf-8"))
        rng = random.Random(self.seed + self.calls)
        self.calls += 1
        candidates = []
        for _ in range(4):
            config: dict[str, Any] = {}
            for param in payload["parameters"]:
                if param["type"] == "float":
                    config[param["name"]] = rng.uniform(float(param["low"]), float(param["high"]))
                elif param["type"] == "int":
                    config[param["name"]] = rng.randint(int(param["low"]), int(param["high"]))
                elif param["type"] == "categorical":
                    config[param["name"]] = rng.choice(list(param["choices"]))
                else:
                    raise ValueError(f"Unsupported mock parameter type: {param['type']}")
            candidates.append({"config": config, "rationale": "mock deterministic sample"})
        return AgentResult(status="success", answer=json.dumps({"candidates": candidates}, sort_keys=True))


def create_general_agent_engine(framework: str) -> GeneralAgentEngine:
    normalized = normalize_agent_framework(framework)
    if normalized == "nanobot":
        return NanobotEngine()
    if normalized == "codex":
        return CodexEngine()
    if normalized == "claude_code":
        return ClaudeCodeEngine()
    if normalized == "openai_compatible":
        return OpenAICompatibleToolEngine()
    if normalized == "mock":
        return MockAgentEngine()
    raise ValueError(f"Unknown general-agent framework `{framework}`.")


def normalize_agent_framework(framework: str) -> str:
    normalized = framework.strip().lower().replace("-", "_")
    if normalized in {"claude", "claude_code", "claudecode"}:
        return "claude_code"
    if normalized in {"nanobot", "nano_bot"}:
        return "nanobot"
    if normalized in {"codex", "codex_cli", "openai_codex"}:
        return "codex"
    if normalized in {"openai", "openai_compatible", "openai_compat"}:
        return "openai_compatible"
    if normalized == "mock":
        return "mock"
    return normalized


def _resolve_workspace(work_copy: AgentWorkCopy, agent_id: str) -> Path | None:
    if work_copy.workspace_root and agent_id:
        candidate = work_copy.workspace_root / agent_id
        if candidate.exists():
            return candidate
    return work_copy.workspace_root


def _latest_nanobot_log_answer(log_dir: Path, session_id: str) -> str | None:
    session_dir = log_dir / session_id
    if not session_dir.exists():
        return None
    try:
        files = sorted(session_dir.glob("*_agent-end.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            answer = content.strip()
            if answer and answer != "[Assistant reply unavailable due to model error.]":
                return answer
    return None


class WorkspaceToolCallLimitExceeded(RuntimeError):
    """Raised when a workspace-backed agent call exceeds the configured tool limit."""


async def _await_process_with_tool_limit(
    proc: Any,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    *,
    timeout: float | None,
    tool_calls_path: Path | None,
    tool_calls_baseline: int,
    max_tool_calls: int,
) -> tuple[bytes, bytes]:
    started = asyncio.get_running_loop().time()
    while True:
        remaining = None if timeout is None else timeout - (asyncio.get_running_loop().time() - started)
        if remaining is not None and remaining <= 0:
            raise asyncio.TimeoutError
        wait_seconds = 0.25 if remaining is None else min(0.25, remaining)
        done, _ = await asyncio.wait({communicate_task}, timeout=wait_seconds)
        if done:
            return communicate_task.result()
        if max_tool_calls <= 0 or tool_calls_path is None:
            continue
        new_tool_calls = _count_nonempty_lines(tool_calls_path) - tool_calls_baseline
        if new_tool_calls >= max_tool_calls:
            _kill_process(proc)
            await communicate_task
            raise WorkspaceToolCallLimitExceeded(
                f"Exceeded max BBO workspace tool calls ({max_tool_calls}) in one agent invocation."
            )


def _workspace_tool_calls_path(workspace_path: Path | None) -> Path | None:
    if workspace_path is None:
        return None
    config_path = workspace_path / "bbo_tool_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_path = config.get("tool_calls_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = workspace_path / path
    return path


def _count_nonempty_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _agent_timeout_error(timeout: float | None) -> str:
    return (
        f"Agent invocation timed out after {timeout}s because thinking/tool use took too long. "
        "Retry with concise reasoning and return exactly the required raw JSON object."
    )


def _kill_process(proc: Any, *, process_group: bool = False) -> None:
    try:
        if getattr(proc, "returncode", None) is None:
            if process_group and getattr(proc, "pid", None):
                try:
                    os.killpg(int(proc.pid), signal.SIGKILL)
                    return
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            proc.kill()
    except ProcessLookupError:
        pass


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [_serialize_claude_block(block) for block in msg.content],
        **({"model": msg.model} if getattr(msg, "model", None) else {}),
        **({"usage": msg.usage} if getattr(msg, "usage", None) else {}),
        **({"stopReason": msg.stop_reason} if getattr(msg, "stop_reason", None) else {}),
    }


def _serialize_claude_user_message(msg: Any) -> dict[str, Any]:
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        serialized = [_serialize_claude_block(block) for block in content]
    else:
        serialized = content
    return {
        "role": "user",
        "content": serialized,
        **({"uuid": msg.uuid} if getattr(msg, "uuid", None) else {}),
    }


def _serialize_claude_block(block: Any) -> dict[str, Any]:
    try:
        from claude_agent_sdk import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock  # type: ignore
    except ImportError:  # pragma: no cover - guarded by caller imports.
        return {"type": "unknown", "data": str(block)}
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "toolResult",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    return {"type": "unknown", "data": str(block)}


def _build_claude_llm_log(
    messages: list[dict[str, Any]],
    result_msg: Any,
    session_id: str | None,
    agent_id: str | None,
) -> dict[str, Any]:
    log: dict[str, Any] = {
        "agentId": agent_id or "",
        "sessionId": session_id or getattr(result_msg, "session_id", "") or "",
        "success": bool(result_msg and getattr(result_msg, "subtype", "") == "success"),
        "messageCount": len(messages),
        "messages": messages,
    }
    if result_msg is not None:
        for source, target in (
            ("duration_ms", "durationMs"),
            ("num_turns", "numTurns"),
            ("usage", "usage"),
            ("total_cost_usd", "totalCostUsd"),
            ("model_usage", "modelUsage"),
        ):
            value = getattr(result_msg, source, None)
            if value is not None:
                log[target] = value
    log["nativeToolCalls"] = [
        block
        for message in messages
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "toolCall"
    ]
    return log


def _parse_codex_jsonl(stdout_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    invalid_lines: list[str] = []
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            invalid_lines.append(stripped)
            continue
        if isinstance(payload, dict):
            events.append(payload)
        else:
            invalid_lines.append(stripped)
    return events, invalid_lines


def _codex_final_answer(events: list[dict[str, Any]]) -> str:
    answers: list[str] = []
    for event in events:
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            answers.append(item["text"].strip())
    return next((answer for answer in reversed(answers) if answer), "")


def _codex_error(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") not in {"error", "turn.failed"}:
            continue
        error = event.get("error")
        if isinstance(error, dict):
            for key in ("message", "detail", "code"):
                if error.get(key):
                    return str(error[key])
        if error:
            return str(error)
        if event.get("message"):
            return str(event["message"])
    return None


def _build_codex_llm_log(
    *,
    events: list[dict[str, Any]],
    invalid_lines: list[str],
    stderr: str,
    agent_id: str | None,
) -> dict[str, Any]:
    thread_id = ""
    usage: dict[str, Any] | None = None
    native_tool_calls: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "thread.started" and event.get("thread_id"):
            thread_id = str(event["thread_id"])
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "web_search",
            "browser_use",
            "computer_use",
            "collab_agent_tool_call",
        }:
            native_tool_calls.append(
                {
                    key: item[key]
                    for key in ("id", "type", "command", "status", "name", "server", "query")
                    if key in item
                }
            )
    return {
        "agentId": agent_id or "",
        "sessionId": thread_id,
        "success": any(event.get("type") == "turn.completed" for event in events),
        "usage": usage or {},
        "nativeToolCalls": native_tool_calls,
        "events": events,
        "invalidStdoutLines": invalid_lines,
        "stderr": stderr,
    }


__all__ = [
    "AgentResult",
    "AgentWorkCopy",
    "ClaudeCodeEngine",
    "CodexEngine",
    "GeneralAgentEngine",
    "MockAgentEngine",
    "NanobotEngine",
    "OpenAICompatibleToolEngine",
    "create_general_agent_engine",
    "normalize_agent_framework",
]
