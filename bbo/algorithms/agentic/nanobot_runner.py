"""Nanobot CLI runner with compatibility patches for BBO agent calls."""

from __future__ import annotations

import html
import json
import os
import re
import shlex
import uuid
from typing import Any


_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_TAG_RE = re.compile(r"<function=([^\n>]+)>?(.*?)</function>", re.IGNORECASE | re.DOTALL)
_PARAMETER_TAG_RE = re.compile(r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>", re.IGNORECASE | re.DOTALL)
_BBO_METHOD_START_RE = re.compile(r"BBO(?:\(\))?\.(?P<tool>[A-Za-z_]\w*)\s*\(", re.IGNORECASE)
_BBO_WORKSPACE_TOOLS = frozenset(
    {
        "analyze_history",
        "compare_trials",
        "estimate_local_effects",
        "fetch_url",
        "find_nearest_trials",
        "fit_and_check_surrogate",
        "get_history_overview",
        "get_incumbent",
        "get_manifest",
        "get_objective",
        "get_recent_search_actions",
        "get_search_space",
        "get_task_context",
        "get_tool_specs",
        "get_trial_history",
        "memory_read",
        "memory_write",
        "measure_search_coverage",
        "sample_candidates",
        "score_virtual_candidates",
        "summarize_objective_metrics",
        "validate_candidate",
        "validate_candidates",
        "web_search",
    }
)
_BBO_METHOD_ALIASES = {
    "history": "get_trial_history",
    "history_overview": "get_history_overview",
    "incumbent": "get_incumbent",
    "manifest": "get_manifest",
    "objective": "get_objective",
    "recent_search_actions": "get_recent_search_actions",
    "sample": "sample_candidates",
    "search_space": "get_search_space",
    "task_context": "get_task_context",
    "tool_specs": "get_tool_specs",
    "validate": "validate_candidates",
}


def _patch_strip_max_tokens() -> None:
    """Strip legacy ``max_tokens`` for compatible endpoints that reject it."""

    try:
        from nanobot.providers import openai_compat_provider as provider_module  # type: ignore

        original = provider_module.OpenAICompatProvider._build_kwargs

        def patched(
            self,
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        ):
            kwargs = original(
                self,
                messages,
                tools,
                model,
                max_tokens,
                temperature,
                reasoning_effort,
                tool_choice,
            )
            forced_max_tokens = os.environ.get("BBO_NANOBOT_MAX_TOKENS")
            if forced_max_tokens:
                try:
                    kwargs["max_tokens"] = max(1, int(forced_max_tokens))
                except ValueError:
                    kwargs.pop("max_tokens", None)
            else:
                kwargs.pop("max_tokens", None)
            return kwargs

        provider_module.OpenAICompatProvider._build_kwargs = patched
    except Exception:
        pass


def _patch_shell_env_for_bbo_call_id() -> None:
    """Allow Nanobot shell commands to preserve BBO attempt attribution."""

    try:
        from nanobot.agent.tools.shell import ExecTool  # type: ignore
    except Exception:
        return

    original_build_env = ExecTool._build_env

    def patched_build_env(self):
        env = original_build_env(self)
        agent_call_id = os.environ.get("BBO_AGENT_CALL_ID")
        if agent_call_id:
            env["BBO_AGENT_CALL_ID"] = agent_call_id
        return env

    ExecTool._build_env = patched_build_env


def _decode_text_tool_value(raw: str) -> Any:
    text = html.unescape(raw).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    if not text:
        return ""
    if text[0] in "[{\"-0123456789tfn":
        try:
            return json.loads(text)
        except Exception:
            repaired = _json_repair_loads(text)
            if repaired is not None:
                return repaired
            return text
    return text


def _json_repair_loads(text: str) -> Any | None:
    try:
        from json_repair import loads as repair_loads  # type: ignore
    except Exception:
        return None
    try:
        return repair_loads(text)
    except Exception:
        return None


def _matching_paren_index(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    quote = ""
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _normalise_bbo_tool_name(raw_name: str) -> str | None:
    name = raw_name.strip().replace("-", "_")
    name = re.sub(r"^BBO(?:\(\))?\.", "", name, flags=re.IGNORECASE)
    name = _BBO_METHOD_ALIASES.get(name, name)
    return name if name in _BBO_WORKSPACE_TOOLS else None


def _normalise_bbo_tool_arguments(tool_name: str, raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None or raw_arguments == "":
        return {}
    if tool_name == "validate_candidate":
        if isinstance(raw_arguments, dict) and (
            "candidate" in raw_arguments or "too_similar_threshold" in raw_arguments
        ):
            return dict(raw_arguments)
        return {"candidate": raw_arguments}
    if tool_name == "validate_candidates":
        if isinstance(raw_arguments, dict) and "candidates" in raw_arguments:
            return dict(raw_arguments)
        if isinstance(raw_arguments, list):
            return {"candidates": raw_arguments}
        return {"candidates": [raw_arguments]}
    if tool_name == "compare_trials":
        if isinstance(raw_arguments, dict) and "trial_ids" in raw_arguments:
            return dict(raw_arguments)
        if isinstance(raw_arguments, list):
            return {"trial_ids": raw_arguments}
    if tool_name == "find_nearest_trials":
        if isinstance(raw_arguments, dict) and "target" in raw_arguments:
            return dict(raw_arguments)
        return {"target": raw_arguments}
    if tool_name == "estimate_local_effects":
        if isinstance(raw_arguments, dict) and "reference" in raw_arguments:
            return dict(raw_arguments)
        return {"reference": raw_arguments}
    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    return {"value": raw_arguments}


def _bbo_exec_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    command = f"python bbo_tool.py {shlex.quote(tool_name)} {shlex.quote(raw_arguments)}"
    agent_call_id = os.environ.get("BBO_AGENT_CALL_ID")
    if agent_call_id:
        command = f"BBO_AGENT_CALL_ID={shlex.quote(agent_call_id)} {command}"
    return {"command": command, "timeout": 60}


def _extract_bbo_method_exec_calls(content: str) -> list[tuple[str, dict[str, Any]]]:
    text = html.unescape(content)
    calls: list[tuple[str, dict[str, Any]]] = []
    for match in _BBO_METHOD_START_RE.finditer(text):
        raw_tool = match.group("tool")
        tool_name = _normalise_bbo_tool_name(raw_tool)
        if tool_name is None:
            continue
        end = _matching_paren_index(text, match.end() - 1)
        if end is None:
            continue
        raw_argument_text = text[match.end() : end].strip()
        raw_arguments = _decode_text_tool_value(raw_argument_text)
        calls.append((tool_name, _normalise_bbo_tool_arguments(tool_name, raw_arguments)))
    return calls


def _extract_bbo_function_exec_call(function_name: str, body: str) -> tuple[str, dict[str, Any]] | None:
    raw_function_name = function_name.strip()
    direct_tool = _normalise_bbo_tool_name(raw_function_name)
    if direct_tool is not None:
        raw_arguments = _decode_text_tool_value(body.strip())
        return direct_tool, _normalise_bbo_tool_arguments(direct_tool, raw_arguments)

    if raw_function_name.lower() not in {"bbo", "bbo()"}:
        return None

    params = {
        match.group(1).strip().replace("-", "_"): _decode_text_tool_value(match.group(2))
        for match in _PARAMETER_TAG_RE.finditer(body)
    }
    raw_tool = params.get("tool_name") or params.get("tool") or params.get("name") or params.get("action")
    if isinstance(raw_tool, str):
        tool_name = _normalise_bbo_tool_name(raw_tool)
        if tool_name is None:
            return None
        raw_arguments = params.get("arguments", params.get("args", params.get("parameters", {})))
        return tool_name, _normalise_bbo_tool_arguments(tool_name, raw_arguments)

    for key, value in params.items():
        tool_name = _normalise_bbo_tool_name(key)
        if tool_name is not None:
            return tool_name, _normalise_bbo_tool_arguments(tool_name, value)
    return None


def _patch_parse_text_tool_calls() -> None:
    """Convert simple XML-like text tool calls into Nanobot tool calls.

    Some local OpenAI-compatible models return the tool-call pattern from their
    chat template as plain assistant text instead of structured ``tool_calls``.
    Nanobot's runner only executes structured calls, so adapt that local-model
    output before the normal agent loop decides whether to execute tools.
    """

    try:
        from nanobot.agent import runner as runner_module  # type: ignore
        from nanobot.providers.base import ToolCallRequest  # type: ignore
    except Exception:
        return

    original_request_model = runner_module.AgentRunner._request_model

    def request(index: int, name: str, arguments: dict[str, Any]) -> Any:
        return ToolCallRequest(
            id=f"call_text_{index}_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=arguments,
        )

    def parse_text_tool_calls(content: str, spec: Any) -> list[Any]:
        calls = []
        for index, block_match in enumerate(_TEXT_TOOL_CALL_RE.finditer(content)):
            block = block_match.group(1)
            bbo_method_calls = _extract_bbo_method_exec_calls(block)
            if bbo_method_calls and spec.tools.has("exec"):
                for offset, (tool_name, arguments) in enumerate(bbo_method_calls):
                    calls.append(request(index + offset, "exec", _bbo_exec_arguments(tool_name, arguments)))
                continue

            function_match = _FUNCTION_TAG_RE.search(block)
            if not function_match:
                continue
            name = function_match.group(1).strip()
            if not spec.tools.has(name):
                bbo_call = _extract_bbo_function_exec_call(name, function_match.group(2))
                if bbo_call is not None and spec.tools.has("exec"):
                    tool_name, arguments = bbo_call
                    calls.append(request(index, "exec", _bbo_exec_arguments(tool_name, arguments)))
                continue
            body = function_match.group(2).strip()
            params: dict[str, Any] = {}
            for param_match in _PARAMETER_TAG_RE.finditer(body):
                params[param_match.group(1).strip()] = _decode_text_tool_value(param_match.group(2))
            if not params and body.startswith("{"):
                decoded = _decode_text_tool_value(body)
                if isinstance(decoded, dict):
                    params = decoded
            calls.append(request(index, name, params))
        if not calls and spec.tools.has("exec"):
            for index, (tool_name, arguments) in enumerate(_extract_bbo_method_exec_calls(content)):
                calls.append(request(index, "exec", _bbo_exec_arguments(tool_name, arguments)))
        return calls

    def strip_text_tool_calls(content: str) -> str:
        return _TEXT_TOOL_CALL_RE.sub("", content).strip()

    async def patched_request_model(self, spec, messages, hook, context):
        response = await original_request_model(self, spec, messages, hook, context)
        content = getattr(response, "content", None)
        if getattr(response, "tool_calls", None) or not isinstance(content, str):
            return response
        calls = parse_text_tool_calls(content, spec)
        if calls:
            response.tool_calls = calls
            response.content = strip_text_tool_calls(content)
            if getattr(response, "finish_reason", None) not in {"tool_calls", "stop"}:
                response.finish_reason = "tool_calls"
        return response

    runner_module.AgentRunner._request_model = patched_request_model


def _patch_log_llm() -> None:
    """Write nanobot's final message snapshot when a log directory is provided."""

    import contextvars
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    log_root = Path(os.environ["BBO_NANOBOT_LOG_DIR"])
    session_key_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "bbo_nanobot_session_key",
        default=None,
    )

    def iso_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def filename_ts() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}Z"

    def reasoning_tokens(usage: dict) -> int | None:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            try:
                return int(details["reasoning_tokens"])
            except Exception:
                return None
        if usage.get("reasoning_tokens") is not None:
            try:
                return int(usage["reasoning_tokens"])
            except Exception:
                return None
        return None

    def reasoning_entries(messages: list) -> list[dict]:
        entries = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            reasoning = message.get("reasoning_content")
            thinking_blocks = message.get("thinking_blocks")
            if not reasoning and not thinking_blocks:
                continue
            entry = {
                "message_index": index,
                "reasoning_content": reasoning if isinstance(reasoning, str) else "",
                "thinking_blocks": thinking_blocks if isinstance(thinking_blocks, list) else [],
                "content_preview": str(message.get("content") or "")[:500],
                "tool_call_count": len(message.get("tool_calls") or []),
            }
            entries.append(entry)
        return entries

    def write_reasoning_trace(session_key: str, messages: list, usage: dict) -> None:
        trace_dir_raw = os.environ.get("BBO_NANOBOT_REASONING_DIR")
        metadata_path_raw = os.environ.get("BBO_NANOBOT_REASONING_METADATA_PATH")
        if not trace_dir_raw and not metadata_path_raw:
            return
        call_id = os.environ.get("BBO_AGENT_CALL_ID") or session_key.replace(":", "_")
        entries = reasoning_entries(messages)
        combined = "\n\n".join(entry["reasoning_content"] for entry in entries if entry.get("reasoning_content"))
        visible = bool(combined.strip() or any(entry.get("thinking_blocks") for entry in entries))
        trace_path: Path | None = None
        payload = {
            "stage": "agent_reasoning",
            "timestamp": iso_now(),
            "sessionKey": session_key,
            "call_id": call_id,
            "model_requested": os.environ.get("BBO_AGENT_MODEL_REQUESTED") or None,
            "provider": os.environ.get("BBO_AGENT_PROVIDER") or None,
            "reasoning_visible": visible,
            "reasoning_content": combined,
            "entries": entries,
            "usage": usage,
            "reasoning_tokens": reasoning_tokens(usage),
        }
        if trace_dir_raw:
            try:
                trace_dir = Path(trace_dir_raw)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"{call_id}_{filename_ts()}_reasoning.json"
                trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                trace_path = None
        if metadata_path_raw:
            try:
                metadata_path = Path(metadata_path_raw)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "timestamp": payload["timestamp"],
                    "sessionKey": session_key,
                    "call_id": call_id,
                    "model_requested": payload["model_requested"],
                    "provider": payload["provider"],
                    "reasoning_visible": visible,
                    "reasoning_chars": len(combined),
                    "reasoning_entry_count": len(entries),
                    "reasoning_tokens": payload["reasoning_tokens"],
                    "trace_path": None if trace_path is None else str(trace_path),
                }
                with metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception:
                pass

    def write_agent_end(session_key: str, messages: list, duration_s: float, usage: dict, success: bool) -> None:
        session_dir = log_root / session_key
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / f"{filename_ts()}_agent-end.json").write_text(
                json.dumps(
                    {
                        "stage": "agent_end",
                        "timestamp": iso_now(),
                        "sessionKey": session_key,
                        "success": success,
                        "durationMs": round(duration_s * 1000, 1),
                        "messageCount": len(messages),
                        "messages": messages,
                        "usage": usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_reasoning_trace(session_key, messages, usage)
        except Exception:
            pass

    try:
        from nanobot.agent import loop as loop_module  # type: ignore

        original_process = loop_module.AgentLoop._process_message
        original_run = loop_module.AgentLoop._run_agent_loop

        async def patched_process(self, msg, session_key=None, **kwargs):
            key = session_key or msg.session_key
            token = session_key_var.set(key)
            try:
                return await original_process(self, msg, session_key=session_key, **kwargs)
            finally:
                session_key_var.reset(token)

        async def patched_run_agent_loop(self, initial_messages, **kwargs):
            start = time.monotonic()
            result = await original_run(self, initial_messages, **kwargs)
            final_content = result[0]
            messages = result[2]
            session_key = session_key_var.get()
            if session_key:
                write_agent_end(
                    session_key=session_key,
                    messages=messages,
                    duration_s=time.monotonic() - start,
                    usage=dict(getattr(self, "_last_usage", {})),
                    success=final_content is not None,
                )
            return result

        loop_module.AgentLoop._process_message = patched_process
        loop_module.AgentLoop._run_agent_loop = patched_run_agent_loop
    except Exception:
        pass


def _patch_bbo_mode_isolation() -> None:
    """Suppress Nanobot's default skill/templates for BBO isolation modes."""

    no_bbo_tool_mode = os.environ.get("BBO_NANOBOT_NO_TOOL_MODE") == "1"
    no_skill_mode = no_bbo_tool_mode or os.environ.get("BBO_NANOBOT_NO_SKILL_MODE") == "1"
    if not (no_bbo_tool_mode or no_skill_mode):
        return

    if no_skill_mode:
        _patch_nanobot_skills_disabled()
    _patch_workspace_template_sync(disable_tools=no_bbo_tool_mode, disable_skills=no_skill_mode)


def _patch_nanobot_skills_disabled() -> None:
    try:
        from nanobot.agent import context as context_module  # type: ignore
        from nanobot.agent import skills as skills_module  # type: ignore
    except Exception:
        return

    def empty_skills_summary(self, exclude=None):  # noqa: ANN001
        del self, exclude
        return ""

    def no_always_skills(self):  # noqa: ANN001
        del self
        return []

    def no_list_skills(self, filter_unavailable=True):  # noqa: ANN001
        del self, filter_unavailable
        return []

    skills_module.SkillsLoader.build_skills_summary = empty_skills_summary
    skills_module.SkillsLoader.get_always_skills = no_always_skills
    skills_module.SkillsLoader.list_skills = no_list_skills

    original_get_identity = context_module.ContextBuilder._get_identity

    def patched_get_identity(self, channel=None):  # noqa: ANN001
        text = original_get_identity(self, channel=channel)
        return "\n".join(
            line
            for line in text.splitlines()
            if "Custom skills:" not in line and "SKILL.md" not in line
        ).strip()

    context_module.ContextBuilder._get_identity = patched_get_identity


def _patch_workspace_template_sync(*, disable_tools: bool, disable_skills: bool) -> None:
    try:
        from nanobot.cli import commands as commands_module  # type: ignore
    except Exception:
        return

    if disable_tools or disable_skills:

        def no_sync(workspace, silent=False):  # noqa: ANN001
            del workspace, silent
            return []

        commands_module.sync_workspace_templates = no_sync
        return


def main() -> None:
    _patch_shell_env_for_bbo_call_id()
    _patch_bbo_mode_isolation()

    if os.environ.get("BBO_NANOBOT_NO_MAX_TOKENS") == "1":
        _patch_strip_max_tokens()

    if os.environ.get("BBO_NANOBOT_PARSE_TEXT_TOOL_CALLS") == "1":
        _patch_parse_text_tool_calls()

    if os.environ.get("BBO_NANOBOT_LOG_DIR"):
        _patch_log_llm()

    from nanobot.cli.commands import app  # noqa: E402  # type: ignore

    app()


if __name__ == "__main__":
    main()
