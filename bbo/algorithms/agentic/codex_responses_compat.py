"""Translate Codex Responses requests to SGLang Chat Completions."""

from __future__ import annotations

import http.client
import json
import threading
import time
import uuid
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import SplitResult, urlsplit


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def _content_text(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                parts.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
            elif isinstance(image_url, dict):
                parts.append({"type": "image_url", "image_url": image_url})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return str(parts[0]["text"])
    return parts


def _append_function_call(
    messages: list[dict[str, Any]],
    item: dict[str, Any],
) -> None:
    call = {
        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": _qualified_name(item),
            "arguments": item.get("arguments") or "{}",
        },
    }
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get(
        "tool_calls"
    ):
        messages[-1]["tool_calls"].append(call)
    else:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
            }
        )


def _qualified_name(item: dict[str, Any]) -> str:
    namespace = item.get("namespace")
    name = str(item.get("name", ""))
    return f"{namespace}.{name}" if namespace else name


def _chat_tools(response_tools: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if not isinstance(response_tools, list):
        return tools
    for tool in response_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters")
                        or {"type": "object"},
                        "strict": bool(tool.get("strict", False)),
                    },
                }
            )
        elif tool_type == "namespace":
            namespace = str(tool.get("name", ""))
            for nested in tool.get("tools") or []:
                if not isinstance(nested, dict) or nested.get("type") != "function":
                    continue
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{namespace}.{nested.get('name', '')}",
                            "description": nested.get("description", ""),
                            "parameters": nested.get("parameters")
                            or {"type": "object"},
                            "strict": bool(nested.get("strict", False)),
                        },
                    }
                )
    return tools


def responses_to_sglang_chat_request(body: bytes) -> bytes:
    """Convert one Responses request, including prior tool turns."""

    payload = json.loads(body)
    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    if payload.get("instructions"):
        system_parts.append(str(payload["instructions"]))

    response_input = payload.get("input")
    if isinstance(response_input, str):
        messages.append({"role": "user", "content": response_input})
    elif isinstance(response_input, list):
        for item in response_input:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                role = str(item.get("role", "user"))
                content = _content_text(item.get("content"))
                if role in {"developer", "system"}:
                    if isinstance(content, str) and content:
                        system_parts.append(content)
                    elif isinstance(content, list):
                        system_parts.extend(
                            str(part.get("text", ""))
                            for part in content
                            if part.get("type") == "text"
                        )
                    continue
                if content:
                    messages.append({"role": role, "content": content})
            elif item_type == "function_call":
                _append_function_call(messages, item)
            elif item_type == "function_call_output":
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id") or item.get("id") or "",
                        "content": output,
                    }
                )
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    request: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_tokens": payload.get("max_output_tokens") or 8192,
        "stream": bool(payload.get("stream", True)),
        "separate_reasoning": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
        },
    }
    tools = _chat_tools(payload.get("tools"))
    if tools:
        request["tools"] = tools
        request["tool_choice"] = payload.get("tool_choice") or "auto"
        request["parallel_tool_calls"] = bool(
            payload.get("parallel_tool_calls", False)
        )
    if request["stream"]:
        request["stream_options"] = {"include_usage": True}
    for key in ("temperature", "top_p"):
        if payload.get(key) is not None:
            request[key] = payload[key]
    return _json_bytes(request)


def _sse_payload(event: bytes) -> dict[str, Any] | str | None:
    data_lines: list[str] = []
    try:
        text = event.decode()
    except UnicodeDecodeError:
        return None
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return data
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class ChatCompletionAccumulator:
    """Accumulate a chat stream and emit one Responses-compatible turn."""

    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.text_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, int] = {}

    def add(self, event: bytes) -> bool:
        chunk = _sse_payload(event)
        if chunk == "[DONE]":
            return True
        if not isinstance(chunk, dict):
            return False
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            }
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if not isinstance(delta, dict):
            return False
        text = delta.get("content")
        if text is None:
            text = delta.get("reasoning_content")
        if text:
            self.text_parts.append(str(text))
        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index") or 0)
            state = self.tool_calls.setdefault(
                index,
                {
                    "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                    "name": "",
                    "arguments": "",
                },
            )
            if tool_call.get("id"):
                state["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                state["name"] = function["name"]
            if function.get("arguments"):
                state["arguments"] += function["arguments"]
        return False

    def events(self) -> list[bytes]:
        response_id = f"resp_{uuid.uuid4().hex}"
        initial = _response_object(
            response_id=response_id,
            request=self.request,
            status="in_progress",
            output=[],
            usage=None,
        )
        output_items: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = [
            {"type": "response.created", "response": initial},
            {"type": "response.in_progress", "response": initial},
        ]

        text = "".join(self.text_parts)
        if text:
            item_id = f"msg_{uuid.uuid4().hex}"
            part = {"type": "output_text", "text": text, "annotations": []}
            item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [part],
                "status": "completed",
            }
            output_index = len(output_items)
            output_items.append(item)
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {**item, "content": [], "status": "in_progress"},
                    },
                    {
                        "type": "response.content_part.added",
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": [],
                    },
                    {
                        "type": "response.output_text.done",
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "text": text,
                        "logprobs": [],
                    },
                    {
                        "type": "response.content_part.done",
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": part,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item,
                    },
                ]
            )

        for tool_call in self.tool_calls.values():
            item_id = f"fc_{uuid.uuid4().hex}"
            qualified_name = str(tool_call["name"])
            namespace = None
            name = qualified_name
            if "." in qualified_name:
                namespace, name = qualified_name.split(".", 1)
            item = {
                "id": item_id,
                "call_id": tool_call["id"],
                "name": name,
                "arguments": tool_call["arguments"] or "{}",
                "type": "function_call",
                "status": "completed",
            }
            if namespace:
                item["namespace"] = namespace
            output_index = len(output_items)
            output_items.append(item)
            in_progress_item = {**item, "arguments": "", "status": "in_progress"}
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": in_progress_item,
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": item["arguments"],
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": output_index,
                        "item_id": item_id,
                        "name": name,
                        "arguments": item["arguments"],
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item,
                    },
                ]
            )

        usage = {
            "input_tokens": self.usage.get("input_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": self.usage.get("output_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": self.usage.get("input_tokens", 0)
            + self.usage.get("output_tokens", 0),
        }
        completed = _response_object(
            response_id=response_id,
            request=self.request,
            status="completed",
            output=output_items,
            usage=usage,
        )
        events.append({"type": "response.completed", "response": completed})
        return [
            _format_response_event({**event, "sequence_number": index})
            for index, event in enumerate(events)
        ]


def _response_object(
    *,
    response_id: str,
    request: dict[str, Any],
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": request.get("metadata") or {},
        "model": request.get("model") or "",
        "output": output,
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", False)),
        "temperature": request.get("temperature"),
        "tool_choice": request.get("tool_choice") or "auto",
        "tools": [],
        "top_p": request.get("top_p"),
        "background": False,
        "completed_at": time.time() if status == "completed" else None,
        "max_output_tokens": request.get("max_output_tokens"),
        "max_tool_calls": request.get("max_tool_calls"),
        "previous_response_id": request.get("previous_response_id"),
        "prompt": None,
        "prompt_cache_key": request.get("prompt_cache_key"),
        "prompt_cache_retention": None,
        "reasoning": request.get("reasoning"),
        "safety_identifier": None,
        "service_tier": request.get("service_tier") or "default",
        "status": status,
        "text": request.get("text") or {"format": {"type": "text"}},
        "top_logprobs": request.get("top_logprobs") or 0,
        "truncation": request.get("truncation") or "disabled",
        "usage": usage,
        "user": request.get("user"),
    }


def _format_response_event(event: dict[str, Any]) -> bytes:
    return (
        b"event: "
        + str(event["type"]).encode()
        + b"\ndata: "
        + _json_bytes(event)
        + b"\n\n"
    )


def chat_to_responses_sse(
    events: Iterable[bytes],
    *,
    request: dict[str, Any],
) -> Iterable[bytes]:
    """Translate a complete chat stream to Responses events."""

    accumulator = ChatCompletionAccumulator(request)
    finished = False
    for event in events:
        if accumulator.add(event):
            finished = True
            break
    if finished:
        yield from accumulator.events()


def _iter_sse_events(response: http.client.HTTPResponse) -> Iterable[bytes]:
    buffer = bytearray()
    while True:
        line = response.readline()
        if not line:
            if buffer:
                yield bytes(buffer)
            return
        buffer.extend(line)
        if line in {b"\n", b"\r\n"}:
            yield bytes(buffer)
            buffer.clear()


class _CompatibilityHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream: SplitResult):
        self.upstream = upstream
        super().__init__(("127.0.0.1", 0), _CompatibilityRequestHandler)


class _CompatibilityRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        server = self.server
        assert isinstance(server, _CompatibilityHTTPServer)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        request_path = self.path.split("?", 1)[0]
        translating = request_path.endswith("/v1/responses")
        original_request: dict[str, Any] = {}
        if translating:
            original_request = json.loads(body)
            body = responses_to_sglang_chat_request(body)
            request_path = "/v1/chat/completions"

        upstream = server.upstream
        connection_cls = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            upstream.hostname,
            upstream.port,
            timeout=600,
        )
        base_path = upstream.path.rstrip("/")
        if base_path.endswith("/v1") and request_path.startswith("/v1/"):
            request_path = request_path[3:]
        upstream_path = f"{base_path}{request_path}"
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        headers["Host"] = upstream.netloc
        try:
            connection.request("POST", upstream_path, body=body, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            if (
                translating
                and response.status < 400
                and "text/event-stream" in content_type
            ):
                self._relay_translated_stream(
                    response,
                    request=original_request,
                )
            else:
                self._relay_buffered(response)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - network failure path
            payload = _json_bytes(
                {
                    "error": {
                        "message": f"SGLang Responses compatibility proxy failed: {exc}",
                        "type": "server_error",
                    }
                }
            )
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            connection.close()

    def _relay_translated_stream(
        self,
        response: http.client.HTTPResponse,
        *,
        request: dict[str, Any],
    ) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in chat_to_responses_sse(
            _iter_sse_events(response),
            request=request,
        ):
            self.wfile.write(event)
            self.wfile.flush()
        self.close_connection = True

    def _relay_buffered(self, response: http.client.HTTPResponse) -> None:
        payload = response.read()
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in _HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class SGLangResponsesCompatibilityProxy:
    """Run an ephemeral local Responses-to-Chat-Completions proxy."""

    def __init__(self, upstream_base_url: str):
        upstream = urlsplit(upstream_base_url)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError(
                f"Invalid SGLang Responses API base URL: {upstream_base_url!r}"
            )
        self._server = _CompatibilityHTTPServer(upstream)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="codex-sglang-responses-compat",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
