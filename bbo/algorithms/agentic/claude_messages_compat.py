"""Translate Claude Code's Messages API to SGLang Chat Completions.

Claude Code speaks Anthropic's Messages protocol while the benchmark's local
models are served by SGLang.  SGLang exposes a Messages endpoint, but current
releases do not forward Qwen's thinking controls and can emit invalid
content-block transitions around native tool calls.  This local proxy performs
the narrow protocol translation and leaves Claude Code's native tools intact.
"""

from __future__ import annotations

import http.client
import json
import threading
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
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def _format_sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    return b"event: " + event_type.encode() + b"\ndata: " + _json_bytes(data) + b"\n\n"


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and source.get("data"):
        media_type = source.get("media_type", "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source['data']}"},
        }
    if source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    if not isinstance(content, list):
        return str(content or "")
    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append({"type": "text", "text": str(item.get("text", ""))})
        elif item.get("type") == "image":
            image = _image_part(item)
            if image is not None:
                parts.append(image)
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0]["text"])
    return parts


def anthropic_to_sglang_chat_request(
    body: bytes,
    *,
    max_output_tokens: int | None = None,
) -> bytes:
    """Convert one Anthropic Messages request to an SGLang chat request."""

    payload = json.loads(body)
    messages: list[dict[str, Any]] = []

    system = payload.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_text = "\n".join(
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and block.get("text") is not None:
                content_parts.append({"type": "text", "text": block["text"]})
            elif block_type == "image":
                image = _image_part(block)
                if image is not None:
                    content_parts.append(image)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input") or {},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id")
                        or block.get("id")
                        or "",
                        "content": _tool_result_content(block.get("content")),
                    }
                )

        if role == "user":
            messages.extend(tool_results)
        openai_message: dict[str, Any] = {"role": role}
        if tool_calls:
            openai_message["tool_calls"] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_message["content"] = content_parts[0]["text"]
            else:
                openai_message["content"] = content_parts
        elif tool_calls:
            openai_message["content"] = None
        else:
            continue
        messages.append(openai_message)

    requested_max_tokens = int(payload["max_tokens"])
    if max_output_tokens is not None:
        requested_max_tokens = min(requested_max_tokens, int(max_output_tokens))
    request: dict[str, Any] = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": requested_max_tokens,
        "stream": bool(payload.get("stream", False)),
        "separate_reasoning": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
        },
    }
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop_sequences", "stop"),
    ):
        if payload.get(source) is not None:
            request[target] = payload[source]
    if request["stream"]:
        request["stream_options"] = {"include_usage": True}

    tools = payload.get("tools")
    if isinstance(tools, list):
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
            for tool in tools
            if isinstance(tool, dict) and not tool.get("defer_loading")
        ]
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "none":
            request["tool_choice"] = "none"
        elif choice_type == "any":
            request["tool_choice"] = "required"
        elif choice_type == "tool":
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        else:
            request["tool_choice"] = "auto"
    elif request.get("tools"):
        request["tool_choice"] = "auto"
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


class OpenAIToAnthropicStream:
    """Translate OpenAI chat chunks into a valid Anthropic SSE stream."""

    def __init__(self, model: str):
        self._model = model
        self._message_id = f"msg_{uuid.uuid4().hex}"
        self._started = False
        self._active_index: int | None = None
        self._active_type: str | None = None
        self._next_index = 0
        self._finish_reason: str | None = None
        self._usage: dict[str, int] = {}

    def translate(self, event: bytes) -> list[bytes]:
        chunk = _sse_payload(event)
        if chunk == "[DONE]":
            return self._finish()
        if not isinstance(chunk, dict):
            return []

        output: list[bytes] = []
        if not self._started:
            self._started = True
            output.append(
                _format_sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self._message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self._model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )
            )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self._usage = {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            }
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return output
        choice = choices[0]
        if not isinstance(choice, dict):
            return output
        if choice.get("finish_reason") is not None:
            self._finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return output

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                name = function.get("name")
                if name:
                    output.extend(
                        self._start_block(
                            "tool_use",
                            {
                                "type": "tool_use",
                                "id": tool_call.get("id")
                                or f"call_{uuid.uuid4().hex}",
                                "name": name,
                                "input": {},
                            },
                        )
                    )
                arguments = function.get("arguments")
                if arguments:
                    if self._active_type != "tool_use":
                        continue
                    output.append(
                        self._delta(
                            {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            }
                        )
                    )

        text = delta.get("content")
        if text is None:
            # This should remain empty when thinking is disabled. Mapping it to
            # text is a lossless fallback for models that ignore the toggle.
            text = delta.get("reasoning_content")
        if text:
            if self._active_type != "text":
                output.extend(
                    self._start_block("text", {"type": "text", "text": ""})
                )
            output.append(self._delta({"type": "text_delta", "text": text}))
        return output

    def _start_block(
        self,
        block_type: str,
        content_block: dict[str, Any],
    ) -> list[bytes]:
        output = []
        if self._active_index is not None:
            output.append(self._stop_block())
        self._active_index = self._next_index
        self._next_index += 1
        self._active_type = block_type
        output.append(
            _format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._active_index,
                    "content_block": content_block,
                },
            )
        )
        return output

    def _delta(self, delta: dict[str, Any]) -> bytes:
        assert self._active_index is not None
        return _format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._active_index,
                "delta": delta,
            },
        )

    def _stop_block(self) -> bytes:
        assert self._active_index is not None
        event = _format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": self._active_index},
        )
        self._active_index = None
        self._active_type = None
        return event

    def _finish(self) -> list[bytes]:
        output = []
        if not self._started:
            return output
        if self._active_index is not None:
            output.append(self._stop_block())
        output.append(
            _format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _STOP_REASON_MAP.get(
                            self._finish_reason or "stop",
                            "end_turn",
                        ),
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": self._usage.get("output_tokens", 0)
                    },
                },
            )
        )
        output.append(_format_sse_event("message_stop", {"type": "message_stop"}))
        return output


def openai_to_anthropic_sse(
    events: Iterable[bytes],
    *,
    model: str,
) -> Iterable[bytes]:
    """Translate a sequence of complete OpenAI SSE events."""

    translator = OpenAIToAnthropicStream(model)
    for event in events:
        yield from translator.translate(event)


def openai_to_anthropic_response(body: bytes, *, model: str) -> bytes:
    """Translate a non-streaming OpenAI response."""

    payload = json.loads(body)
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        try:
            tool_input = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "input": tool_input,
            }
        )
    usage = payload.get("usage") or {}
    return _json_bytes(
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model,
            "stop_reason": _STOP_REASON_MAP.get(
                choice.get("finish_reason") or "stop",
                "end_turn",
            ),
            "stop_sequence": None,
            "usage": {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            },
        }
    )


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

    def __init__(
        self,
        upstream: SplitResult,
        *,
        max_output_tokens: int | None,
    ):
        self.upstream = upstream
        self.max_output_tokens = max_output_tokens
        super().__init__(("127.0.0.1", 0), _CompatibilityRequestHandler)


class _CompatibilityRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        server = self.server
        assert isinstance(server, _CompatibilityHTTPServer)
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length)
        request_path = self.path.split("?", 1)[0]
        translating_messages = request_path.endswith("/v1/messages")
        model = ""
        if translating_messages:
            original = json.loads(body)
            model = str(original.get("model", ""))
            body = anthropic_to_sglang_chat_request(
                body,
                max_output_tokens=server.max_output_tokens,
            )
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
            is_stream = "text/event-stream" in content_type
            if translating_messages and is_stream:
                self._relay_translated_stream(response, model=model)
            elif translating_messages and response.status < 400:
                self._relay_translated_buffered(response, model=model)
            else:
                self._relay_buffered(response)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - network failure path
            payload = _json_bytes(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"SGLang Messages compatibility proxy failed: {exc}",
                    },
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
        model: str,
    ) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in openai_to_anthropic_sse(
            _iter_sse_events(response),
            model=model,
        ):
            self.wfile.write(event)
            self.wfile.flush()
        self.close_connection = True

    def _relay_translated_buffered(
        self,
        response: http.client.HTTPResponse,
        *,
        model: str,
    ) -> None:
        payload = openai_to_anthropic_response(response.read(), model=model)
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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


class SGLangMessagesCompatibilityProxy:
    """Run an ephemeral local Messages-to-Chat-Completions proxy."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        max_output_tokens: int | None = None,
    ):
        upstream = urlsplit(upstream_base_url)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError(
                f"Invalid SGLang Messages API base URL: {upstream_base_url!r}"
            )
        self._server = _CompatibilityHTTPServer(
            upstream,
            max_output_tokens=max_output_tokens,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="claude-sglang-messages-compat",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> SGLangMessagesCompatibilityProxy:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
