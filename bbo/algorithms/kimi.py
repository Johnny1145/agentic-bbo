"""Shared Kimi Code API defaults for OpenAI-compatible adapters."""

from __future__ import annotations

DEFAULT_KIMI_API_KEY_ENV = "KIMI_API_KEY"
DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/"
DEFAULT_KIMI_OPENAI_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_KIMI_MODEL = "kimi-for-coding"


def normalize_kimi_openai_base_url(base_url: str | None = None) -> str:
    """Return a Kimi base URL suitable for OpenAI SDK clients.

    Kimi Code documents ``https://api.kimi.com/coding/`` as the Anthropic-compatible
    base and ``https://api.kimi.com/coding/v1`` as the OpenAI-compatible base.
    Users often provide the shorter coding URL, so normalize that form here.
    """

    base = (base_url or DEFAULT_KIMI_OPENAI_BASE_URL).strip().rstrip("/")
    if not base:
        return DEFAULT_KIMI_OPENAI_BASE_URL
    if base.endswith("/coding"):
        return f"{base}/v1"
    return base
