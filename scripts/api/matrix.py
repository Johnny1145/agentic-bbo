#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def _load_sglang_matrix():
    target = Path(__file__).resolve().parents[1] / "sglang" / "matrix.py"
    spec = importlib.util.spec_from_file_location("_bbo_sglang_matrix_impl", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load matrix implementation from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_dry_run(argv: list[str]) -> bool:
    return any(arg in {"--dry-run", "--list", "--list-families"} for arg in argv)


def _prepare_api_env(argv: list[str]) -> None:
    dry_run = _is_dry_run(argv)
    key_env = os.environ.get("API_KEY_ENV", "OPENAI_API_KEY")
    api_key = os.environ.get(key_env) or os.environ.get("LOCAL_LLM_API_KEY")
    model = os.environ.get("API_MODEL") or os.environ.get("OPENAI_MODEL")
    base_url = (
        os.environ.get("API_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )

    if not model and not dry_run:
        raise SystemExit("Set API_MODEL or OPENAI_MODEL before running API baselines.")
    if not api_key and not dry_run:
        raise SystemExit(f"Set {key_env} or LOCAL_LLM_API_KEY before running API baselines.")

    os.environ["SGLANG_BASE_URL"] = base_url
    os.environ["SGLANG_MODEL"] = model or "API_MODEL"
    os.environ["LOCAL_LLM_API_KEY"] = api_key or "API_KEY"
    os.environ.setdefault("SGLANG_TIMEOUT_SECONDS", os.environ.get("API_TIMEOUT_SECONDS", "600"))
    os.environ.setdefault("SGLANG_MAX_RETRIES", os.environ.get("API_MAX_RETRIES", "1"))
    os.environ.setdefault("SKIP_SGLANG_CHECK", "1")


_MODULE = _load_sglang_matrix()
for _name, _value in vars(_MODULE).items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value


def main(argv: list[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    _prepare_api_env(actual_argv)
    return _MODULE.main(actual_argv)


if __name__ == "__main__":
    raise SystemExit(main())
