#!/usr/bin/env bash
set -euo pipefail

BBO_SGLANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBO_REPO_ROOT="$(cd "${BBO_SGLANG_DIR}/../.." && pwd)"

bbo_sglang_matrix() {
  local family="$1"
  local algorithm="$2"
  shift 2
  exec python "${BBO_REPO_ROOT}/scripts/sglang/matrix.py" \
    --family "${family}" \
    --algorithm "${algorithm}" \
    "$@"
}

bbo_arg_present() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

bbo_env_csv() {
  local value="${1:-}"
  value="${value// /,}"
  printf '%s' "${value}"
}

bbo_print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

bbo_exec() {
  bbo_print_command "$@"
  exec "$@"
}

bbo_nanobot_validation_env() {
  export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:18300/v1}"
  export SGLANG_MODEL="${SGLANG_MODEL:-qwen3.5-9b}"
  export SGLANG_TIMEOUT_SECONDS="${SGLANG_TIMEOUT_SECONDS:-600}"
  export SGLANG_MAX_RETRIES="${SGLANG_MAX_RETRIES:-0}"
  export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${LOCAL_LLM_API_KEY}}"
  export SGLANG_API_KEY="${SGLANG_API_KEY:-${LOCAL_LLM_API_KEY}}"
  export AGENT_HISTORY_LIMIT="${AGENT_HISTORY_LIMIT:-200}"
  export AGENT_MAX_TOOL_CALLS="${AGENT_MAX_TOOL_CALLS:-64}"
  export AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS:-900}"
  export AGENT_MAX_RETRIES="${AGENT_MAX_RETRIES:-4}"
  export BBO_NANOBOT_MAX_TOKENS="${BBO_NANOBOT_MAX_TOKENS:-2048}"
  export NANOBOT_VALIDATION_EVALUATIONS="${NANOBOT_VALIDATION_EVALUATIONS:-5}"
  export JOBS="${JOBS:-4}"
  export RUN_ROOT="${RUN_ROOT:-${BBO_REPO_ROOT}/workflow/script_runs/sglang_nanobot_validation_$(date +%Y%m%d_%H%M%S)}"
}

bbo_sglang_config_tasks_csv() {
  local family="$1"
  python - "${BBO_REPO_ROOT}/scripts/sglang/configs/${family}.toml" <<'PY'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("rb") as handle:
    config = tomllib.load(handle)
print(",".join(str(task) for task in config.get("tasks", [])))
PY
}

bbo_sglang_config_scalar() {
  local family="$1"
  local key="$2"
  python - "${BBO_REPO_ROOT}/scripts/sglang/configs/${family}.toml" "${key}" <<'PY'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
key = sys.argv[2]
with config_path.open("rb") as handle:
    config = tomllib.load(handle)
value = config.get(key)
if value is None:
    raise SystemExit(f"{config_path} has no top-level key {key!r}")
print(value)
PY
}
