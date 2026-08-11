#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
bbo_nanobot_validation_env

tasks_csv="$(bbo_sglang_config_tasks_csv synthetic)"
cmd=(
  uv run --extra nanobot python -m bbo.benchmark.nanobot matrix
  --skill-modes skill
  --initial-random 0
  --max-evaluations "${NANOBOT_VALIDATION_EVALUATIONS}"
  --results-root "${RUN_ROOT}/synthetic/skill"
  --no-plots
  --agent-provider openai
  --agent-model "${SGLANG_MODEL}"
  --agent-api-base "${SGLANG_BASE_URL}"
  --agent-api-key-env LOCAL_LLM_API_KEY
  --agent-tool-mode workspace_json
  --agent-prompt-style workspace
  --agent-history-limit "${AGENT_HISTORY_LIMIT}"
  --agent-max-tool-calls "${AGENT_MAX_TOOL_CALLS}"
  --agent-timeout-seconds "${AGENT_TIMEOUT_SECONDS}"
  --agent-max-retries "${AGENT_MAX_RETRIES}"
  --agent-enable-memory
  --no-agent-enable-code-interpreter
  --agent-code-backend local_disabled
  --agent-web-search-provider disabled
  --no-agent-allow-fallback
)

if ! bbo_arg_present "--tasks" "$@"; then
  cmd+=(--tasks "${tasks_csv}")
fi
if ! bbo_arg_present "--seeds" "$@"; then
  cmd+=(--seeds "$(bbo_env_csv "${SEEDS:-1}")")
fi

cmd+=("$@")
bbo_exec "${cmd[@]}"
