#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
bbo_nanobot_validation_env

workflow_script="${BBO_REPO_ROOT}/workflow/20_bboplace_llm_experiments/nanobot_qwen35_n32_seed1_20260618/run_nanobot_local_qwen35_bboplace_full16_n32_seed1_history20_directjson.py"
cmd=(
  uv run --extra bo-tutorial --extra nanobot python "${workflow_script}"
  --bboplace-base-urls "${BBOPLACE_BASE_URLS:-http://127.0.0.1:8270,http://127.0.0.1:8280,http://127.0.0.1:8281,http://127.0.0.1:8282,http://127.0.0.1:8283,http://127.0.0.1:8284,http://127.0.0.1:8285,http://127.0.0.1:8286,http://127.0.0.1:8287}"
  --setting-version sglang_nanobot_validation_skill_5eval_v1
  --n-initial-points 0
  --optimizer-budget "${NANOBOT_VALIDATION_EVALUATIONS}"
  --output-dir "${RUN_ROOT}/bboplace/skill/outputs"
  --summary-dir "${RUN_ROOT}/bboplace/skill/summary"
  --jobs "${JOBS}"
  --local-llm-api-key-env LOCAL_LLM_API_KEY
  --local-llm-model "${SGLANG_MODEL}"
  --local-llm-base-url "${SGLANG_BASE_URL}"
  --agent-prompt-style workspace
  --agent-tool-mode workspace_json
  --agent-history-limit "${AGENT_HISTORY_LIMIT}"
  --agent-max-tool-calls "${AGENT_MAX_TOOL_CALLS}"
  --agent-timeout-seconds "${AGENT_TIMEOUT_SECONDS}"
  --agent-max-retries "${AGENT_MAX_RETRIES}"
  --agent-enable-memory
  --no-agent-enable-code-interpreter
  --agent-code-backend local_disabled
  --agent-web-search-provider disabled
  --no-agent-allow-fallback
  --agent-enable-bbo-skills
)

if ! bbo_arg_present "--seed" "$@"; then
  seed_values="${SEEDS:-1}"
  for seed in ${seed_values//,/ }; do
    cmd+=(--seed "${seed}")
  done
fi

cmd+=("$@")
bbo_exec "${cmd[@]}"
