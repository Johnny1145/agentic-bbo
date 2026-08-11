#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
bbo_nanobot_validation_env

workflow_script="${BBO_REPO_ROOT}/workflow/30_guacamol_opro_nanobot/sglang_seed1_init50_opt200_20260709/run_guacamol_opro_nanobot_sglang_seed1_init50_opt200.py"
default_init_source="${BBO_REPO_ROOT}/scripts/sglang/data/guacamol_init_smiles_smoke.txt"
init_source="${SMILES_INIT_SOURCE:-${SMILES_INIT_CSV:-${SMILES_INIT_TXT:-${default_init_source}}}}"
cmd=(
  uv run --extra bo-tutorial --extra nanobot python "${workflow_script}"
  --algorithm nanobot
  --n-initial-points 0
  --optimizer-budget "${NANOBOT_VALIDATION_EVALUATIONS}"
  --init-source "${init_source}"
  --output-dir "${RUN_ROOT}/molecule/skill/outputs"
  --summary-dir "${RUN_ROOT}/molecule/skill/summary"
  --jobs "${JOBS}"
  --local-llm-api-key-env LOCAL_LLM_API_KEY
  --local-llm-model "${SGLANG_MODEL}"
  --local-llm-base-url "${SGLANG_BASE_URL}"
  --local-llm-timeout-seconds "${SGLANG_TIMEOUT_SECONDS}"
  --local-llm-max-retries "${SGLANG_MAX_RETRIES}"
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
