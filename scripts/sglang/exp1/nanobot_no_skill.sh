#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${SGLANG_BASE_URL:=http://127.0.0.1:18300/v1}"
: "${SGLANG_MODEL:=qwen3.5-9b}"
: "${LOCAL_LLM_API_KEY:=EMPTY}"
export LOCAL_LLM_API_KEY

extra_args=("$@")
has_results_root=0
for arg in "${extra_args[@]}"; do
  if [[ "${arg}" == "--results-root" || "${arg}" == --results-root=* ]]; then
    has_results_root=1
    break
  fi
done
if [[ "${has_results_root}" == "0" ]]; then
  extra_args+=(--results-root "${REPO_ROOT}/workflow/exp1/outputs/sglang_no_skill")
fi

cd "${REPO_ROOT}"
exec uv run --extra nanobot python \
  workflow/exp1/run_prior_restriction_nanobot_skill_compare.py \
  --variant no-skill \
  --agent-provider openai \
  --agent-model "${SGLANG_MODEL}" \
  --agent-api-base "${SGLANG_BASE_URL}" \
  --agent-api-key-env LOCAL_LLM_API_KEY \
  --agent-tool-mode workspace_json \
  --agent-web-search-provider disabled \
  --agent-code-backend local_disabled \
  --no-agent-enable-code-interpreter \
  --no-agent-allow-fallback \
  "${extra_args[@]}"
