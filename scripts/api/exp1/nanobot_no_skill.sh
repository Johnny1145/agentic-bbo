#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

API_BASE_URL="${API_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_MODEL_VALUE="${API_MODEL:-${OPENAI_MODEL:-}}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"

dry_run=0
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    dry_run=1
  fi
done

if [[ -z "${API_MODEL_VALUE}" && "${dry_run}" != "1" ]]; then
  echo "Set API_MODEL or OPENAI_MODEL before running API anonymous Nanobot no-skill." >&2
  exit 2
fi
if [[ -z "${!API_KEY_ENV:-}" && "${dry_run}" != "1" ]]; then
  echo "Set ${API_KEY_ENV} or choose another API_KEY_ENV." >&2
  exit 2
fi

extra_args=("$@")
has_results_root=0
for arg in "${extra_args[@]}"; do
  if [[ "${arg}" == "--results-root" || "${arg}" == --results-root=* ]]; then
    has_results_root=1
    break
  fi
done
if [[ "${has_results_root}" == "0" ]]; then
  extra_args+=(--results-root "${REPO_ROOT}/workflow/exp1/outputs/api_no_skill")
fi

cd "${REPO_ROOT}"
exec uv run --extra nanobot python \
  workflow/exp1/run_prior_restriction_nanobot_skill_compare.py \
  --variant no-skill \
  --agent-provider openai \
  --agent-model "${API_MODEL_VALUE:-API_MODEL}" \
  --agent-api-base "${API_BASE_URL}" \
  --agent-api-key-env "${API_KEY_ENV}" \
  --agent-tool-mode workspace_json \
  --agent-web-search-provider disabled \
  --agent-code-backend local_disabled \
  --no-agent-enable-code-interpreter \
  --no-agent-allow-fallback \
  "${extra_args[@]}"
