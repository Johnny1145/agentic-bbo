#!/usr/bin/env bash
set -euo pipefail

BBO_API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBO_REPO_ROOT="$(cd "${BBO_API_DIR}/../.." && pwd)"

bbo_api_matrix() {
  local family="$1"
  local algorithm="$2"
  shift 2
  local family_upper
  family_upper="$(printf '%s' "${family}" | tr '[:lower:]-' '[:upper:]_')"
  local family_run_root_var="${family_upper}_RUN_ROOT"
  if [[ -z "${RUN_ROOT:-}" && -z "${!family_run_root_var:-}" ]]; then
    RUN_ROOT="${BBO_REPO_ROOT}/workflow/script_runs/api_${family}_$(date +%Y%m%d_%H%M%S)"
    export RUN_ROOT
  fi
  exec python "${BBO_REPO_ROOT}/scripts/api/matrix.py" \
    --family "${family}" \
    --algorithm "${algorithm}" \
    "$@"
}
