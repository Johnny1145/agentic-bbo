#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

exec uv run --extra nanobot python -m bbo.benchmark.nanobot run \
  --task "${TASK:-bbob_f01_d10}" \
  --seed "${SEED:-1}" \
  --skill-mode "${SKILL_MODE:-no-skill}" \
  --initial-random "${INITIAL_RANDOM:-20}" \
  --optimizer-budget "${OPTIMIZER_BUDGET:-100}" \
  --agent-tool-mode workspace_json \
  --no-plots \
  "$@"
