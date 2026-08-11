#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/sglang/dbtune/run_problem.sh <problem_name> [algorithm_name] [extra bbo.run args...]" >&2
  exit 2
fi

problem_name="$1"
shift

algorithm_name="${BBO_ALGORITHM:-random_search}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  algorithm_name="$1"
  shift
fi

probe_http() {
  local url="$1"
  python - "$url" <<'PY'
import sys, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=1.0) as resp:
        sys.exit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_http() {
  local url="$1"
  local tries="${2:-40}"
  local delay="${3:-0.5}"
  for _ in $(seq 1 "$tries"); do
    if probe_http "$url"; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

start_bboplace_local_bridge() {
  local upstream_root="${BBOPLACE_UPSTREAM_ROOT:-}"
  if [[ -z "$upstream_root" ]]; then
    for candidate in \
      "/opt/BBOPlace-miniBench" \
      "/home/trx/cm/BBOPlace-miniBench" \
      "/opt/BBOPlace-Bench" \
      "/home/trx/cm/BBOPlace-Bench"
    do
      if [[ -d "$candidate" ]]; then
        upstream_root="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$upstream_root" || ! -d "$upstream_root" ]]; then
    echo "bboplace_bench requires BBOPLACE_UPSTREAM_ROOT or a known local BBOPlace checkout." >&2
    return 1
  fi

  export BBOPLACE_UPSTREAM_ROOT="$upstream_root"
  local host="${BBOPLACE_HOST:-127.0.0.1}"
  local port="${BBOPLACE_PORT:-8070}"
  local log_dir="${BBO_RUN_LOG_DIR:-artifacts/service_logs}"
  mkdir -p "$log_dir"
  local log_path="$log_dir/bboplace_local_bridge.log"

  uv run python -m bbo.tasks.bboplace.local_service \
    --upstream-root "$upstream_root" \
    --host "$host" \
    --port "$port" \
    >"$log_path" 2>&1 &
  BBOPLACE_BRIDGE_PID=$!
  export BBOPLACE_BASE_URL="http://${host}:${port}"

  if ! wait_http "${BBOPLACE_BASE_URL}/health"; then
    echo "failed to start local BBOPlace bridge; see ${log_path}" >&2
    return 1
  fi
  trap 'if [[ -n "${BBOPLACE_BRIDGE_PID:-}" ]]; then kill "${BBOPLACE_BRIDGE_PID}" >/dev/null 2>&1 || true; fi' EXIT
}

case "$problem_name" in
  bboplace_bench)
    if ! probe_http "${BBOPLACE_BASE_URL:-http://127.0.0.1:8070}/health"; then
      start_bboplace_local_bridge
    fi
    ;;
  knob_http_mariadb_*)
    echo "MariaDB dbtune tasks are no longer part of the active dbtune task set; use one of the knob_http_surrogate_* tasks." >&2
    exit 2
    ;;
  knob_http_surrogate_*)
    if ! probe_http "${AGENTBBO_HTTP_SURROGATE_BASE_URL:-http://127.0.0.1:8090}/health"; then
      echo "Surrogate evaluator is not reachable. It still needs the separate Python 3.7 image." >&2
      exit 1
    fi
    ;;
esac

uv_extra_args=()
if [[ -n "${BBO_UV_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  uv_extra_args=(${BBO_UV_EXTRA_ARGS})
fi

exec uv run \
  "${uv_extra_args[@]}" \
  python -m bbo.run \
  --task "$problem_name" \
  --algorithm "$algorithm_name" \
  "$@"
