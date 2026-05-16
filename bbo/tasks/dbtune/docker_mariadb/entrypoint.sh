#!/usr/bin/env bash
set -euo pipefail

ROOT_PW="123456"
DB_NAME="sbtest"
MYSQL_SOCKET="/var/run/mysqld/mysqld.sock"

export MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-${ROOT_PW}}"
export MYSQL_DATABASE="${MYSQL_DATABASE:-${DB_NAME}}"

wait_mysql() {
  local retries="${1:-60}"
  local delay_sec="${2:-1}"
  for _ in $(seq 1 "${retries}"); do
    if mysqladmin ping -uroot -p"${MYSQL_ROOT_PASSWORD}" --silent >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay_sec}"
  done
  return 1
}

echo "1. Bootstrapping MySQL 5.7..."
/usr/local/bin/docker-entrypoint.sh mysqld --daemonize --skip-networking=0

if ! wait_mysql 90 1; then
  echo "MySQL 5.7 did not become ready in time." >&2
  exit 1
fi

echo "2. Initializing database..."
mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" --protocol=socket --socket="${MYSQL_SOCKET}" \
  -e "CREATE DATABASE IF NOT EXISTS \`${MYSQL_DATABASE}\`;" || true

echo "3. Checking sysbench tables..."
RAW_COUNT="$(mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" --protocol=socket --socket="${MYSQL_SOCKET}" -N -e \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='${MYSQL_DATABASE}';" \
  2>/dev/null | tr -d '[:space:]' || true)"
TABLE_COUNT="${RAW_COUNT:-0}"

if [ "${TABLE_COUNT}" -eq 0 ]; then
  echo "   First run: preparing sysbench data (may take several minutes)..."
  sysbench --db-driver=mysql --mysql-user=root --mysql-password="${MYSQL_ROOT_PASSWORD}" \
    --mysql-db="${MYSQL_DATABASE}" --tables=10 --table-size=100000 \
    oltp_read_write prepare
fi

echo "4. Starting evaluation API on :8080 ..."
exec python3 /app/server.py
