#!/bin/sh
set -eu

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  attempts=30
  until alembic upgrade head; do
    attempts=$((attempts - 1))
    if [ "$attempts" -le 0 ]; then
      echo "Database migrations failed after retries." >&2
      exit 1
    fi
    echo "Database unavailable or migration failed; retrying in 2 seconds..." >&2
    sleep 2
  done
fi

exec "$@"
