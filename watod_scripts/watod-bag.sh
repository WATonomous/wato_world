#!/usr/bin/env bash
# Mount data/bags and exec ros2 bag inside the ingest image.
# Mirrors wato_monorepo/watod_scripts/watod-bag.sh.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})

# Bring up the ingest service if not already running, then exec ros2 bag.
docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    --profile ingest --profile infra \
    up -d ingest

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    exec ingest ros2 bag "$@"
