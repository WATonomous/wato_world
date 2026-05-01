#!/usr/bin/env bash
# Generic passthrough to `docker compose` with the compose-file stack and
# profile flags assembled by the watod entrypoint.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"
: "${PROFILE_ARGS_STR:?PROFILE_ARGS_STR must be set}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})
# shellcheck disable=SC2206
PROFILE_ARGS=(${PROFILE_ARGS_STR})

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    "${PROFILE_ARGS[@]}" \
    "$@"
