#!/usr/bin/env bash
# Run pytest inside a module's dev container.  Usage:
#   watod test [module] [pytest_args...]

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})

MODULE="${1:-}"
shift || true

if [[ -z "${MODULE}" ]]; then
    echo "Usage: watod test <module> [pytest_args...]" >&2
    exit 64
fi

normalize_module() {
    echo "$1"
}

TARGET="$(normalize_module "${MODULE}")"
case "${TARGET}" in
    ingest|perception_2d|lidar_preprocessing|proposal_generation|tracking|label_refinement|open_vocab_discovery|student_training) ;;
    *)
    echo "Unknown module: ${MODULE}" >&2
    exit 64
    ;;
esac
SERVICE="${TARGET}_dev"

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    --profile "${TARGET}_dev" \
    exec "${SERVICE}" pytest /ws/src/"${TARGET}"/tests "$@"
