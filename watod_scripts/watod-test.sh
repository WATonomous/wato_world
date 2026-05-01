#!/usr/bin/env bash
# Run pytest inside a component's dev container.  Usage:
#   watod test [component] [pytest_args...]

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})

COMPONENT="${1:-}"
shift || true

if [[ -z "${COMPONENT}" ]]; then
    echo "Usage: watod test <component> [pytest_args...]" >&2
    exit 64
fi

normalize_component() {
    echo "$1"
}

TARGET="$(normalize_component "${COMPONENT}")"
case "${TARGET}" in
    ingest|perception_2d|lidar_preprocessing|proposal_generation|tracking|label_refinement|open_vocab_discovery|student_training) ;;
    *)
    echo "Unknown component: ${COMPONENT}" >&2
    exit 64
    ;;
esac
SERVICE="${TARGET}_dev"

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    --profile "${TARGET}_dev" --profile "infra" \
    exec "${SERVICE}" pytest /ws/src/"${TARGET}"/tests "$@"
