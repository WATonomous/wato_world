#!/usr/bin/env bash
# One-shot module runner.  Usage:
#   watod run <module> <bag-or-bag-id> [chunk_id]
# Runs the requested module in a fresh container, with artifacts/bags bind-mounted.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"

if [[ $# -lt 2 ]]; then
    echo "Usage: watod run <module> <bag-or-bag-id> [chunk_id]" >&2
    exit 64
fi

MODULE="$1"
BAG="$2"
CHUNK_ID="${3:-}"

normalize_module() {
    echo "$1"
}

TARGET="$(normalize_module "${MODULE}")"

case "${TARGET}" in
    ingest)               SERVICE="ingest";               PKG="wato_ingest"  ;;
    perception_2d)        SERVICE="perception_2d";        PKG="wato_perception_2d"  ;;
    lidar_preprocessing)  SERVICE="lidar_preprocessing";  PKG="wato_lidar_preprocessing"  ;;
    proposal_generation)  SERVICE="proposal_generation";  PKG="wato_proposal_generation"  ;;
    tracking)             SERVICE="tracking";             PKG="wato_tracking"  ;;
    label_refinement)     SERVICE="label_refinement";     PKG="wato_label_refinement"  ;;
    open_vocab_discovery) SERVICE="open_vocab_discovery"; PKG="wato_open_vocab_discovery" ;;
    student_training)     SERVICE="student_training";     PKG="wato_student_training"  ;;
    *)
        echo "Unknown module: ${MODULE}" >&2
        exit 64
        ;;
esac

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})

ARGS=(run --bag "${BAG}")
if [[ -n "${CHUNK_ID}" ]]; then
    ARGS+=(--chunk "${CHUNK_ID}")
fi

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    --profile "${TARGET}" \
    run --rm "${SERVICE}" python -m "${PKG}" "${ARGS[@]}"
