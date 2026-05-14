#!/usr/bin/env bash
# One-shot module runner.  Usage:
#   watod run <module> <bag-or-bag-id> [chunk_id]
# Runs the requested module in a fresh container, with artifacts/bags bind-mounted.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"

if [[ $# -lt 2 ]]; then
    echo "Usage: watod run <module> [--bag] <bag-or-bag-id> [extra-flags]" >&2
    echo "       watod run <module> <subcommand> [--bag] <bag-or-bag-id> [extra-flags]" >&2
    exit 64
fi

MODULE="$1"
shift 1

# If the next argument is a known CLI subcommand (e.g. "viz", "reduce"),
# pass it as-is rather than treating it as the bag path.
SUBCMD="run"
case "${1:-}" in
    viz|reduce) SUBCMD="$1"; shift 1 ;;
esac

# Accept both positional and flag-style bag argument:
#   watod run <module> <bag>
#   watod run <module> --bag <bag>
if [[ "${1:-}" == "--bag" ]]; then
    shift 1
fi
BAG="${1:?bag argument required}"
shift 1
EXTRA_ARGS=("$@")

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

# Use the dev service (with source bind-mounts) when the module is active in
# dev mode, so `watod run` picks up live source changes without a rebuild.
# shellcheck disable=SC2206
SELECTED_SERVICES=(${SELECTED_SERVICES_STR:-})
for s in "${SELECTED_SERVICES[@]}"; do
    if [[ "${s}" == "${TARGET}_dev" ]]; then
        SERVICE="${TARGET}_dev"
        break
    fi
done

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})

ARGS=("${SUBCMD}" --bag "${BAG}" "${EXTRA_ARGS[@]}")

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    --profile "${TARGET}" \
    run --rm "${SERVICE}" python3 -m "${PKG}" "${ARGS[@]}"
