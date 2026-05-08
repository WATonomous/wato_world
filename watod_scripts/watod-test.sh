#!/usr/bin/env bash
# Build a module's dev image, then run pytest in a one-shot container.
# Usage:
#   watod test [module] [pytest_args...]

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"

declare -a COMPOSE_FILES=(
    "-f" "${WATO_WORLD_DIR}/modules/docker-compose.yaml"
    "-f" "${WATO_WORLD_DIR}/modules/docker-compose.dev.yaml"
)

ALL_MODULES=(
    ingest
    perception_2d
    lidar_preprocessing
    proposal_generation
    tracking
    label_refinement
    open_vocab_discovery
    student_training
)

is_module() {
    local candidate="$1"
    local module
    for module in "${ALL_MODULES[@]}"; do
        [[ "${candidate}" == "${module}" ]] && return 0
    done
    return 1
}

module_arg="${1:-}"
declare -a targets=()

if [[ -n "${module_arg}" ]] && is_module "${module_arg}"; then
    targets+=("${module_arg}")
    shift
elif [[ -n "${module_arg}" ]] && [[ "${module_arg}" != -* ]]; then
    echo "Unknown module: ${module_arg}" >&2
    exit 64
else
    # Test every active service from watod. Strip _dev for active dev modules.
    # shellcheck disable=SC2206
    selected_services=(${SELECTED_SERVICES_STR:-})
    for service in "${selected_services[@]}"; do
        target="${service%_dev}"
        if is_module "${target}"; then
            targets+=("${target}")
        fi
    done
fi

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "Usage: watod test [module] [pytest_args...]" >&2
    exit 64
fi

run_docker_compose() {
    DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} \
    COMPOSE_BAKE=${COMPOSE_BAKE:-true} \
        docker compose \
            --env-file "${WATO_WORLD_DIR}/modules/.env" \
            "${COMPOSE_FILES[@]}" \
            "$@"
}

run_module_tests() {
    local target="$1"
    shift

    echo "[watod test] Building ${target}_dev"
    bash "${WATO_WORLD_DIR}/watod_scripts/watod-compose.sh" build \
        --pre-profiles "${target}_pre" \
        --all-profiles "${target}_dev"

    echo "[watod test] Running pytest for ${target}"
    run_docker_compose \
        --profile "${target}_dev" \
        run --rm "${target}_dev" pytest "/ws/src/${target}/tests" "$@"
}

for target in "${targets[@]}"; do
    run_module_tests "${target}" "$@"
done
