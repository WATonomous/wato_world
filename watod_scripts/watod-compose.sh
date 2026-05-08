#!/usr/bin/env bash
# Generic passthrough to `docker compose` with the compose-file stack and
# profile flags assembled by the watod entrypoint.
#
# On `build`, mirrors wato_monorepo/watod_scripts/watod-compose.sh:
#   1. Pull each active module's BASE_IMAGE from the registry so the per-module
#      FROM resolves locally. If a base isn't on the registry yet (first-time
#      setup before CI has published), `watod build-base` seeds it locally.
#   2. Run `compose build` with the _pre profiles to produce the
#      <comp>:source_<TAG> and <comp>:deps_<TAG> images.
#   3. Run `compose build` with the runtime/dev profiles. The runtime stage
#      is template.Dockerfile, whose `FROM ${MODULE_SOURCE}` and
#      `FROM ${MODULE_DEPS}` resolve to the images produced in step 2.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"
: "${PROFILE_ARGS_STR:?PROFILE_ARGS_STR must be set}"
: "${PRE_PROFILE_ARGS_STR:=}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})
# shellcheck disable=SC2206
PRE_PROFILE_ARGS=(${PRE_PROFILE_ARGS_STR})
# shellcheck disable=SC2206
PROFILE_ARGS=(${PROFILE_ARGS_STR})

run_compose() {
    DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} \
        docker compose \
            --env-file "${WATO_WORLD_DIR}/modules/.env" \
            "${COMPOSE_FILES[@]}" \
            "$@"
}

COMPOSE_CMD="${1:-}"

# ---------------------------------------------------------------------------
# Build path: pull bases, then two-phase compose build.
# ---------------------------------------------------------------------------
if [[ "${COMPOSE_CMD}" == "build" ]]; then
    shift  # consume "build"; remaining $@ is extra build args.
    EXTRA_BUILD_ARGS=("$@")

    # Derive Dockerfiles from the active runtime --profile flags so we know
    # which BASE_IMAGE values to pre-pull.
    declare -a active_dockerfiles=()
    for ((i = 0; i < ${#PROFILE_ARGS[@]}; i++)); do
        if [[ "${PROFILE_ARGS[i]}" == "--profile" ]]; then
            module_name="${PROFILE_ARGS[i+1]%_dev}"
            dockerfile="${WATO_WORLD_DIR}/docker/${module_name}.Dockerfile"
            [[ -f "${dockerfile}" ]] && active_dockerfiles+=("${dockerfile}")
        fi
    done

    if [[ ${#active_dockerfiles[@]} -gt 0 ]]; then
        echo "Pulling base images for active modules..."
        # shellcheck disable=SC1091
        set -a; source "${WATO_WORLD_DIR}/modules/.env"; set +a

        base_images=$(grep -h '^ARG BASE_IMAGE=' "${active_dockerfiles[@]}" 2>/dev/null \
                      | sed 's/ARG BASE_IMAGE=//' \
                      | envsubst \
                      | sort -u)

        if [[ -n "${base_images}" ]]; then
            while IFS= read -r base_image; do
                [[ -z "${base_image}" ]] && continue
                echo "  Pulling ${base_image}..."
                docker pull "${base_image}" 2>/dev/null \
                    || echo "    (Skipped — using cached version, offline, or run 'watod build-base' to seed locally)"
            done <<< "${base_images}"
        fi
    fi

    # Phase 1: build the source + dependencies layers (per-component).
    if [[ ${#PRE_PROFILE_ARGS[@]} -gt 0 ]]; then
        echo "[watod build] Phase 1/2 — building source + deps layers"
        run_compose "${PRE_PROFILE_ARGS[@]}" build "${EXTRA_BUILD_ARGS[@]}"
    fi

    # Phase 2: build runtime (template.Dockerfile) and dev variants.
    echo "[watod build] Phase 2/2 — building runtime / dev images via template"
    exec_status=0
    run_compose "${PROFILE_ARGS[@]}" build "${EXTRA_BUILD_ARGS[@]}" || exec_status=$?
    exit ${exec_status}
fi

# ---------------------------------------------------------------------------
# All other commands: simple passthrough with runtime/dev profiles only.
# (_pre services don't run — they're build-only.)
# ---------------------------------------------------------------------------
exec_status=0
run_compose "${PROFILE_ARGS[@]}" "$@" || exec_status=$?
exit ${exec_status}
