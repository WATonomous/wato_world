#!/usr/bin/env bash
# Generic passthrough to `docker compose` with the compose-file stack and
# profile flags assembled by the watod entrypoint.
#
# On `build`, mirrors wato_monorepo/watod_scripts/watod-compose.sh: pull each
# active module's BASE_IMAGE from the registry first so the per-module FROM
# resolves locally.  If a base isn't on the registry yet (first-time setup
# before CI has published), falls back silently and `docker compose build`
# will fail with a clear "image not found" error — run `watod build-base`
# once to seed the bases locally.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"
: "${COMPOSE_FILES_STR:?COMPOSE_FILES_STR must be set}"
: "${PROFILE_ARGS_STR:?PROFILE_ARGS_STR must be set}"

# shellcheck disable=SC2206
COMPOSE_FILES=(${COMPOSE_FILES_STR})
# shellcheck disable=SC2206
PROFILE_ARGS=(${PROFILE_ARGS_STR})

COMPOSE_CMD="${1:-}"

# ---------------------------------------------------------------------------
# On `build`, pull base images referenced by the active modules' Dockerfiles.
# Same approach as wato_monorepo/watod_scripts/watod-compose.sh.
# ---------------------------------------------------------------------------
if [[ "${COMPOSE_CMD}" == "build" ]]; then
    # Derive Dockerfiles from the active --profile flags.
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
        # Source .env so $REGISTRY etc. are available for ARG expansion below.
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
fi

exec docker compose \
    --env-file "${WATO_WORLD_DIR}/modules/.env" \
    "${COMPOSE_FILES[@]}" \
    "${PROFILE_ARGS[@]}" \
    "$@"
