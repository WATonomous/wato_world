#!/usr/bin/env bash
# watod-compose.sh - Handles all docker compose operations with preprocessing
# Usage: watod-compose.sh <command> [--pre-profiles <p>...] [--all-profiles <p>...] [--compose-files <f>...] [extra-args...]

set -euo pipefail

run_docker_compose() {
  DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} \
    COMPOSE_BAKE=${COMPOSE_BAKE:-true} \
    docker compose \
      --env-file "${WATO_WORLD_DIR}/modules/.env" \
      "$@"
}

WATO_WORLD_DIR="${WATO_WORLD_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WATO_WORLD_DIR"

declare -a PRE_PROFILES=()
declare -a ALL_PROFILES=()
declare -a CUSTOM_COMPOSE_FILES=()
declare -a EXTRA_COMPOSE_ARGS=()

if [[ $# -lt 1 ]]; then
  echo "Error: Missing compose command" >&2
  echo "Usage: watod-compose.sh <command> [--pre-profiles <p>...] [--all-profiles <p>...] [--compose-files <f>...] [extra-args...]" >&2
  exit 1
fi

COMPOSE_CMD=("$1")
shift

while [[ $# -gt 0 ]]; do
  case $1 in
    --pre-profiles)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        PRE_PROFILES+=("$1")
        shift
      done
      ;;
    --all-profiles)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ALL_PROFILES+=("$1")
        shift
      done
      ;;
    --compose-files)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CUSTOM_COMPOSE_FILES+=("-f" "$1")
        shift
      done
      ;;
    *)
      EXTRA_COMPOSE_ARGS+=("$1")
      shift
      ;;
  esac
done

# Assemble default compose files; include GPU overrides when GPU_AVAILABLE=true.
declare -a DEFAULT_COMPOSE_FILES=("-f" "modules/docker-compose.yaml")
if [[ "${GPU_AVAILABLE:-false}" == "true" ]]; then
  DEFAULT_COMPOSE_FILES+=("-f" "modules/docker-compose.gpu.yaml")
fi
DEFAULT_COMPOSE_FILES+=("-f" "modules/docker-compose.dev.yaml")

if [[ ${#CUSTOM_COMPOSE_FILES[@]} -gt 0 ]]; then
  COMPOSE_FILES=("${CUSTOM_COMPOSE_FILES[@]}")
else
  COMPOSE_FILES=("${DEFAULT_COMPOSE_FILES[@]}")
fi

# On `build`, pull base images for active modules.
if [[ "${COMPOSE_CMD[0]}" == "build" ]]; then
  echo "Pulling base images for active modules..."
  declare -a active_dockerfiles=()
  for p in "${PRE_PROFILES[@]}"; do
    module_name="${p%_pre}"
    dockerfile="docker/${module_name}.Dockerfile"
    if [[ -f "$dockerfile" ]]; then
      active_dockerfiles+=("$dockerfile")
    fi
  done

  if [[ ${#active_dockerfiles[@]} -gt 0 ]]; then
    # shellcheck disable=SC1091
    set -a; source "modules/.env"; set +a
    base_images=$(grep -h "^ARG BASE_IMAGE=" "${active_dockerfiles[@]}" 2>/dev/null | \
                  sed 's/ARG BASE_IMAGE=//' | \
                  envsubst | \
                  sort -u)
    if [[ -n "$base_images" ]]; then
      while IFS= read -r base_image; do
        [[ -z "$base_image" ]] && continue
        echo "  Pulling $base_image..."
        docker pull "$base_image" 2>/dev/null || \
          echo "    (Skipped — using cached version or run 'watod build-base' to seed locally)"
      done <<< "$base_images"
    fi
  fi
fi

# Convert profile names to --profile flags.
declare -a PRE_PROFILE_FLAGS=()
for p in "${PRE_PROFILES[@]}"; do
  PRE_PROFILE_FLAGS+=("--profile" "$p")
done

declare -a ALL_PROFILE_FLAGS=()
for p in "${ALL_PROFILES[@]}"; do
  ALL_PROFILE_FLAGS+=("--profile" "$p")
done

# Pre-build phase: source + dependencies stages.
if [[ "${COMPOSE_CMD[0]}" == "build" && ${#PRE_PROFILES[@]} -gt 0 ]]; then
  echo "RUNNING PRE-BUILD"
  run_docker_compose "${COMPOSE_FILES[@]}" "${PRE_PROFILE_FLAGS[@]}" build "${EXTRA_COMPOSE_ARGS[@]}"
  if [[ -n ${CI:-} || -n ${GITHUB_ACTIONS:-} ]]; then
    echo "CI detected: Pushing PRE-BUILD images to registry..."
    run_docker_compose "${COMPOSE_FILES[@]}" "${PRE_PROFILE_FLAGS[@]}" push
  fi
fi

# Force detached mode for `up`.
if [[ "${COMPOSE_CMD[0]}" == "up" && ! " ${EXTRA_COMPOSE_ARGS[*]:-} " =~ " -d " ]]; then
  EXTRA_COMPOSE_ARGS+=(-d)
fi

if [[ "${COMPOSE_CMD[0]}" == "build" ]]; then
  echo "RUNNING BUILD"
fi

run_docker_compose "${COMPOSE_FILES[@]}" "${ALL_PROFILE_FLAGS[@]}" "${COMPOSE_CMD[@]}" "${EXTRA_COMPOSE_ARGS[@]}"

# In CI, push final images after successful build.
if [[ "${COMPOSE_CMD[0]}" == "build" && ( -n ${CI:-} || -n ${GITHUB_ACTIONS:-} ) && ${#ALL_PROFILES[@]} -gt 0 ]]; then
  echo "CI detected: Pushing final images to registry..."
  run_docker_compose "${COMPOSE_FILES[@]}" "${ALL_PROFILE_FLAGS[@]}" push
fi
