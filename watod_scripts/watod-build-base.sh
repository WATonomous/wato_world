#!/usr/bin/env bash
# Build base images locally and tag them with the names referenced by every
# component Dockerfile.  Each component's Dockerfile says
#     FROM ${BASE_IMAGE}
# where BASE_IMAGE defaults to ${REGISTRY}/base:cpu-ubuntu24.04 (or the cuda
# variant).  When the registry doesn't have those images yet (i.e. always for
# local-only setups), we build + tag them ourselves so the FROM resolves.
#
# Usage:
#   watod-build-base.sh             # build both flavors
#   watod-build-base.sh cpu         # only CPU
#   watod-build-base.sh cuda        # only CUDA
#   watod-build-base.sh --force     # rebuild even if the tag already exists

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set by the watod entrypoint}"
: "${REGISTRY:=ghcr.io/watonomous/wato_world}"

FORCE=0
WHICH=""
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        cpu|cuda)   WHICH="${WHICH} ${arg}" ;;
        *)          echo "watod-build-base: unknown arg: ${arg}" >&2; exit 64 ;;
    esac
done
[[ -z "${WHICH}" ]] && WHICH="cpu cuda"

build_flavor() {
    local flavor="$1"
    local dockerfile tag
    case "${flavor}" in
        cpu)
            dockerfile="${WATO_WORLD_DIR}/docker/base/inject_cpu_base.Dockerfile"
            tag="${REGISTRY}/base:cpu-ubuntu24.04"
            ;;
        cuda)
            dockerfile="${WATO_WORLD_DIR}/docker/base/inject_cuda_base.Dockerfile"
            tag="${REGISTRY}/base:cuda-12.4"
            ;;
        *)  echo "build_flavor: unknown flavor ${flavor}" >&2; return 1 ;;
    esac

    if [[ ${FORCE} -eq 0 ]] && docker image inspect "${tag}" >/dev/null 2>&1; then
        echo "watod-build-base: ${tag} already present (use --force to rebuild)"
        return 0
    fi

    echo "watod-build-base: building ${tag} from ${dockerfile}"
    docker build \
        -f "${dockerfile}" \
        -t "${tag}" \
        "${WATO_WORLD_DIR}"
}

for flavor in ${WHICH}; do
    build_flavor "${flavor}"
done
