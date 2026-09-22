#!/usr/bin/env bash
# fetch_wheels.sh — pre-download the torch wheel closure for perception_2d.
#
# pypi.nvidia.com routinely times out / drops TLS on long downloads from
# many networks, which kills `uv pip install torch` inside the Docker build.
# This script downloads every wheel torch needs once into
# data/wheels/perception_2d/ so the build can install offline via a
# bind-mount + --no-index --find-links.
#
# pip download is idempotent: wheels already on disk are skipped. Re-run
# until the set converges if a download dies mid-batch. If a wheel is
# fundamentally unreachable from this network, fetch it by hand (browser,
# different network, mirror) and drop it into data/wheels/perception_2d/.
#
# Usage:
#   src/perception_2d/scripts/fetch_wheels.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="${REPO_ROOT}/data/wheels/perception_2d"

# Must match the pins in docker/perception_2d.Dockerfile.
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"

mkdir -p "${DEST}"

echo "Downloading torch==${TORCH_VERSION} wheel closure → ${DEST}"
echo "(pip download skips existing wheels — safe to re-run.)"
echo

# Run inside python:3.12-slim/linux/amd64 so resolved wheels match the
# Ubuntu 24.04 + python3.12 target image, regardless of host OS/arch.
docker run --rm \
    --platform linux/amd64 \
    -v "${DEST}:/wheels" \
    python:3.12-slim \
    bash -c "
        pip install --quiet --upgrade pip &&
        pip download \
            --dest /wheels \
            --retries 10 \
            --timeout 120 \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            --only-binary=:all: \
            'torch==${TORCH_VERSION}' 'torchvision==${TORCHVISION_VERSION}'
    "

count=$(find "${DEST}" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')
size=$(du -sh "${DEST}" | cut -f1)
echo
echo "Done. ${count} wheels (${size}) in ${DEST}"
