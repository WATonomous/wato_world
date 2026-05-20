#!/usr/bin/env bash
# Pre-fetch torch + CUDA wheels for the lidar_preprocessing Docker build.
#
# The deps stage needs torch==2.2.0+cu121 plus ~2.3 GB of nvidia-* CUDA
# wheels. Downloading them inside Docker is fragile on flaky networks
# because a single mid-stream connection reset aborts the entire RUN.
# This script downloads the same wheel set onto the host once; the
# Dockerfile then COPYs the directory in and installs with --find-links.
#
# Host matches the deps image (Ubuntu 24.04, glibc 2.39, Python 3.12),
# so pip picks the same wheels Docker would.
#
# Re-runnable: pip skips wheels already present in --dest.
#
# Usage:
#   scripts/prefetch_torch_wheels.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST="docker/wheels/torch"
mkdir -p "$DEST"

ATTEMPTS=20
for i in $(seq 1 "$ATTEMPTS"); do
    echo "=== Attempt $i/$ATTEMPTS — pip download torch==2.2.0+cu121 → $DEST ==="
    if pip download \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --dest "$DEST" \
        --only-binary=:all: \
        --retries 5 \
        --timeout 300 \
        'torch==2.2.0+cu121'; then
        echo
        echo "=== Done. Wheels in $DEST: ==="
        ls -lh "$DEST"/*.whl
        exit 0
    fi
    echo "Attempt $i/$ATTEMPTS failed. Sleeping 5s before retry." >&2
    sleep 5
done

echo "ERROR: pip download failed after $ATTEMPTS attempts." >&2
exit 1
