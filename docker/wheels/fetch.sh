#!/usr/bin/env bash
# Download torch + matched NVIDIA CUDA wheels into this directory.
# Safe to interrupt and re-run — completed wheels persist; pip download skips
# files already present.  Resilient to connection resets via the outer loop.
#
# Target environment: Python 3.12, Linux x86_64, CUDA 12.8 wheels.  Adjust the
# --python-version / --platform / --extra-index-url flags below if those change.

set -euo pipefail

WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_SPEC="torch>=2.7,<3"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "Downloading $TORCH_SPEC + deps into $WHEELS_DIR"
echo "  (re-runs skip already-downloaded wheels; Ctrl-C to abort)"
echo

# No --platform / --python-version — host is Python 3.12 + x86_64 + glibc 2.39
# which matches the docker target (ubuntu24.04 + python3.12).  Letting pip
# resolve against the host env avoids strict-tag mismatches between torch's
# manylinux_2_28 wheels and the older manylinux variants some nvidia-* wheels
# use.  --only-binary still prevents source-build attempts.
ATTEMPT=0
until python3 -m pip download \
        --dest "$WHEELS_DIR" \
        --index-url "$PYTORCH_INDEX" \
        --only-binary=:all: \
        "$TORCH_SPEC"; do
    ATTEMPT=$((ATTEMPT + 1))
    echo
    echo "[fetch.sh] pip download failed (attempt $ATTEMPT) — retrying in 5s..."
    sleep 5
done

echo
echo "Done.  Wheels in $WHEELS_DIR:"
ls -1sh "$WHEELS_DIR"/*.whl 2>/dev/null || echo "  (no wheels found?)"
