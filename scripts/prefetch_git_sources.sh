#!/usr/bin/env bash
# Pre-fetch the git sources the lidar_preprocessing deps stage installs
# (MinkowskiEngine, MapMOS). Mirrors scripts/prefetch_torch_wheels.sh:
# fetching inside Docker is fragile because github.com sometimes resets
# the connection mid-stream on flaky networks, killing the whole RUN step.
# Fetching on the host with retries makes the build reproducibly cacheable;
# the Dockerfile then COPYs the tree in and `uv pip install`s from the
# local path.
#
# Strategy: download GitHub's per-commit archive tarball with `curl -C -`
# (resumes partial downloads using HTTP Range). This is more robust than
# `git fetch --depth 1 <sha>` for two reasons:
#   1. Not every GitHub repo enables uploadpack.allowAnySHA1InWant; without
#      it, fetching an arbitrary SHA fails with "not our ref".
#   2. git transport doesn't resume — a mid-pack connection reset wastes
#      all progress. HTTP Range on the tarball survives resets cleanly.
#
# Re-runnable: a `.commit_sha` marker in each checkout dir is the
# idempotency check.
#
# Usage:
#   scripts/prefetch_git_sources.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST="docker/sources"
mkdir -p "$DEST"

fetch_archive_with_retry() {
    local repo=$1   # e.g. NVIDIA/MinkowskiEngine
    local sha=$2
    local target=$3
    local marker="$target/.commit_sha"

    if [[ -f "$marker" && "$(cat "$marker")" == "$sha" ]]; then
        echo "=== $target: already at $sha (skipping) ==="
        return 0
    fi

    local archive_url="https://codeload.github.com/${repo}/tar.gz/${sha}"
    local tarball="${target}.tar.gz"
    local tmp="${target}.tmp"

    local attempts=20
    for i in $(seq 1 "$attempts"); do
        echo "=== Attempt $i/$attempts — fetch ${repo} @ ${sha} ==="
        # -L: follow redirects (codeload may redirect)
        # -C -: resume partial download from byte offset
        # --retry 5: curl-level retry for transient HTTP errors
        # --retry-delay 5: 5s between curl-level retries
        # --connect-timeout 30: bail on stalled handshakes
        if curl -fL -C - \
              --retry 5 --retry-delay 5 \
              --connect-timeout 30 \
              -o "$tarball" \
              "$archive_url"; then
            echo "=== Extracting → $target ==="
            rm -rf "$tmp" "$target"
            mkdir -p "$tmp"
            if tar -xzf "$tarball" -C "$tmp" --strip-components=1; then
                echo "$sha" > "$tmp/.commit_sha"
                mv "$tmp" "$target"
                rm -f "$tarball"
                echo "=== $target: extracted ${sha} ==="
                return 0
            fi
            echo "Extraction failed — corrupted tarball, removing and retrying" >&2
            rm -rf "$tmp" "$tarball"
        fi
        echo "Attempt $i/$attempts failed. Sleeping 5s before retry." >&2
        sleep 5
    done
    echo "ERROR: failed to fetch ${repo} @ ${sha} after $attempts attempts" >&2
    return 1
}

# MinkowskiEngine: NVIDIA/master HEAD (verified 2026-05-19). An earlier
# version of this script and the Dockerfile pinned 02fc60802a2e0fcf…2aebd,
# which does not exist on the upstream repo — looks like a transcription
# error of master HEAD (same first 7 chars, then diverges). Keep this
# value in sync with the comment block around the MinkowskiEngine install
# in docker/lidar_preprocessing.Dockerfile.
fetch_archive_with_retry \
    NVIDIA/MinkowskiEngine \
    02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
    "$DEST/MinkowskiEngine"

fetch_archive_with_retry \
    PRBonn/MapMOS \
    8947300698c61257ddb1e1e9f927382f0c0a0bac \
    "$DEST/MapMOS"

echo
echo "=== Done. Sources in $DEST: ==="
ls -la "$DEST"
