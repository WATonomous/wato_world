# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound; MF-MOS (Step A.5) needs torch+CUDA.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned — see the note in docker/ingest.Dockerfile.
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04@sha256:6a12d42e81c19ed7317de06d9b583590e4aed3be352a009254375e2842095d27

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# Record which digest-pinned base this image was built FROM, so
# wato_common.provenance can report it and every artifact manifest
# names the base that produced it.
#
# BASE_IMAGE is declared before the first FROM, which makes it a global
# ARG — global args are NOT in scope inside a build stage until
# re-declared. Without this bare `ARG BASE_IMAGE` the ENV below silently
# expands to an empty string.
ARG BASE_IMAGE
ENV WATO_BASE_IMAGE=${BASE_IMAGE}
ENV UV_HTTP_TIMEOUT=10000
# libeigen3-dev  — required by pypatchworkpp C++ build
# libegl1 libgl1 — required by Open3D's OffscreenRenderer (headless EGL)
# libglib2.0-0 libsm6 libxext6 libxrender1 — required by OpenCV / Open3D headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libegl1 libgl1 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        git \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# MF-MOS source (Multi-Frame Moving Object Segmentation).
# Cloned at build time into /opt/mf_mos so no local checkout or bind-mount
# is required.  _runtime.py adds this path to sys.path on first import.
# License: MIT (SCNU-RISLAB/MF-MOS).
# Pinned to an exact commit: upstream `main` moves, and MF-MOS decides which
# points this component calls "moving" — an unpinned clone silently changes the
# static/dynamic split between rebuilds. Fetch just that commit rather than
# cloning the branch tip. Bump deliberately.
ARG MF_MOS_COMMIT=0c702445a39b978efc107cf7d0a2a33246f857ba
RUN git init /opt/mf_mos \
 && git -C /opt/mf_mos remote add origin https://github.com/SCNU-RISLAB/MF-MOS.git \
 && git -C /opt/mf_mos fetch --depth=1 origin ${MF_MOS_COMMIT} \
 && git -C /opt/mf_mos checkout FETCH_HEAD

# Locked dependency set — see docker/requirements/lidar_preprocessing.txt,
# which also documents the torch drift this lock froze (2.11 + cuda-toolkit,
# diverging from perception_2d's 2.7.1).
#
# Direct intent (the lock file additionally carries the transitive closure):
#   base        pyarrow numpy scipy pydantic fsspec click pyyaml matplotlib tqdm
#   numba       numba (+llvmlite) — JIT for the log-odds ray-casting classifier.
#               Required: log-odds is the only classifier, and
#               classify.process_chunk hard-fails at runtime without it.
#   open3d      point-cloud visualization for stages A, B, D in `watod viz`;
#               skipped gracefully at runtime if absent.
#   patchwork   pypatchworkpp==1.0.4 — ground segmentation. Version matches the
#               tag pinned in wato_monorepo. Needs libeigen3-dev to build, which
#               is installed and purged in the same layer below.
#   torch       cu128 wheels matched to the base image's CUDA 12.8 runtime.
#
# --no-deps installs exactly the locked set and never re-resolves, so the image
# contents are a pure function of this repo's contents.
#
# The uv cache mount is kept: uv has no per-wheel HTTP range resume, so a
# connection reset mid-download on one of the 700MB+ wheels (torch, triton,
# nvidia-cusparselt-cu12...) discards that wheel's partial bytes. The cache
# preserves every wheel that finished cleanly, so the retry loop only re-fetches
# the one that crashed. Up to 10 attempts.
# Kept in the image (not deleted) so wato_common.provenance can hash it at
# runtime — see the note in docker/ingest.Dockerfile.
RUN mkdir -p /opt/watonomous
COPY docker/requirements/lidar_preprocessing.txt /opt/watonomous/requirements.lock.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    apt-get update && apt-get install -y --no-install-recommends libeigen3-dev \
 && n=0; until uv pip install --system --break-system-packages --no-deps \
            --index-strategy unsafe-first-match \
            -r /opt/watonomous/requirements.lock.txt; do \
        n=$((n+1)); \
        if [ "$n" -ge 10 ]; then echo "dependency install failed after $n attempts" >&2; exit 1; fi; \
        echo "dependency install attempt $n failed, retrying in 5s..." >&2; \
        sleep 5; \
    done \
 && apt-get purge -y libeigen3-dev \
 && apt-get -qq autoremove -y && apt-get -qq clean \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*
