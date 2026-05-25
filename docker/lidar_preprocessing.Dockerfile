# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound; MF-MOS (Step A.5) needs torch+CUDA.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.
#
# TORCH INSTALL: torch wheels are 700+ MB and uv has no HTTP range-request
# resume.  On flaky connections, mid-download resets force a full restart and
# the wheel never completes.  Workaround: pre-download all torch wheels on
# the host with `wget -c` (resume-capable) into docker/wheels/, then this
# Dockerfile installs from the local directory — no network access for torch
# during `docker build`.  See docker/wheels/README.md for the wget commands.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
ENV UV_HTTP_TIMEOUT=10000
# libeigen3-dev  — required by pypatchworkpp C++ build
# libegl1 libgl1 — required by Open3D's OffscreenRenderer (headless EGL)
# libglib2.0-0 libsm6 libxext6 libxrender1 — required by OpenCV / Open3D headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libegl1 libgl1 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        matplotlib tqdm

# Numba JIT for the log-odds ray-casting classifier (Step B).  Without this
# numba>=0.59 install, classify.process_chunk hard-fails at runtime when
# classification_method=log_odds (the default).  Pulls llvmlite automatically.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages 'numba>=0.59'

# Open3D for point-cloud visualization (stages A, B, D in `watod viz`).
# ~80 MB wheel; skipped gracefully at runtime if absent.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages open3d

# Patchwork++ Python bindings (v1.0.4 matches the monorepo's pinned tag).
# libeigen3-dev is a build-only dep; install + purge in one layer to avoid bloat.
RUN --mount=type=cache,target=/root/.cache/uv \
    apt-get update && apt-get install -y --no-install-recommends libeigen3-dev \
    && uv pip install --system --break-system-packages pypatchworkpp==1.0.4 \
    && apt-get purge -y libeigen3-dev \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# PyTorch + matched NVIDIA CUDA libs, installed from pre-downloaded wheels.
# See docker/wheels/README.md for the host-side `wget -c` commands to populate
# docker/wheels/ before running `./watod build`.  --find-links points uv at
# the local wheelhouse; --no-index forbids any network access (so this fails
# loudly if a wheel is missing instead of silently re-fetching from pypi).
COPY docker/wheels /tmp/wheels
RUN uv pip install --system --break-system-packages \
        --no-index --find-links /tmp/wheels \
        torch \
    && rm -rf /tmp/wheels
