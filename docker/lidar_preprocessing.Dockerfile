# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# libeigen3-dev  — required by pypatchworkpp C++ build
# libegl1 libgl1 — required by Open3D's OffscreenRenderer (headless EGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libeigen3-dev \
        libegl1 libgl1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        matplotlib tqdm

# Numba JIT for the log-odds ray-casting classifier (Step B).  Without this
# numba>=0.59 install, classify.process_chunk hard-fails at runtime when
# classification_method=log_odds (the default).  Pulls llvmlite automatically.
RUN uv pip install --system --break-system-packages 'numba>=0.59'

# Open3D for point-cloud visualization (stages A, B, D in `watod viz`).
# ~80 MB wheel; skipped gracefully at runtime if absent.
RUN uv pip install --system --break-system-packages open3d

# Patchwork++ Python bindings (v1.0.4 matches the monorepo's pinned tag).
# Requires libeigen3-dev (added above) and cmake (present in base image).
RUN uv pip install --system --break-system-packages pypatchworkpp==1.0.4
