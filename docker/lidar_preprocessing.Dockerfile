# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# Torch + CUDA wheels are large (>700 MB); extend the uv download timeout.
ENV UV_HTTP_TIMEOUT=2000
# libeigen3-dev  — required by pypatchworkpp C++ build
# libegl1 libgl1 — required by Open3D's OffscreenRenderer (headless EGL)
# libglib2.0-0 libsm6 libxext6 libxrender1 — required by OpenCV / Open3D headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libeigen3-dev \
        libegl1 libgl1 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        matplotlib

# Open3D for point-cloud visualization (stages A, B, D in `watod viz`).
# ~80 MB wheel; skipped gracefully at runtime if absent.
RUN uv pip install --system --break-system-packages open3d

# Patchwork++ Python bindings (v1.0.4 matches the monorepo's pinned tag).
# Requires libeigen3-dev (added above) and cmake (present in base image).
RUN uv pip install --system --break-system-packages pypatchworkpp==1.0.4

# PyTorch matched to CUDA 12.8 base.
# When mf_mos.enabled: false (default) torch is never imported at runtime.
# Use pip (not uv) here: pip retries on connection resets; uv does not.
# Use --index-url (not --extra-index-url) so the cu128 index is primary:
# torch+cu128 is a self-contained wheel — no separate nvidia-* downloads from pypi.nvidia.com.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        --retries 10 \
        --timeout 300 \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<3"

# MF-MOS runtime deps.
RUN uv pip install --system --break-system-packages \
        pytorch-lightning numba pyquaternion easydict tqdm
