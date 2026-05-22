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
        libgomp1 \
        libegl1 libgl1 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
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
# libeigen3-dev is a build-only dep; install + purge in one layer to avoid bloat.
RUN apt-get update && apt-get install -y --no-install-recommends libeigen3-dev \
    && uv pip install --system --break-system-packages pypatchworkpp==1.0.4 \
    && apt-get purge -y libeigen3-dev \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# PyTorch matched to CUDA 12.8 base.
# When mf_mos.enabled: false (default) torch is never imported at runtime.
# --no-deps: torch wheels embed direct URLs to pypi.nvidia.com with pinned
# hashes; nvidia republishes those files in-place, breaking pip's hash check.
# The cuda12.8.1-cudnn-runtime base provides CUDA/cuDNN at system paths so
# the nvidia-* Python wrappers are not needed. Pure-Python torch deps are
# installed explicitly below (jinja2/typing-extensions/fsspec already in base).
RUN uv pip install --system --break-system-packages \
        --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<3" \
    && uv pip install --system --break-system-packages \
        filelock sympy networkx jinja2 typing-extensions

# MF-MOS runtime deps (numba + tqdm already installed above).
RUN uv pip install --system --break-system-packages \
        pytorch-lightning pyquaternion easydict
