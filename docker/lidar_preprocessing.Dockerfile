# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound; MF-MOS (Step A.5) needs torch+CUDA.
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
RUN git clone --depth=1 https://github.com/SCNU-RISLAB/MF-MOS.git /opt/mf_mos

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

# PyTorch + matched NVIDIA CUDA libs. cu128 index matches the runtime base
# image's CUDA 12.8 toolkit.
#
# uv has no per-wheel HTTP range resume, so a connection reset mid-download
# on one of the 700MB+ wheels (torch, triton, nvidia-cusparselt-cu12...)
# discards that wheel's partial bytes. The uv cache mount preserves any
# wheel that finished cleanly though, so a retry only re-fetches the one
# that crashed. Loop up to 10 attempts.
RUN --mount=type=cache,target=/root/.cache/uv \
    n=0; until uv pip install --system --break-system-packages \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            'torch>=2.7,<3'; do \
        n=$((n+1)); \
        if [ "$n" -ge 10 ]; then echo "torch install failed after $n attempts" >&2; exit 1; fi; \
        echo "torch install attempt $n failed, retrying in 5s..." >&2; \
        sleep 5; \
    done
