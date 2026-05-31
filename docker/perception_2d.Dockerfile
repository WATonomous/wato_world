# Perception 2D — SAM 3.1 multiplex concept-video tracker (detect+segment+track)
# + Depth Anything V2 + DINOv2 ReID + cross-camera merge.
# Heaviest GPU component — budget several hundred GPU-hours per hour of bag.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/perception_2d /ws/src/perception_2d

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# Base image already provides: libgl1, libglib2.0-0, libsm6, libxext6,
# libxrender1, ffmpeg (see docker/base/inject_cuda_base.Dockerfile).
# python3-tk — matplotlib's TkAgg backend, so `watod run perception_2d viz`
# can pop the interactive depth viewer (see src/.../viz.py). ~tiny.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Generous HTTP timeout — torch's CUDA wheel set is multi-GB and pypi.nvidia.com
# is frequently slow.  Default uv timeout is 30s which is far too short here.
ENV UV_HTTP_TIMEOUT=900

# Common light deps (CPU-only). numpy pinned <2 — the `sam3` package requires it.
RUN uv pip install --system --break-system-packages \
        pyarrow "numpy<2" scipy pydantic fsspec click pyyaml pillow

# PyTorch matched to base CUDA 12.8.1.
#
# Pinned to 2.7.x deliberately: torch 2.10+ adds a runtime dep on the
# `cuda-toolkit` PyPI meta-wheel (multi-GB; pulled from pypi.nvidia.com).
# The base image already ships the CUDA 12.8.1 runtime + cuDNN, so that
# wheel is pure bloat for our use case and the download routinely times
# out during builds.  2.7.x has cu128 wheels without that dependency.
#
# Even on 2.7.1, the nvidia-*-cu12 sub-wheels still come from
# pypi.nvidia.com which is flaky.  Pre-fetch the closure on the host with
# src/perception_2d/scripts/fetch_wheels.sh and install offline from the
# bind-mounted dir so build-time network failures stop being a problem.
RUN --mount=type=bind,source=data/wheels/perception_2d,target=/wheels \
    uv pip install --system --break-system-packages \
        --no-index --find-links /wheels \
        "torch==2.7.1" "torchvision==0.22.1"

# HuggingFace Hub (checkpoint downloads for SAM 3.1 + Depth-Anything-V2) plus
# einops/timm/safetensors used by Depth-Anything-V2's DPT head.
RUN uv pip install --system --break-system-packages \
        huggingface_hub safetensors einops timm

# SAM 3.1 multiplex tracker via Meta's official `sam3` package (vendored at the
# repo root). The published checkpoint sam3.1_multiplex.pt (facebook/sam3.1)
# loads ONLY through this code — SAM 3.1 has no HuggingFace Transformers
# integration. Pulls light deps (ftfy/regex/iopath/numpy<2); config use_fa3=False
# avoids the FlashAttention-3 dependency. Checkpoint is pre-fetched by
# scripts/fetch_models.py into HF_HOME. License: SAM License.
# Runtime deps the multiplex build path pulls (beyond sam3's light core deps):
# hydra/omegaconf (config instantiation), open_clip_torch (text encoder),
# pycocotools (mask RLE), psutil, and scikit-image/learn used by helpers.
RUN uv pip install --system --break-system-packages \
        hydra-core omegaconf open_clip_torch psutil pycocotools \
        scikit-image scikit-learn
COPY sam3 /tmp/sam3
RUN uv pip install --system --break-system-packages /tmp/sam3 \
    && rm -rf /tmp/sam3

# Depth Anything V2 (depth.py).  The upstream Meta/ByteDance repo isn't
# pip-installable (no pyproject.toml / setup.py); use the community PyPI
# wrapper instead — same module layout, `from depth_anything_v2.dpt
# import DepthAnythingV2` works unchanged.
# opencv (cv2.resize inside infer_image) and matplotlib (colormap utilities)
# are direct module deps.
# License: Apache 2.0.
RUN uv pip install --system --break-system-packages \
        opencv-python-headless matplotlib \
        depth-anything-v2

# DINOv2 (reid.py) is loaded via torch.hub.load("facebookresearch/dinov2",
# ...) at runtime — no pip install required, but xformers gives memory-
# efficient attention for the ViT-L model.
# xformers must match the pinned torch version (2.7.x).
RUN uv pip install --system --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "xformers==0.0.31"
