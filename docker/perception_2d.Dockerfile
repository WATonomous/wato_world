# Perception 2D v2 — Florence-2 discovery + SAM 3.1 text-prompted segmentation
# + Depth Anything V2 + SAM 3.1 video tracker + DINOv2 ReID + x-cam merge.
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

# Generous HTTP timeout — torch's CUDA wheel set is multi-GB and pypi.nvidia.com
# is frequently slow.  Default uv timeout is 30s which is far too short here.
ENV UV_HTTP_TIMEOUT=900

# Common light deps (CPU-only).
RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml pillow

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

# HuggingFace stack — required by:
#   - Florence-2 (loaded from HF Hub via transformers + trust_remote_code).
#     Note: Florence-2 is NOT on PyPI or GitHub as a pip package; it ships
#     custom code inside the HF Hub model repo and is fetched at runtime.
#   - CLIP text embeddings (phrase_dedup.py).
# einops + timm are required by Florence-2's custom DaViT vision encoder
# and by Depth-Anything-V2.
RUN uv pip install --system --break-system-packages \
        transformers accelerate huggingface_hub safetensors \
        einops timm

# SAM 3.1 video predictor (sam3_tracker.py) + image predictor (segmenter.py).
# SAM-style models need hydra-core + iopath for config loading.
# Licenses: SAM3 ("SAM License").
RUN uv pip install --system --break-system-packages \
        hydra-core iopath tqdm \
        git+https://github.com/facebookresearch/sam3

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
