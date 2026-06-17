# Perception 2D — GroundingDINO detector + SAM2 video tracker (segment+track)
# + Depth Anything V2 + DINOv2 ReID.
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
# Pinned to 2.7.x: torch 2.10+ adds a runtime dep on the `cuda-toolkit`
# PyPI meta-wheel (multi-GB; pulled from pypi.nvidia.com). The base image
# already ships the CUDA 12.8.1 runtime + cuDNN, so 2.7.x cu128 wheels
# (no cuda-toolkit dep) are the right choice. UV_HTTP_TIMEOUT=900 handles
# pypi.nvidia.com latency for the nvidia-*-cu12 sub-wheels.
RUN uv pip install --system --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.1" "torchvision==0.22.1"

# HuggingFace Hub + Transformers. Transformers hosts the GroundingDINO detector
# (AutoModelForZeroShotObjectDetection, IDEA-Research/grounding-dino-base) and
# the optional Florence-2 discovery backend — no CUDA custom-op compile, unlike
# the standalone groundingdino package. einops/timm/safetensors are also used by
# Depth-Anything-V2's DPT head. Checkpoints are pre-fetched by
# scripts/fetch_models.py into HF_HOME.
RUN uv pip install --system --break-system-packages \
        huggingface_hub transformers safetensors einops timm

# SAM2.1 via Meta's official `sam2` package. SAM2VideoPredictor.from_pretrained
# (facebook/sam2.1-hiera-large) downloads the checkpoint from HF and instantiates
# it with the package's bundled hydra config. The package's setup.py pulls its
# own light deps (hydra-core, iopath); torch is already present and satisfies its
# >=2.5.1 floor, so it is not reinstalled. License: Apache 2.0.
# TODO: pin @<commit> once a known-good sam2 revision is verified against this
# torch (mirrors the sam3 pin we replaced); unpinned tracks main for now.
RUN uv pip install --system --break-system-packages \
        hydra-core iopath \
        git+https://github.com/facebookresearch/sam2.git

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
