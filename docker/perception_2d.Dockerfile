# Perception 2D — 2D foundation pass.
# GroundingDINO / YOLO-World detection + SAM 2 + DEVA + DINOv2 ReID + x-cam merge.
# Heaviest GPU component by far — budget several hundred GPU-hours per hour of bag.
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
ENV UV_HTTP_TIMEOUT=10000

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Common deps (light).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        pyarrow numpy pydantic fsspec click pyyaml \
        zarr opencv-python-headless pillow

# PyTorch + matched NVIDIA CUDA libs. cu128 index matches the runtime base
# image's CUDA 12.8 toolkit.
#
# Retry loop mirrors lidar_preprocessing: uv has no per-wheel HTTP range resume,
# so a connection reset mid-download discards partial bytes. The cache mount
# preserves any wheel that finished cleanly, so a retry only re-fetches the one
# that crashed. Loop up to 10 attempts.
RUN --mount=type=cache,target=/root/.cache/uv \
    n=0; until uv pip install --system --break-system-packages \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            'torch>=2.7,<3' torchvision xformers; do \
        n=$((n+1)); \
        if [ "$n" -ge 10 ]; then echo "torch install failed after $n attempts" >&2; exit 1; fi; \
        echo "torch install attempt $n failed, retrying in 5s..." >&2; \
        sleep 5; \
    done

# SAM2 — segment anything v2.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        git+https://github.com/facebookresearch/sam2

# GroundingDINO — open-vocabulary text-prompted detector.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        git+https://github.com/IDEA-Research/GroundingDINO

# DINOv2 is pulled via torch.hub at runtime — no pip install needed.
# Pre-warm the hub cache by uncommenting the line below if you want the weights
# baked into the image (avoids a first-run download inside the container):
# RUN python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')"
