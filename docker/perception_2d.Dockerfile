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
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Common deps (light).
RUN uv pip install --system --break-system-packages \
        pyarrow numpy pydantic fsspec click pyyaml \
        zarr opencv-python-headless pillow

# Heavy ML deps.
# Licenses: Florence-2 (MIT), SAM3 ("SAM License"), DA-V2 (Apache 2.0),
#           DINOv2 (Apache 2.0), transformers (Apache 2.0).
RUN uv pip install --system --break-system-packages --extra-index-url \
        https://download.pytorch.org/whl/cu124 torch torchvision
RUN uv pip install --system --break-system-packages \
        git+https://github.com/microsoft/Florence-2 \
        git+https://github.com/facebookresearch/sam3 \
        git+https://github.com/DepthAnything/Depth-Anything-V2 \
        transformers accelerate
