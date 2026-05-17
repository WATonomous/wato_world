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
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Common deps (light).
RUN uv pip install --system --break-system-packages \
        pyarrow numpy pydantic fsspec click pyyaml \
        zarr opencv-python-headless pillow

# Torch + torchvision: required by both Depth Anything V2 and ultralytics
# (YOLO-World).  cu124 wheels match the CUDA 12.8 runtime base image.
RUN uv pip install --system --break-system-packages --extra-index-url \
        https://download.pytorch.org/whl/cu124 torch torchvision

# Depth Anything V2 (relative+metric monocular depth) and ultralytics
# (YOLO-World v2-L).  Both consume torch.  Weights live under MODELS_ROOT,
# not the image — fetch via `watod fetch-models`.
RUN uv pip install --system --break-system-packages \
        git+https://github.com/DepthAnything/Depth-Anything-V2.git \
        ultralytics

# Heavy ML deps for the existing perception_2d code paths (SAM2 + GroundingDINO
# + DINOv2).  Uncomment once you're ready to put the full perception_2d image
# into rotation — DA V2 / YOLO-World alone don't need these.
# RUN uv pip install --system --break-system-packages \
#         git+https://github.com/facebookresearch/sam2 \
#         git+https://github.com/IDEA-Research/GroundingDINO \
#         git+https://github.com/facebookresearch/dinov2
