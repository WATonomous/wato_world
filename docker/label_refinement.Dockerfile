# Label refinement — track-centric refinement.
# Multimodal LabelFormer: per-track transformer over LiDAR + multi-camera tokens
# + virtual points (Fusion4DAL) + multi-view silhouette consistency loss.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/label_refinement /ws/src/label_refinement

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml

# Heavy refinement deps.
# RUN uv pip install --system --break-system-packages --extra-index-url \
#         https://download.pytorch.org/whl/cu124 torch torchvision xformers
# RUN uv pip install --system --break-system-packages \
#         git+https://github.com/erikwijmans/Pointnet2_PyTorch \
#         git+https://github.com/facebookresearch/pytorch3d
