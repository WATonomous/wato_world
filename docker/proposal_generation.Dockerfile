# Proposal generation.
# LiDAR detector ensemble (CenterPoint + TransFusion-L or DSVT) + Segment-Lift-Fit
# + proposal fusion.  GPU.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/proposal_generation /ws/src/proposal_generation

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml

# Heavy 3D-detection deps.  Pick mmdet3d OR OpenPCDet (default: OpenPCDet).
# RUN uv pip install --system --break-system-packages --extra-index-url \
#         https://download.pytorch.org/whl/cu124 torch torchvision
# RUN uv pip install --system --break-system-packages spconv-cu124
# RUN git clone https://github.com/open-mmlab/OpenPCDet /opt/OpenPCDet \
#  && cd /opt/OpenPCDet && python setup.py develop
