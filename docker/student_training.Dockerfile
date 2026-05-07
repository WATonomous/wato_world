# Student training — self-trained detector.
# Trains BEVFusion / TransFusion on refined_tracks + camera/lidar inputs.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/student_training /ws/src/student_training

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        wandb tqdm

# Heavy training deps.
# RUN uv pip install --system --break-system-packages --extra-index-url \
#         https://download.pytorch.org/whl/cu124 torch torchvision
# RUN uv pip install --system --break-system-packages spconv-cu124 \
#  && git clone https://github.com/mit-han-lab/bevfusion /opt/bevfusion \
#  && cd /opt/bevfusion && python setup.py develop
