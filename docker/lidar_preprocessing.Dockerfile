# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 libusb-1.0-0 \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        open3d

# Patchwork++ (ground segmentation) — build from source when filling this in.
# RUN git clone https://github.com/url-kaist/patchwork-plusplus /tmp/patchworkpp \
#  && cd /tmp/patchworkpp && pip install --break-system-packages .
