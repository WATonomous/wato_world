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
        libgomp1 libeigen3-dev \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Note: Open3D used to be installed here for reduce.py's voxel_down_sample;
# replaced with a numpy voxel-snap (see reduce.py) so the apt deps for
# Open3D's GL/USB stack and the ~80 MB wheel are no longer needed.
RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml

# Patchwork++ Python bindings (v1.0.4 matches the monorepo's pinned tag).
# Requires libeigen3-dev (added above) and cmake (present in base image).
RUN uv pip install --system --break-system-packages pypatchworkpp==1.0.4
