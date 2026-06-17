# Semantic lifting — occlusion-aware 2D→3D label assignment from masks + metric depth.
# CPU-only: purely NumPy/projection math, no GPU required.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/semantic_lifting /ws/src/semantic_lifting

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# System libs + the stable Python utility wheels via apt.  Ubuntu's mirrors
# are fast and resumable; these versions are tolerant enough that open3d's
# resolver accepts them in-place without re-pulling from PyPI.
#   libgl1 + libglib2.0-0  — Open3D headless rendering.
#   python3-numpy/scipy    — UniLiPs Eq.1 minimum_filter (visibility.py).
#   python3-pil            — mask PNG load (io.py).
#   python3-pydantic       — config + artifact schemas.
#   python3-fsspec/click/yaml — pipeline glue.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
        python3-numpy python3-scipy python3-pil \
        python3-pydantic python3-fsspec python3-click python3-yaml \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# pyarrow + open3d aren't packaged for Ubuntu.  open3d is ~430 MiB; uv's
# default 30s HTTP timeout kills the download on slow links before it can
# complete, with no resume.
ENV UV_HTTP_TIMEOUT=900
RUN uv pip install --system --break-system-packages pyarrow open3d
