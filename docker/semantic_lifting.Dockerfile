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
# Every Python dep comes from PyPI in one resolve — the same set ingest locks.
# Don't take numpy/scipy from apt (python3-numpy/-scipy): any PyPI wheel that
# depends on numpy installs numpy 2.x into /usr/local, shadowing apt's numpy
# 1.26, and apt's scipy (built against numpy 1.x) then fails to import
# scipy.ndimage.
#   numpy/scipy         — UniLiPs Eq.1 minimum_filter (visibility.py).
#   pillow              — mask PNG load (io.py).
#   pyarrow             — parquet artifacts (wato_common.io.parquet_io).
#   pydantic            — config + artifact schemas.
#   fsspec/click/pyyaml — pipeline glue.
RUN uv pip install --system --break-system-packages \
        numpy scipy pillow pyarrow pydantic fsspec click pyyaml
