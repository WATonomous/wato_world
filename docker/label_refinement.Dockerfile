# Label refinement — per-track refinement: one box per parked vehicle from its
# points over every frame, one size and a smoothed path per moving track, and
# SLF (a shape fit to the track's camera masks and LiDAR points, PyTorch on the
# GPU; a CPU path for tests and GPU-less hosts).
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned, the same CUDA base as perception_2d. The tag is mutable — CI
# republishes it on every push to docker/base/** — and a silent rebuild on a
# different base is a different labeler. Bump the digest deliberately
# (docker buildx imagetools inspect <tag>).
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04@sha256:6a12d42e81c19ed7317de06d9b583590e4aed3be352a009254375e2842095d27

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/label_refinement /ws/src/label_refinement

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# Record which digest-pinned base this image was built FROM, so
# wato_common.provenance can report it and every artifact manifest names the
# base that produced it. BASE_IMAGE is a global ARG (declared before the first
# FROM): without this bare re-declaration the ENV expands to empty.
ARG BASE_IMAGE
ENV WATO_BASE_IMAGE=${BASE_IMAGE}

# Generous HTTP timeout: torch's CUDA wheel set is multi-GB.
ENV UV_HTTP_TIMEOUT=900

# Locked dependency set — see docker/requirements/label_refinement.txt for the
# rationale and the regeneration procedure.
#
# Direct intent (the lock file additionally carries the transitive closure):
#     numpy scipy pyarrow pydantic fsspec click pyyaml pillow torch==2.7.1
#   numpy/scipy         — box fits, point groups, local line fits.
#   torch               — SLF's differentiable fit; cu128, perception_2d's
#                         version, matched to the base's CUDA 12.8.1.
#   pillow              — reading perception_2d's mask PNGs
#                         (wato_common.artifacts.masklets).
#   pyarrow             — parquet artifacts (wato_common.io.parquet_io).
#   pydantic            — config + artifact schemas.
#   fsspec/click/pyyaml — pipeline glue.
#
# --no-deps installs exactly the locked set and never re-resolves;
# unsafe-first-match takes each pinned version from the first index that has
# it (torch's +cu128 wheels from download.pytorch.org, the rest from PyPI; see
# the lock's INDEX RESOLUTION note). The lock is kept at
# /opt/watonomous/requirements.lock.txt: wato_common.provenance hashes it into
# every manifest.
RUN mkdir -p /opt/watonomous
COPY docker/requirements/label_refinement.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps \
        --index-strategy unsafe-first-match -r /opt/watonomous/requirements.lock.txt
