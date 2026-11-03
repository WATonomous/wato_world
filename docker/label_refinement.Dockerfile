# Label refinement — training-free track refinement: one box per parked
# vehicle from its points over every frame, one size and a smoothed path per
# moving track. CPU-only (NumPy/SciPy). SLF, fitting shapes to camera masks
# per track, will need torch and the CUDA base.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned, the same base as tracking. The `base:cpu-ubuntu24.04` tag is
# mutable — CI republishes it on every push to docker/base/** — and a silent
# rebuild on a different base is a different labeler. Bump the digest
# deliberately (docker buildx imagetools inspect <tag>).
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04@sha256:92a0d985245262d3bf9014630814af5c0c91ff586f4e01290f90801eb6568727

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

# Every Python dep comes from PyPI in one resolve.
#
# Locked dependency set — see docker/requirements/label_refinement.txt for the
# rationale and the regeneration procedure.
#
# Direct intent (the lock file additionally carries the transitive closure):
#     numpy scipy pyarrow pydantic fsspec click pyyaml
#   numpy/scipy         — box fits, point groups, local line fits.
#   pyarrow             — parquet artifacts (wato_common.io.parquet_io).
#   pydantic            — config + artifact schemas.
#   fsspec/click/pyyaml — pipeline glue.
#
# --no-deps installs exactly the locked set and never re-resolves. The lock is
# kept at /opt/watonomous/requirements.lock.txt: wato_common.provenance hashes
# it into every manifest.
RUN mkdir -p /opt/watonomous
COPY docker/requirements/label_refinement.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps \
        -r /opt/watonomous/requirements.lock.txt
