# Perception 2D — GroundingDINO detector + SAM2 video tracker (segment+track)
# + Depth Anything V2 + DINOv2 ReID.
# Heaviest GPU component — budget several hundred GPU-hours per hour of bag.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned — see the note in docker/ingest.Dockerfile.
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04@sha256:6a12d42e81c19ed7317de06d9b583590e4aed3be352a009254375e2842095d27

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/perception_2d /ws/src/perception_2d

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
# Record which digest-pinned base this image was built FROM, so
# wato_common.provenance can report it and every artifact manifest
# names the base that produced it.
#
# BASE_IMAGE is declared before the first FROM, which makes it a global
# ARG — global args are NOT in scope inside a build stage until
# re-declared. Without this bare `ARG BASE_IMAGE` the ENV below silently
# expands to an empty string.
ARG BASE_IMAGE
ENV WATO_BASE_IMAGE=${BASE_IMAGE}
# Base image already provides: libgl1, libglib2.0-0, libsm6, libxext6,
# libxrender1, ffmpeg (see docker/base/inject_cuda_base.Dockerfile).
# python3-tk — matplotlib's TkAgg backend, so `watod run perception_2d viz`
# can pop the interactive depth viewer (see src/.../viz.py). ~tiny.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Generous HTTP timeout — torch's CUDA wheel set is multi-GB and pypi.nvidia.com
# is frequently slow.  Default uv timeout is 30s which is far too short here.
ENV UV_HTTP_TIMEOUT=900

# Locked dependency set — see docker/requirements/perception_2d.txt, which also
# documents the drift this lock froze (numpy, opencv, gradio, transformers).
#
# Direct intent (the lock file additionally carries the transitive closure):
#   base        pyarrow numpy scipy pydantic fsspec click pyyaml pillow
#   torch       torch==2.7.1 torchvision==0.22.1  (cu128, matched to the base
#               image's CUDA 12.8.1; deliberately NOT 2.10+, which adds a
#               runtime dep on the multi-GB `cuda-toolkit` PyPI meta-wheel the
#               base image already makes redundant)
#   detector    huggingface_hub transformers safetensors einops timm
#               (GroundingDINO via AutoModelForZeroShotObjectDetection, plus
#               Florence-2 discovery; no CUDA custom-op compile, unlike the
#               standalone groundingdino package)
#   depth       depth-anything-v2 opencv-python-headless matplotlib
#   tracker     hydra-core iopath  (+ the `sam2` git install below)
#   reid        xformers==0.0.31   (memory-efficient attention for DINOv2
#               ViT-L, loaded at runtime via torch.hub — must match torch 2.7.x)
#
# --no-deps installs exactly the locked set and never re-resolves, so the image
# contents are a pure function of this repo's contents. This matters more here
# than anywhere else in the pipeline: these four models jointly determine every
# 2D label the pipeline emits.
# Kept in the image (not deleted) so wato_common.provenance can hash it at
# runtime — see the note in docker/ingest.Dockerfile.
RUN mkdir -p /opt/watonomous
COPY docker/requirements/perception_2d.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps \
            --index-strategy unsafe-first-match -r /opt/watonomous/requirements.lock.txt

# SAM2.1 via Meta's official `sam2` package. SAM2VideoPredictor.from_pretrained
# (facebook/sam2.1-hiera-large) instantiates it with the package's bundled hydra
# config. Installed separately from the lock file above because it builds from a
# git checkout and its setup.py imports torch — so torch must already be present
# in site-packages, which --no-deps cannot guarantee for an unordered install.
#
# Pinned to an exact commit: upstream has no PyPI release and `main` moves, so
# an unpinned install silently changes the tracker between rebuilds. This is the
# revision that was verified against torch 2.7.1. License: Apache 2.0.
ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
RUN uv pip install --system --break-system-packages --no-deps \
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}"
