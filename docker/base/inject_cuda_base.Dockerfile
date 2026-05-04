# CUDA base injection.  Mirrors wato_monorepo/docker/base/inject_*.Dockerfile.
# Takes a generic external image via ARG GENERIC_IMAGE and adds the wato_world
# Python toolchain on top.
#
# Built and pushed by .github/workflows/build_base_images.yml as
# ${REGISTRY}/base:cuda-12.4 .  For local-only first-time setup, run
# `watod build-base cuda` to build it locally with the same tag.

ARG GENERIC_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04
FROM ${GENERIC_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        git curl ca-certificates build-essential cmake ninja-build pkg-config \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /ws
