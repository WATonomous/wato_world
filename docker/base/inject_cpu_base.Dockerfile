# CPU base injection.  Mirrors wato_monorepo/docker/base/inject_*.Dockerfile:
# takes a generic external image via ARG GENERIC_IMAGE and adds the wato_world
# Python toolchain (python3 + uv) on top.
#
# Built and pushed by .github/workflows/build_base_images.yml as
# ${REGISTRY}/base:cpu-ubuntu24.04 .  For local-only first-time setup, run
# `watod build-base cpu` to build it locally with the same tag.

# Reproducibility: GENERIC_IMAGE is digest-pinned and uv is version-pinned.
# Both the `ubuntu:24.04` tag and the uv install script are mutable
# upstream, so
# without these pins two builds of this file months apart produce different
# bases — and every component image, and therefore every label, inherits that
# difference. Bump both deliberately.
ARG GENERIC_IMAGE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
FROM ${GENERIC_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        git curl ca-certificates build-essential cmake ninja-build pkg-config \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.11.9
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /ws
