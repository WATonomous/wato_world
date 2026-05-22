# LiDAR preprocessing — MapMOS-capable image.
#
# Why this component diverges from the shared ghcr base:
# MapMOS pulls in MinkowskiEngine 02fc6080 (NVIDIA, last touched 2023),
# which only compiles cleanly against CUDA 12.1-era thrust headers. Pinned
# torch (2.2.0+cu121) also targets CUDA 12.1. So we drop down to the
# upstream NVIDIA CUDA 12.1 devel image and install Python 3.12 on top.
#
# WSL crash fix:
# The previous version OOM'd WSL during MinkowskiEngine compilation because
# `MAX_COMPILATION_THREADS` is NOT the variable torch's BuildExtension reads
# — it reads `MAX_JOBS`. With `MAX_JOBS` unset, nvcc fan-outs to nproc and
# peak RSS is `nproc × ~3 GB × num_archs`. Fixes here:
#   1. MAX_JOBS=1 (set globally; also pinned for MAPMOS_BUILD_JOBS).
#   2. TORCH_CUDA_ARCH_LIST trimmed via build arg (default: 8.6, Ampere).
#   3. NVCC_PREPEND_FLAGS=-Xfatbin=-compress-all to keep temp file size down.
#
# If you have a different GPU, override at build time:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0" ...
# Common values: 7.5 (Turing/RTX 20xx, T4), 8.0 (A100), 8.6 (RTX 30xx),
# 8.9 (RTX 40xx, L4/L40), 9.0 (H100).

# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="8.6"

# ---------------------------------------------------------------------------
# local_base — Python 3.12 + uv on top of the upstream cuda devel image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS local_base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git build-essential cmake ninja-build pkg-config \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
        libgomp1 libeigen3-dev libegl1 libopenblas-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && rm -rf /usr/lib/python3/dist-packages/pkg_resources \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*
# /usr/lib/python3/dist-packages/pkg_resources is the apt-installed
# python3-pkg-resources from Ubuntu 22.04 (setuptools 59.6, pre-PEP 632).
# It's in deadsnakes Python 3.12's sys.path before the pip-installed
# setuptools at /usr/local/lib/python3.12/dist-packages, so any
# pkg_resources import resolves to the old version, which calls
# pkgutil.ImpImporter — removed in Python 3.12. lightning_fabric uses
# pkg_resources.declare_namespace, hitting that crash on the very first
# `import pytorch_lightning`. Removing the apt copy lets the newer
# pip-installed one (installed via `uv pip install setuptools` later)
# win.

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /ws

# ---------------------------------------------------------------------------
FROM local_base AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
# dependencies — geometry deps + torch + MinkowskiEngine + MapMOS.
# ---------------------------------------------------------------------------
FROM local_base AS dependencies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_CUDA_ARCH_LIST
ENV UV_HTTP_TIMEOUT=300 \
    MAX_JOBS=1 \
    NVCC_PREPEND_FLAGS="-Xfatbin=-compress-all" \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ---- Pure-Python deps (one layer, no compilation) ----
# numpy<2 because torch 2.2's C extensions target the NumPy 1.x ABI.
# setuptools<81 is required because:
#   - MinkowskiEngine's legacy setup.py uses setuptools.build_meta as its
#     PEP 517 backend, and deadsnakes' python3.12 doesn't ship setuptools.
#   - setuptools 81 removed the bundled `pkg_resources` module (extracted
#     into a separate distribution). lightning_fabric 2.2 (transitive dep
#     of pytorch_lightning 2.2.5) still does `import pkg_resources` at
#     module load, which would ImportError on bare setuptools >= 81.
#   - We also rm the apt-shipped `python3-pkg-resources` (setuptools 59.6)
#     above so we don't accidentally resolve to that pre-PEP-632 copy.
# scikit-build-core+pybind11+cmake are MapMOS's build backend (installed
# upfront because we use --no-build-isolation later).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages \
        'setuptools<81' wheel \
        'numpy<2' scipy pyarrow pydantic fsspec click pyyaml \
        matplotlib tqdm 'numba>=0.59' \
        pypatchworkpp==1.0.4 \
        scikit-build-core pybind11 cmake

# ---- Open3D (large wheel, often flaky) ----
# Kept separate so its failure doesn't poison the rest. The PyPI wheel is
# ~427 MB and the CDN reliably resets the connection mid-download from
# some networks. We retry up to 3 times with backoff; if it still fails,
# the build proceeds with a clear WARN — open3d only powers the optional
# `watod viz` stages (A/B/D in lidar_preprocessing.viz), and the runtime
# code skips them gracefully when open3d isn't importable. If you need
# viz, pre-fetch the wheel via scripts/ + bind-mount like torch_wheels.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    ( uv pip install --system --break-system-packages open3d \
   || (echo "open3d install attempt 1 failed; retrying in 30s..." >&2 && sleep 30 \
          && uv pip install --system --break-system-packages open3d) \
   || (echo "open3d install attempt 2 failed; retrying in 60s..." >&2 && sleep 60 \
          && uv pip install --system --break-system-packages open3d) ) \
 || echo "WARN: open3d failed all 3 install attempts; viz stages will be unavailable in this image" >&2

# ---- Torch + nvidia-cu12-* wheels (pre-fetched on host) ----
# Bind-mounted from the `torch_wheels` named context to avoid ~2.5 GB of
# downloads + layer bloat. --find-links prefers local; --extra-index-url
# is a fallback when the local dir is empty.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=torch_wheels,target=/tmp/wheels/torch \
    uv pip install --system --break-system-packages \
        --find-links /tmp/wheels/torch \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        'torch==2.2.0+cu121' 'pytorch_lightning==2.2.5'

# ---- MinkowskiEngine ----
# Patches required (see comments in original Dockerfile for rationale):
#   1. setup.py line 123: comment out self-uninstall (PEP 668 refuses).
#   2. setup.py: default BLAS="openblas" to bypass numpy.distutils (removed
#      in Python 3.12 per PEP 632).
#   3. setup.py: force FORCE_CUDA=True (docker build has no GPU; the
#      argparse-based detection silently falls back to CPU_ONLY otherwise).
#   4. src/*.cu, src/*.cuh: thrust >= 2.0 (shipped with CUDA 12.x) split
#      each algorithm into its own header AND stopped transitively
#      pulling in <thrust/execution_policy.h>, so symbols like
#      `thrust::device`, `thrust::remove_if`, `thrust::unique`,
#      `thrust::count_if`, `thrust::for_each`, `thrust::inclusive_scan`,
#      `thrust::reduce_by_key`, `thrust::sequence`, `thrust::transform`
#      are no longer in scope when MinkowskiEngine's .cu files include
#      only e.g. <thrust/sort.h>. We write a single "thrust_compat.h"
#      that pulls in everything MinkowskiEngine touches, and prepend it
#      to every src/ .cu / .cuh that mentions `thrust::`.
#
# MAX_JOBS=1 above caps nvcc parallelism — this is the WSL OOM fix.
# Build into a writable /tmp/ copy because setup.py writes into the tree.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=mink_src,target=/src/MinkowskiEngine \
    set -eux; \
    cp -r /src/MinkowskiEngine /tmp/MinkowskiEngine; \
    cd /tmp/MinkowskiEngine; \
    sed -i 's|^run_command("pip", "uninstall"|# patched out (PEP 668): &|' setup.py; \
    sed -i 's|^BLAS, argv = _argparse("--blas", argv, False)$|&\nBLAS = BLAS or "openblas"  # patched|' setup.py; \
    sed -i 's|^FORCE_CUDA, argv = _argparse("--force_cuda", argv)$|&\nFORCE_CUDA = True  # patched|' setup.py; \
    echo "=== writing src/thrust_compat.h ==="; \
    printf '%s\n' \
        '#pragma once' \
        '// Force-included by every thrust-using .cu/.cuh so we do not rely on the' \
        '// pre-thrust-2.0 transitive includes MinkowskiEngine 02fc6080 assumes.' \
        '// See the comment block above this RUN in the Dockerfile.' \
        '#include <thrust/execution_policy.h>' \
        '#include <thrust/copy.h>' \
        '#include <thrust/count.h>' \
        '#include <thrust/fill.h>' \
        '#include <thrust/for_each.h>' \
        '#include <thrust/reduce.h>' \
        '#include <thrust/remove.h>' \
        '#include <thrust/scan.h>' \
        '#include <thrust/sequence.h>' \
        '#include <thrust/sort.h>' \
        '#include <thrust/transform.h>' \
        '#include <thrust/unique.h>' \
        '#include <thrust/iterator/counting_iterator.h>' \
        '#include <thrust/iterator/transform_iterator.h>' \
        '#include <thrust/iterator/zip_iterator.h>' \
        '#include <thrust/functional.h>' \
        '#include <thrust/tuple.h>' \
        '#include <thrust/pair.h>' \
        '#include <thrust/device_vector.h>' \
        '#include <thrust/host_vector.h>' \
        > src/thrust_compat.h; \
    echo "=== prepending thrust_compat.h to every src/ file using thrust:: ==="; \
    mapfile -t THRUST_FILES < <(grep -rl --include='*.cu' --include='*.cuh' 'thrust::' src); \
    for f in "${THRUST_FILES[@]}"; do \
        sed -i '1i#include "thrust_compat.h"' "$f"; \
    done; \
    echo "Building MinkowskiEngine: arch=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"; \
    uv pip install --system --break-system-packages --no-build-isolation . ; \
    cd / && rm -rf /tmp/MinkowskiEngine

# ---- MapMOS ----
# scikit_build_core writes its CMake build tree inside the source dir, so
# we copy out of the read-only mount before installing.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=mapmos_src,target=/src/MapMOS \
    set -eux; \
    cp -r /src/MapMOS /tmp/MapMOS; \
    uv pip install --system --break-system-packages --no-build-isolation /tmp/MapMOS; \
    rm -rf /tmp/MapMOS
