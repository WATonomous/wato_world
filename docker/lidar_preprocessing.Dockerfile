# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction, MapMOS inference (when cfg.mapmos.enabled).
#
# Single CUDA-based deps stage with everything pre-installed: torch +
# MinkowskiEngine + MapMOS sit alongside the geometry-only deps so the
# stub stays runnable today and the real Step-3 inference doesn't need
# a separate image. When cfg.mapmos.enabled=false, the heavy ML deps
# are present but never imported (lazy `import torch` in mapmos/weights.py).
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
# dependencies — geometry-only deps PLUS torch + MinkowskiEngine + MapMOS.
#
# Geometry-only deps:
#   libeigen3-dev  — pypatchworkpp C++ build
#   libegl1 libgl1 — Open3D OffscreenRenderer (headless EGL)
#
# MapMOS deps (pinned versions verified against PRBonn/MapMOS @ 8947300,
# 2025-09-11. PRBonn's pyproject.toml doesn't pin torch/MinkowskiEngine
# — their README defers MinkowskiEngine to a manual install — so the
# pin choices below are best-effort known-good combos):
#   - torch 2.2.0+cu121     lowest cp312-compatible torch (Ubuntu 24.04
#                           base ships Python 3.12; torch 2.1 and below
#                           only have cp38/39/310/311 wheels). cu121
#                           wheels are forward-compatible with the
#                           CUDA 12.8 host runtime.
#   - pytorch_lightning 2.2.5   matches torch 2.2's API surface and
#                           satisfies PRBonn's >=1.6.4 floor.
#   - MinkowskiEngine commit 02fc6080…   NVIDIA/master HEAD; repo mostly
#                           unmaintained since 2023. May not compile
#                           cleanly against torch 2.2+ — if this RUN
#                           fails, see remediation notes below the call.
#   - mapmos commit 8947300…   the SHA cited in mapmos/inference.py:run_model.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Persistent uv cache between builds. Combined with the pre-fetched torch
# wheels under docker/wheels/torch (see scripts/prefetch_torch_wheels.sh),
# this keeps the heavy ML deps install resilient: pre-fetched wheels avoid
# downloading torch + ~2 GB of nvidia-* wheels from the registry at all,
# and the uv cache absorbs any incidental re-downloads (pytorch_lightning,
# mapmos transitive deps) across rebuilds.
ENV UV_HTTP_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libeigen3-dev \
        libegl1 libgl1 \
        libopenblas-dev \
        ninja-build \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# numpy is pinned <2 because torch 2.2's prebuilt C extensions were
# compiled against the NumPy 1.x ABI. With NumPy 2.x in the env, importing
# torch emits an _ARRAY_API warning and any C extension that crossed the
# torch<->numpy boundary at build time may crash. MinkowskiEngine's
# setup.py imports torch during the build, so a numpy 2.x env makes the
# extension compile fail before it starts. The next torch bump (2.3+) was
# rebuilt against numpy 2 and can drop this pin.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages \
        pyarrow 'numpy<2' scipy pydantic fsspec click pyyaml \
        matplotlib tqdm

# Numba JIT for the log-odds ray-casting classifier (Step B). Without
# numba>=0.59, classify.process_chunk hard-fails at runtime when
# classification_method=log_odds (the default). Pulls llvmlite automatically.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages 'numba>=0.59'

# Open3D for point-cloud visualization (stages A, B, D in `watod viz`).
# ~80 MB wheel; skipped gracefully at runtime if absent.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages open3d

# Patchwork++ Python bindings (v1.0.4 matches wato_monorepo's pinned tag).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages pypatchworkpp==1.0.4

# ---- MapMOS dependency chain ----
# Torch must come BEFORE MinkowskiEngine and mapmos because both depend
# on torch's headers at build time.
#
# Ubuntu 24.04 base ships Python 3.12. Torch 2.1.x and below have no
# cp312 wheels (cp38/39/310/311 only). 2.2.0 is the lowest version with
# Python 3.12 support. If a future bump is needed (e.g., MinkowskiEngine
# compatibility), 2.3 / 2.4 / 2.5 also have cp312 wheels available.
#
# Torch + the nvidia-cu12-* wheels total ~2.5 GB. To avoid re-downloading
# them on every cache miss (and to survive flaky upstream connections that
# kill mid-stream), pre-fetch the wheels onto the host via
# `scripts/prefetch_torch_wheels.sh` and COPY them in here. `--find-links`
# makes uv prefer the local wheels; `--extra-index-url` is kept as a
# fallback so the build still works if the local dir is empty.
COPY docker/wheels/torch /tmp/wheels/torch
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages \
        --find-links /tmp/wheels/torch \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        'torch==2.2.0+cu121' \
    && rm -rf /tmp/wheels

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages 'pytorch_lightning==2.2.5'

# MinkowskiEngine — fragile compile. TORCH_CUDA_ARCH_LIST covers
# Ampere / Ada / Hopper consumer + datacenter GPUs. openblas backend
# avoids the MKL dep.
#
# Common failure modes against torch 2.2+ on Python 3.12:
#   1. c10::TensorImpl API changes broke 02fc6080's headers.
#      Remediation: try a community fork that patches for torch 2.2+,
#      e.g. git+https://github.com/shenh10/MinkowskiEngine.git@<sha>
#      (search for "MinkowskiEngine torch 2.x" on GitHub for known forks).
#   2. setup.py uses `setuptools.distutils` which is removed in Python 3.12.
#      Remediation: ensure setuptools<60 is installed first, or use
#      SETUPTOOLS_USE_DISTUTILS=stdlib environment override.
#   3. --global-option deprecated in newer pip; uv may also reject it.
#      Remediation: drop the flag (defaults to openblas if MKL absent),
#      or use --config-settings instead.
ENV MAX_COMPILATION_THREADS=2
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# MinkowskiEngine + MapMOS sources are pre-cloned on the host (see
# scripts/prefetch_git_sources.sh) to avoid github.com connection resets
# during the build. `uv pip install <local-path>` is equivalent to
# `uv pip install git+https://...@<sha>` once the source tree is on disk.
#
# Note: uv (unlike legacy pip) does not accept `--global-option`. The flag
# was previously used to force `--blas=openblas` in MinkowskiEngine's
# setup.py argv. Dropped here: MKL is not installed and `libopenblas-dev`
# is (see the apt-get step above), so MinkowskiEngine's setup.py BLAS
# auto-detection lands on openblas without an explicit flag.
COPY docker/sources/MinkowskiEngine /tmp/sources/MinkowskiEngine
# Two setup.py patches needed to build under Python 3.12 / numpy 1.26:
#   1. Line 123 self-uninstalls via system pip (`pip uninstall MinkowskiEngine
#      -y`). Ubuntu 24.04's system pip refuses under PEP 668. Nothing to
#      uninstall in a fresh deps layer anyway — comment it out.
#   2. Line 201 imports `numpy.distutils.system_info` to auto-detect BLAS.
#      `numpy.distutils` requires Python's stdlib `distutils`, which Python
#      3.12 removed (PEP 632). We default `BLAS="openblas"` so the import
#      is never reached; the apt step above installs `libopenblas-dev`, so
#      linking succeeds.
RUN sed -i 's|^run_command("pip", "uninstall"|# patched out (PEP 668): &|' \
        /tmp/sources/MinkowskiEngine/setup.py \
 && sed -i 's|^BLAS, argv = _argparse("--blas", argv, False)$|&\nBLAS = BLAS or "openblas"  # patched: bypass numpy.distutils on Python 3.12|' \
        /tmp/sources/MinkowskiEngine/setup.py
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-build-isolation \
        /tmp/sources/MinkowskiEngine

# MapMOS — pinned to the commit cited in mapmos/inference.py:run_model
# (mapmos_net.py:38-82 is the verbatim source of the voxel↔point mapping).
COPY docker/sources/MapMOS /tmp/sources/MapMOS
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-build-isolation \
        /tmp/sources/MapMOS \
    && rm -rf /tmp/sources
