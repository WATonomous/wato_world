# detection_3d — learned 3D boxes from BEVFusion (camera + LiDAR), the model
# the car runs (wato_monorepo src/perception/bevfusion): WATonomous's fork of
# NVIDIA's CUDA-BEVFusion on TensorRT, called from Python through the fork's
# own pybind11 module (libpybev). GPU only.
#
# Defines `vendor` (compiles the fork), `source` and `dependencies` build
# stages. The full image (build / deploy / develop) is composed by
# docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned, the same CUDA base as perception_2d and label_refinement. The
# tag is mutable — CI republishes it on every push to docker/base/** — and a
# silent rebuild on a different base is a different labeler. Bump the digest
# deliberately (docker buildx imagetools inspect <tag>).
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda12.8.1-cudnn-runtime-ubuntu24.04@sha256:6a12d42e81c19ed7317de06d9b583590e4aed3be352a009254375e2842095d27

# TensorRT 10, as the car installs it (libnvinfer10 from NVIDIA's CUDA apt
# repo; the car leaves it unpinned). The +cuda12.9 build, not the +cuda13.x
# one apt would pick by default: the base ships the CUDA 12.8 runtime.
ARG TENSORRT_VERSION=10.16.1.11-1+cuda12.9

# The fork at the commit its master pointed to on 2026-11-09. The car's image
# clones master unpinned; this pins it. libspconv.so comes from git LFS at this
# commit.
ARG WATO_CUDA_BEVFUSION_COMMIT=08ec4db80ec768e9410c11e032e2a26dd24ded52

# GPU compute capability the kernels are compiled for: the car's default (Ada,
# RTX 40-series), as in wato_monorepo's template.Dockerfile. PTX runs on this
# capability and newer, never older: set 86 for an RTX 30-series.
ARG CUDASM=89

# ---------------------------------------------------------------------------
# TensorRT runtime, shared by the vendor build and the runtime image.
FROM ${BASE_IMAGE} AS tensorrt
ARG TENSORRT_VERSION
# python3-libnvinfer builds the TensorRT engines from the model's ONNX files
# (wato_detection_3d.engines); libprotobuf32t64 is libbevfusion_core's ONNX
# parser runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer10=${TENSORRT_VERSION} \
        libnvinfer-plugin10=${TENSORRT_VERSION} \
        libnvonnxparsers10=${TENSORRT_VERSION} \
        libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
        python3-libnvinfer=${TENSORRT_VERSION} \
        libprotobuf32t64=3.21.12-8.2ubuntu0.4 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Compile the fork: libbevfusion_core, libcustom_layernorm (the TensorRT plugin
# head.bbox needs) and libpybev, as wato_monorepo's cuda_bevfusion_vendor
# builds the first two.
FROM tensorrt AS vendor
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TENSORRT_VERSION
ARG WATO_CUDA_BEVFUSION_COMMIT
ARG CUDASM
# TensorRT's headers only: the fork includes NvInfer.h and NvInferPlugin.h and
# links libnvinfer / libnvinfer_plugin, which the tensorrt stage installed.
# libnvinfer-dev would add 2.4 GB of static libraries nothing links.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8=12.8.93-1 \
        cuda-cudart-dev-12-8=12.8.90-1 \
        cuda-driver-dev-12-8=12.8.90-1 \
        libcublas-dev-12-8=12.8.4.1-1 \
        libnvinfer-headers-dev=${TENSORRT_VERSION} \
        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
        protobuf-compiler=3.21.12-8.2ubuntu0.4 \
        libprotobuf-dev=3.21.12-8.2ubuntu0.4 \
        git-lfs=3.4.1-1ubuntu0.4 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/tensorrt_link \
    && ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 /opt/tensorrt_link/libnvinfer.so \
    && ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 /opt/tensorrt_link/libnvinfer_plugin.so

# Only the submodules the libraries compile against: pybind11, dlpack, stb.
# The fork's other submodules (the PyTorch training repo, cuPCL) aren't built.
#
# pybind11 is moved from the fork's pin (2.11.0.dev1, May 2023) to v2.13.6:
# NumPy 2 changed its C ABI and pybind11 supports it from 2.12. Built against
# the old one, libpybev reads this image's NumPy 2 arrays with the old layout
# and returns garbage (every value one constant).
ARG PYBIND11_COMMIT=a2e59f0e7065404b44dfe92a28aca47ba1378dc4
WORKDIR /opt/src
RUN git clone https://github.com/WATonomous/wato-cuda-bevfusion.git \
    && cd wato-cuda-bevfusion \
    && git checkout --detach "${WATO_CUDA_BEVFUSION_COMMIT}" \
    && git submodule update --init dependencies/pybind11 dependencies/dlpack dependencies/stb \
    && git -C dependencies/pybind11 fetch --depth 1 origin "${PYBIND11_COMMIT}" \
    && git -C dependencies/pybind11 checkout --detach "${PYBIND11_COMMIT}" \
    && git lfs install --local \
    && git lfs pull --include "libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda12.8/libspconv.so" \
    && file_size=$(stat -c %s libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda12.8/libspconv.so) \
    && if [ "${file_size}" -lt 1000000 ]; then echo "libspconv.so is still an LFS pointer" >&2; exit 1; fi

WORKDIR /opt/src/wato-cuda-bevfusion/CUDA-BEVFusion
# The upstream Python binding fixes camera buffers/geometry at six. Match
# their size to the exported camera engine, and reject mismatched inputs.
COPY docker/patches/bevfusion-camera-count.patch /tmp/bevfusion-camera-count.patch
RUN git apply --check /tmp/bevfusion-camera-count.patch \
    && git apply /tmp/bevfusion-camera-count.patch
# Regenerate the ONNX protobuf sources with this image's protoc, so they match
# the libprotobuf they link against.
#
# libspconv.so has a hard ELF NEEDED on libcuda.so.1, the driver library, which
# a build has no GPU for: link against the CUDA stub (the car's vendor
# CMakeLists does the same). The real driver is mounted at run time.
RUN bash src/onnx/make_pb.sh \
    && mkdir -p /tmp/cuda_stubs \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda_stubs/libcuda.so.1 \
    && export CUDASM="${CUDASM}" SPCONV_CUDA_VERSION=12.8 \
        CUDA_Inc=/usr/local/cuda/include CUDA_Lib=/usr/local/cuda/lib64 \
        TensorRT_Inc=/usr/include/x86_64-linux-gnu TensorRT_Lib=/opt/tensorrt_link \
        USE_Python=ON \
        Python_Inc="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
        Python_Lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
        Python_Soname= \
    && stub_flags="-L/tmp/cuda_stubs -L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/tmp/cuda_stubs" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_SHARED_LINKER_FLAGS=${stub_flags}" "-DCMAKE_EXE_LINKER_FLAGS=${stub_flags}" \
    && cmake --build build --target bevfusion_core custom_layernorm pybev -j"$(nproc)" \
    && mkdir -p /opt/bevfusion/lib \
    && cp build/libbevfusion_core.so build/libcustom_layernorm.so build/libpybev.so \
        ../libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda12.8/libspconv.so \
        /opt/bevfusion/lib/

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/detection_3d /ws/src/detection_3d

# ---------------------------------------------------------------------------
FROM tensorrt AS dependencies
# Record which digest-pinned base this image was built FROM, so
# wato_common.provenance can report it and every artifact manifest names the
# base that produced it. BASE_IMAGE is a global ARG (declared before the first
# FROM): without this bare re-declaration the ENV expands to empty.
ARG BASE_IMAGE
ENV WATO_BASE_IMAGE=${BASE_IMAGE}
ARG TENSORRT_VERSION
ARG WATO_CUDA_BEVFUSION_COMMIT
ARG CUDASM

COPY --from=vendor /opt/bevfusion /opt/bevfusion
# libpybev is imported from /opt/bevfusion/lib, which also holds the libraries
# it loads (BEVFusion's core dlopens libcustom_layernorm.so by name). The build
# pins are restated as environment so each manifest records them
# (wato_detection_3d.pipeline).
ENV PYTHONPATH=/opt/bevfusion/lib \
    LD_LIBRARY_PATH=/opt/bevfusion/lib \
    WATO_CUDA_BEVFUSION_COMMIT=${WATO_CUDA_BEVFUSION_COMMIT} \
    WATO_TENSORRT_VERSION=${TENSORRT_VERSION} \
    WATO_CUDASM=${CUDASM} \
    WATO_BEVFUSION_CAMERA_COUNT_ABI=1

# Locked dependency set — see docker/requirements/detection_3d.txt for the
# rationale and the regeneration procedure.
#
# Direct intent (the lock file additionally carries the transitive closure):
#     numpy scipy pillow pyarrow pydantic fsspec click pyyaml
#   numpy/scipy         — point and box geometry, pose interpolation
#                         (wato_common).
#   pillow              — decoding and resampling camera images.
#   pyarrow             — parquet artifacts (wato_common.io.parquet_io).
#   pydantic            — config + artifact schemas.
#   fsspec/click/pyyaml — pipeline glue.
# TensorRT's Python bindings come from apt (python3-libnvinfer, above), matched
# to the TensorRT libraries libbevfusion_core links.
#
# --no-deps installs exactly the locked set and never re-resolves. The lock is
# kept at /opt/watonomous/requirements.lock.txt: wato_common.provenance hashes
# it into every manifest.
RUN mkdir -p /opt/watonomous
COPY docker/requirements/detection_3d.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps \
        -r /opt/watonomous/requirements.lock.txt
