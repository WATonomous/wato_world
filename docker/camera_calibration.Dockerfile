# File-based calibration module, using the existing source/dependencies pattern.
# Koide's NIDCost is compiled into a headless runner; no ROS/GL runtime.
# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04@sha256:92a0d985245262d3bf9014630814af5c0c91ff586f4e01290f90801eb6568727
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/camera_calibration /ws/src/camera_calibration

FROM ${BASE_IMAGE} AS dependencies
ARG BASE_IMAGE
ENV WATO_BASE_IMAGE=${BASE_IMAGE}
ARG KOIDE_COMMIT=02a0dc039f5509708f384be4ff3228e0ae09352d
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get -o Acquire::Retries=3 update && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    build-essential cmake git ca-certificates libeigen3-dev libceres-dev libboost-dev \
    libopencv-core-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git init /opt/watonomous/koide \
 && git -C /opt/watonomous/koide fetch --depth 1 https://github.com/koide3/direct_visual_lidar_calibration.git ${KOIDE_COMMIT} \
 && git -C /opt/watonomous/koide checkout FETCH_HEAD \
 && git -C /opt/watonomous/koide submodule update --init --depth 1 thirdparty/Sophus thirdparty/json
COPY src/camera_calibration/native /opt/watonomous/calibration_native
RUN cmake -S /opt/watonomous/calibration_native -B /opt/watonomous/calibration_build \
      -DKOIDE_SOURCE_DIR=/opt/watonomous/koide -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/watonomous \
 && cmake --build /opt/watonomous/calibration_build -j2 \
 && cmake --install /opt/watonomous/calibration_build \
 && dpkg-query -W > /opt/watonomous/calibration_build/apt-packages.txt
# Same Python intent as semantic_lifting. Frozen lock is generated from the deps image.
COPY docker/requirements/camera_calibration.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps -r /opt/watonomous/requirements.lock.txt
ENV WATO_KOIDE_EXECUTABLE=/opt/watonomous/bin/wato_koide_calibrate
