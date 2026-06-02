# Ingest — rosbag decoding and synchronization artifacts. CPU-only.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile via
# MODULE_SOURCE / MODULE_DEPS build args (see modules/docker-compose.yaml).
# Mirrors the per-component shape used in wato_monorepo/docker/*.Dockerfile.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/ingest /ws/src/ingest

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies

# ROS 2 Jazzy apt repository (for rosbag2_py / rclpy).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common locales gnupg2 \
    && locale-gen en_US en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-jazzy-rosbag2 ros-jazzy-rosbag2-py \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs ros-jazzy-nav-msgs \
    && apt-get -qq autoremove -y && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

ENV ROS_DISTRO=jazzy
ENV PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH}

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml pillow tqdm
