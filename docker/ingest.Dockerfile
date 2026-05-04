# Ingest — rosbag decoding and synchronization artifacts.
# CPU-only.  Depends on rosbag2 Python bindings.

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
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy
ENV PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH}

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml pillow

# ---------------------------------------------------------------------------
FROM dependencies AS build
COPY --from=source /ws /ws
RUN uv pip install --system --break-system-packages --no-deps \
        -e /ws/src/common -e /ws/src/ingest

# ---------------------------------------------------------------------------
FROM build AS deploy
WORKDIR /ws
ENTRYPOINT ["python", "-m", "wato_ingest"]
CMD ["--help"]

# ---------------------------------------------------------------------------
FROM build AS develop
ARG USERNAME=wato
ARG USER_UID=1000
ARG USER_GID=1000
ARG CLAUDE_CODE=false
RUN groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} 2>/dev/null || true \
 && apt-get update && apt-get install -y --no-install-recommends sudo less vim tmux \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && rm -rf /var/lib/apt/lists/*
RUN uv pip install --system --break-system-packages pytest pytest-cov ruff black ipython
USER ${USERNAME}
WORKDIR /ws
CMD ["sleep", "infinity"]
