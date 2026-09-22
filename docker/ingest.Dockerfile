# Ingest — rosbag decoding and synchronization artifacts. CPU-only.
#
# Defines `source` and `dependencies` build stages. The full image
# (build / deploy / develop) is composed by docker/template.Dockerfile via
# MODULE_SOURCE / MODULE_DEPS build args (see modules/docker-compose.yaml).
# Mirrors the per-component shape used in wato_monorepo/docker/*.Dockerfile.

# syntax=docker/dockerfile:1.6
# Digest-pinned. The `base:cpu-ubuntu24.04` tag is mutable — CI republishes it
# on every push to docker/base/**. A tag reference would mean this component
# silently rebuilds on a different base, which for a pipeline whose output is
# training data is a silent change of labeler. Bump the digest deliberately
# when you want the new base (docker buildx imagetools inspect <tag>).
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04@sha256:92a0d985245262d3bf9014630814af5c0c91ff586f4e01290f90801eb6568727

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/ingest /ws/src/ingest

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

# Locked dependency set — see docker/requirements/ingest.txt for the rationale
# and the regeneration procedure.
#
# Direct intent (the lock file additionally carries the transitive closure):
#     pyarrow numpy scipy pydantic fsspec click pyyaml pillow tqdm
#
# --no-deps installs exactly the locked set and never re-resolves, so the image
# contents are a pure function of this repo's contents.
# The lock is KEPT in the image at /opt/watonomous/requirements.lock.txt rather than
# deleted: wato_common.provenance hashes it at runtime so every manifest
# records which dependency set produced the artifact, and it makes a
# running container self-describing (`cat /opt/watonomous/requirements.lock.txt`).
RUN mkdir -p /opt/watonomous
COPY docker/requirements/ingest.txt /opt/watonomous/requirements.lock.txt
RUN uv pip install --system --break-system-packages --no-deps -r /opt/watonomous/requirements.lock.txt
