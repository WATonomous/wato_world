# LiDAR preprocessing.
# Motion compensation, multi-sweep aggregation, static/dynamic decomposition,
# ground extraction.  Mostly CPU-bound.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/lidar_preprocessing /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        open3d

# Patchwork++ (ground segmentation) — build from source when filling this in.
# RUN git clone https://github.com/url-kaist/patchwork-plusplus /tmp/patchworkpp \
#  && cd /tmp/patchworkpp && pip install --break-system-packages .

# ---------------------------------------------------------------------------
FROM dependencies AS build
COPY --from=source /ws /ws
RUN uv pip install --system --break-system-packages --no-deps \
        -e /ws/src/common -e /ws/src/lidar_preprocessing

# ---------------------------------------------------------------------------
FROM build AS deploy
WORKDIR /ws
ENTRYPOINT ["python", "-m", "wato_lidar_preprocessing"]
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
