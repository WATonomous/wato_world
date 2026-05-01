# Student training — self-trained detector.
# Trains BEVFusion / TransFusion on refined_tracks + camera/lidar inputs.

# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ghcr.io/watonomous/wato_world/base:cuda-12.4

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS source
WORKDIR /ws
COPY src/common /ws/src/common
COPY src/student_training /ws/src/student_training

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --break-system-packages \
        pyarrow numpy scipy pydantic fsspec click pyyaml \
        wandb tqdm

# Heavy training deps.
# RUN uv pip install --system --break-system-packages --extra-index-url \
#         https://download.pytorch.org/whl/cu124 torch torchvision
# RUN uv pip install --system --break-system-packages spconv-cu124 \
#  && git clone https://github.com/mit-han-lab/bevfusion /opt/bevfusion \
#  && cd /opt/bevfusion && python setup.py develop

# ---------------------------------------------------------------------------
FROM dependencies AS build
COPY --from=source /ws /ws
RUN uv pip install --system --break-system-packages --no-deps \
        -e /ws/src/common -e /ws/src/student_training

# ---------------------------------------------------------------------------
FROM build AS deploy
WORKDIR /ws
ENTRYPOINT ["python", "-m", "wato_student_training"]
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
