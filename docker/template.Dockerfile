# Universal component builder. Each component Dockerfile defines two named
# build layers: `source` (copies component src/) and `dependencies` (apt + uv pip),
# then this template imports them via the MODULE_SOURCE / MODULE_DEPS args.
#
# This mirrors wato_monorepo/docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6

ARG MODULE_SOURCE
ARG MODULE_DEPS

# ---------------------------------------------------------------------------
# Resolve the two prebuilt build layers.
# ---------------------------------------------------------------------------
FROM ${MODULE_DEPS} AS deps_resolved
FROM ${MODULE_SOURCE} AS source_resolved

# ---------------------------------------------------------------------------
# Build: install src/common + the component's own package as editable wheels.
# Each component Dockerfile sets PACKAGE_DIR (e.g. "perception_2d").
# ---------------------------------------------------------------------------
FROM deps_resolved AS build
ARG COMPONENT_PACKAGE_DIR
COPY --from=source_resolved /ws /ws
RUN uv pip install --system --no-deps -e /ws/src/common \
                                      -e /ws/src/${COMPONENT_PACKAGE_DIR}

# ---------------------------------------------------------------------------
# Deploy: minimal runtime image. ENTRYPOINT set per component via PACKAGE_NAME.
# ---------------------------------------------------------------------------
FROM build AS deploy
ARG COMPONENT_PACKAGE
ENV COMPONENT_PACKAGE=${COMPONENT_PACKAGE}
WORKDIR /ws
ENTRYPOINT ["sh", "-c", "exec python -m \"${COMPONENT_PACKAGE}\" \"$@\"", "--"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Develop: adds host-user mapping + dev tooling.  `command: sleep infinity`
# in docker-compose.dev.yaml so you can `exec` in.
# ---------------------------------------------------------------------------
FROM build AS develop
ARG USERNAME=wato
ARG USER_UID=1000
ARG USER_GID=1000
ARG CLAUDE_CODE=false

RUN groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} 2>/dev/null || true \
 && apt-get update && apt-get install -y --no-install-recommends \
        sudo less vim tmux htop \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system pytest pytest-cov ruff black ipython

# Optional: Claude Code CLI in the dev image.
RUN if [ "${CLAUDE_CODE}" = "true" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash || true ; \
    fi

USER ${USERNAME}
WORKDIR /ws
CMD ["sleep", "infinity"]
