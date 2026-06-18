# Universal component builder. Each component Dockerfile defines two named
# build layers: `source` (copies component src/) and `dependencies` (apt + uv pip),
# then this template imports them via the MODULE_SOURCE / MODULE_DEPS args.
#
# This mirrors wato_monorepo/docker/template.Dockerfile.

# syntax=docker/dockerfile:1.6

ARG MODULE_SOURCE=ingest:source_ingest
ARG MODULE_DEPS=ingest:deps_ingest

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
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG COMPONENT_PACKAGE_DIR
ARG COMPONENT_PACKAGE
ENV COMPONENT_PACKAGE=${COMPONENT_PACKAGE}
COPY --from=source_resolved /ws /ws
RUN uv pip install --system --break-system-packages --no-deps \
        -e /ws/src/common -e /ws/src/${COMPONENT_PACKAGE_DIR}

# Entrypoint: sources /opt/ros/${ROS_DISTRO}/setup.bash if ROS is present, then exec "$@".
# Mirrors wato_monorepo/docker/config/wato_entrypoint.sh.
# Set here so both deploy and develop stages inherit it.
RUN mkdir -p /opt/watonomous
COPY docker/config/wato_entrypoint.sh /opt/watonomous/wato_entrypoint.sh
RUN chmod +x /opt/watonomous/wato_entrypoint.sh
WORKDIR /ws
ENTRYPOINT ["/opt/watonomous/wato_entrypoint.sh"]

# ---------------------------------------------------------------------------
# Deploy: minimal runtime image.
# ---------------------------------------------------------------------------
FROM build AS deploy
# Shell-form CMD so ${COMPONENT_PACKAGE} expands at runtime.
CMD python3 -m ${COMPONENT_PACKAGE} --help

# ---------------------------------------------------------------------------
# Develop: adds host-user mapping + dev tooling.  `command: sleep infinity`
# in docker-compose.dev.yaml so you can `exec` in.
# ---------------------------------------------------------------------------
FROM build AS develop
ARG USERNAME=wato
ARG USER_UID=1000
ARG USER_GID=1000
ARG CLAUDE_CODE=false
ARG DEV_PYTEST_SPEC=pytest

# UID 1000 is taken by the default `ubuntu` user on ubuntu24.04 — delete that
# entry first so useradd can claim 1000 for ${USERNAME}. Also seeds the new
# user's home with the default bashrc/profile so `tmux`/`exec` shells get
# history-search, prompt, and PATH out of the box. Mirrors
# wato_monorepo/docker/template.Dockerfile.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN existing_user=$(getent passwd ${USER_UID} | cut -d: -f1 || true) \
 && if [ -n "$existing_user" ]; then userdel -r "$existing_user" 2>/dev/null || true; fi \
 && if ! getent group ${USER_GID} >/dev/null; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} --shell /bin/bash \
 && apt-get update && apt-get install -y --no-install-recommends \
        sudo less vim tmux htop git curl wget tree nano \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && cp /etc/skel/.bashrc /home/${USERNAME}/.bashrc \
 && cp /etc/skel/.profile /home/${USERNAME}/.profile \
 && chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc /home/${USERNAME}/.profile \
 && apt-get -qq autoremove -y && apt-get -qq clean \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system --break-system-packages "${DEV_PYTEST_SPEC}" pytest-cov ruff black ipython

# Hand /ws to the dev user so pytest can write __pycache__, ruff its cache,
# etc. without permission errors. Mirrors the chown step in
# wato_monorepo/docker/template.Dockerfile.
RUN chown -R ${USER_UID}:${USER_GID} /ws

# Optional: Claude Code CLI in the dev image.
RUN if [ "${CLAUDE_CODE}" = "true" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash || true ; \
    fi

USER ${USERNAME}
WORKDIR /ws

# Setup dev bashrc — mirrors wato_monorepo/docker/template.Dockerfile.
COPY docker/config/wato_dev.bashrc /opt/watonomous/wato_dev.bashrc
RUN echo "source /opt/watonomous/wato_dev.bashrc" >> ~/.bashrc

CMD ["sleep", "infinity"]
