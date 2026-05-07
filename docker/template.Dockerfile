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
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
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
 && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc /home/${USERNAME}/.profile \
 && apt-get -qq autoremove -y && apt-get -qq clean \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

RUN uv pip install --system pytest pytest-cov ruff black ipython

# Hand /ws to the dev user so pytest can write __pycache__, ruff its cache,
# etc. without permission errors. Mirrors the chown step in
# wato_monorepo/docker/template.Dockerfile.
RUN chown -R ${USERNAME}:${USERNAME} /ws

# Optional: Claude Code CLI in the dev image.
RUN if [ "${CLAUDE_CODE}" = "true" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash || true ; \
    fi

USER ${USERNAME}
WORKDIR /ws
CMD ["sleep", "infinity"]
