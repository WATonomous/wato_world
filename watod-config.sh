# watod-config.sh — user-editable defaults for the auto-labeling pipeline.
# Override values locally by creating watod-config.local.sh (git-ignored).

# ---------------------------------------------------------------------------
# Component selection. Override on the CLI with -c TARGET (repeatable).
# Append :dev for the develop target with source bind-mounts.
# ---------------------------------------------------------------------------
export ACTIVE_COMPONENTS="ingest"

# ---------------------------------------------------------------------------
# Hardware.
# ---------------------------------------------------------------------------
# Set to "true" if the host has an NVIDIA GPU + nvidia-container-toolkit.
export GPU_AVAILABLE="false"

# ---------------------------------------------------------------------------
# Dev tooling baked into the develop target.
# ---------------------------------------------------------------------------
export CLAUDE_CODE="false"

# ---------------------------------------------------------------------------
# Image registry. Component images derive from this prefix.
# ---------------------------------------------------------------------------
export REGISTRY="ghcr.io/watonomous/wato_world"

# ---------------------------------------------------------------------------
# Storage roots inside containers (bind-mounted from ${WATO_WORLD_DIR}/data).
# ---------------------------------------------------------------------------
export ARTIFACT_ROOT_URI="file:///data/artifacts"
export BAG_ROOT="/data/bags"
export MODELS_ROOT="/data/models"
