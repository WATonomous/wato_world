#!/usr/bin/env bash
# Symlink ./watod into /usr/local/bin/watod.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"

TARGET="/usr/local/bin/watod"
SOURCE="${WATO_WORLD_DIR}/watod"

if [[ ! -x "${SOURCE}" ]]; then
    chmod +x "${SOURCE}"
fi

if [[ -L "${TARGET}" || -e "${TARGET}" ]]; then
    sudo rm -f "${TARGET}"
fi

sudo ln -s "${SOURCE}" "${TARGET}"
echo "Linked ${TARGET} -> ${SOURCE}"
