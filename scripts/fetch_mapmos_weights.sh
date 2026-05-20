#!/usr/bin/env bash
# Download PRBonn's pretrained MapMOS checkpoint into data/models/mapmos/.
#
# Source: https://www.ipb.uni-bonn.de/html/projects/MapMOS/mapmos.ckpt
#   (linked from PRBonn/MapMOS README "Downloads" section)
#
# The checkpoint is mounted into the lidar_preprocessing container at
# /data/models/mapmos/mapmos.ckpt and consumed by
# mapmos.weights.load_and_validate when cfg.mapmos.enabled.
#
# Run on the host (not inside the container) so the file lands on the
# bind-mounted host path.

set -euo pipefail

URL="https://www.ipb.uni-bonn.de/html/projects/MapMOS/mapmos.ckpt"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
DEST_DIR="${REPO_ROOT}/data/models/mapmos"
DEST="${DEST_DIR}/mapmos.ckpt"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
    echo "mapmos.ckpt already present at ${DEST}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading MapMOS pretrained checkpoint..."
echo "  from: ${URL}"
echo "  to:   ${DEST}"

if command -v wget &> /dev/null; then
    wget -O "${DEST}" "${URL}"
elif command -v curl &> /dev/null; then
    curl -L -o "${DEST}" "${URL}"
else
    echo "ERROR: neither wget nor curl available on host" >&2
    exit 1
fi

echo "Done. Checkpoint size:"
ls -lh "${DEST}"
