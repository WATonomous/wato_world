#!/usr/bin/env bash
# watod-fetch-models — download pretrained model checkpoints into MODELS_ROOT.
#
# Pretrained weights live in the host directory bind-mounted as /data/models
# inside each GPU container.  Each component lazily loads its weights from
# `${MODELS_ROOT}/<subdir>/<file>` at runtime.  Without this script, first
# run would fail (offline) or auto-download (slow + breaks air-gapped envs).
#
# Idempotent — skips files that already exist with the expected size.
# Currently registers:
#   * Depth Anything V2 Large  (perception_2d)
#   * YOLO-World v8-L           (perception_2d)
# Future PRs extend the manifest below as new components land.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set by the watod entrypoint}"

# Host-side path that gets mounted into containers as /data/models.
HOST_MODELS_ROOT="${WATO_WORLD_DIR}/data/models"

COMPONENT_FILTER=""

usage() {
    cat <<EOF
watod fetch-models — populate \$MODELS_ROOT with pretrained checkpoints.

Usage:
  watod fetch-models [OPTIONS]

Options:
  -c, --component NAME   Download only models for component NAME (e.g.
                         perception_2d).  Default: all registered models.
  -h, --help             Show this help.

Registered models (this PR):
  perception_2d:
    depth_anything_v2/depth_anything_v2_vitl.pth   (~1.3 GB)
    yolo_world/yolov8l-worldv2.pt                  (~150 MB)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--component) COMPONENT_FILTER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

# --- Manifest of (component, subdir, filename, url, min_size_bytes) -----
# Each line: COMPONENT|SUBDIR|FILENAME|URL|MIN_SIZE_BYTES
read -r -d '' MANIFEST <<'EOF' || true
perception_2d|depth_anything_v2|depth_anything_v2_vitl.pth|https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth|1100000000
perception_2d|yolo_world|yolov8l-worldv2.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt|100000000
EOF

# --- Pre-flight ----------------------------------------------------------
mkdir -p "${HOST_MODELS_ROOT}"
echo "MODELS_ROOT host path: ${HOST_MODELS_ROOT}"
echo "                       (mounted as /data/models inside containers)"
echo

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 1
fi

# --- Download loop -------------------------------------------------------
total=0
skipped=0
downloaded=0
failed=0

while IFS='|' read -r component subdir filename url min_size; do
    [[ -z "${component}" ]] && continue
    if [[ -n "${COMPONENT_FILTER}" && "${component}" != "${COMPONENT_FILTER}" ]]; then
        continue
    fi
    total=$((total + 1))
    target_dir="${HOST_MODELS_ROOT}/${subdir}"
    target_file="${target_dir}/${filename}"
    mkdir -p "${target_dir}"

    if [[ -f "${target_file}" ]]; then
        actual_size=$(stat -c%s "${target_file}" 2>/dev/null || stat -f%z "${target_file}")
        if [[ "${actual_size}" -ge "${min_size}" ]]; then
            echo "[skip] ${component}/${subdir}/${filename}  (already present, $(numfmt --to=iec --suffix=B "${actual_size}"))"
            skipped=$((skipped + 1))
            continue
        else
            echo "[warn] ${target_file} exists but is below expected size (${actual_size} < ${min_size}); re-downloading"
        fi
    fi

    echo "[get]  ${component}/${subdir}/${filename}"
    echo "       ${url}"
    if curl --fail --location --progress-bar -o "${target_file}.tmp" "${url}"; then
        mv "${target_file}.tmp" "${target_file}"
        actual_size=$(stat -c%s "${target_file}" 2>/dev/null || stat -f%z "${target_file}")
        echo "[ok]   $(numfmt --to=iec --suffix=B "${actual_size}")"
        downloaded=$((downloaded + 1))
    else
        echo "[fail] $(basename "${url}") — leaving partial at ${target_file}.tmp" >&2
        failed=$((failed + 1))
    fi
done <<< "${MANIFEST}"

# --- Summary -------------------------------------------------------------
echo
echo "Summary: ${total} registered, ${downloaded} downloaded, ${skipped} skipped, ${failed} failed."
echo "Disk usage at ${HOST_MODELS_ROOT}:"
du -sh "${HOST_MODELS_ROOT}" 2>/dev/null || true

if [[ "${failed}" -gt 0 ]]; then
    exit 1
fi
