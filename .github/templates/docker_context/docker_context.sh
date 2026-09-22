#!/usr/bin/env bash
set -euo pipefail

emit() {
    echo "$1"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "$1" >> "${GITHUB_OUTPUT}"
    fi
}

REGISTRY_URL="ghcr.io/watonomous/wato_world"
REGISTRY="${REGISTRY_URL%%/*}"
REPOSITORY="${REGISTRY_URL#*/}"

ALL_MODULES=(
    ingest
    perception_2d
    semantic_lifting
    lidar_preprocessing
    proposal_generation
    tracking
    label_refinement
    open_vocab_discovery
    student_training
)

module_json() {
    local module="$1"

    case "${module}" in
        ingest)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/ingest.Dockerfile" \
                --arg pkg "wato_ingest" \
                --arg dir "ingest" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        perception_2d)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/perception_2d.Dockerfile" \
                --arg pkg "wato_perception_2d" \
                --arg dir "perception_2d" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        semantic_lifting)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/semantic_lifting.Dockerfile" \
                --arg pkg "wato_semantic_lifting" \
                --arg dir "semantic_lifting" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        lidar_preprocessing)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/lidar_preprocessing.Dockerfile" \
                --arg pkg "wato_lidar_preprocessing" \
                --arg dir "lidar_preprocessing" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        proposal_generation)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/proposal_generation.Dockerfile" \
                --arg pkg "wato_proposal_generation" \
                --arg dir "proposal_generation" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        tracking)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/tracking.Dockerfile" \
                --arg pkg "wato_tracking" \
                --arg dir "tracking" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        label_refinement)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/label_refinement.Dockerfile" \
                --arg pkg "wato_label_refinement" \
                --arg dir "label_refinement" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        open_vocab_discovery)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/open_vocab_discovery.Dockerfile" \
                --arg pkg "wato_open_vocab_discovery" \
                --arg dir "open_vocab_discovery" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        student_training)
            jq -nc --arg module "${module}" \
                --arg dockerfile "docker/student_training.Dockerfile" \
                --arg pkg "wato_student_training" \
                --arg dir "student_training" \
                '{module:$module,dockerfile:$dockerfile,pkg:$pkg,dir:$dir}'
            ;;
        *)
            echo "Unknown module in MODIFIED_MODULES: ${module}" >&2
            return 1
            ;;
    esac
}

if [[ -n "${MODIFIED_MODULES:-}" ]]; then
    # shellcheck disable=SC2206
    MODULES_TO_PROCESS=(${MODIFIED_MODULES})
else
    MODULES_TO_PROCESS=("${ALL_MODULES[@]}")
fi

declare -a json_rows=()
for module in "${MODULES_TO_PROCESS[@]}"; do
    json_rows+=("$(module_json "${module}")")
done

matrix=$(printf '%s\n' "${json_rows[@]}" | jq -s '{include: .}' | jq -c .)

emit "docker_matrix=${matrix}"
emit "registry=${REGISTRY}"
emit "repository=${REPOSITORY}"
