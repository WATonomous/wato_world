#!/usr/bin/env bash
set -euo pipefail

MODIFIED_MODULES=""

add_module() {
    local module="$1"
    MODIFIED_MODULES+="${module} "
}

if [[ "${ALL_CHANGED}" == "true" ]]; then
    echo "::notice:: Detected shared or non-src changes; testing all modules"
    add_module ingest
    add_module perception_2d
    add_module lidar_preprocessing
    add_module proposal_generation
    add_module tracking
    add_module label_refinement
    add_module open_vocab_discovery
    add_module student_training
else
    [[ "${INGEST_CHANGED}" == "true" ]] && add_module ingest
    [[ "${PERCEPTION_2D_CHANGED}" == "true" ]] && add_module perception_2d
    [[ "${LIDAR_PREPROCESSING_CHANGED}" == "true" ]] && add_module lidar_preprocessing
    [[ "${PROPOSAL_GENERATION_CHANGED}" == "true" ]] && add_module proposal_generation
    [[ "${TRACKING_CHANGED}" == "true" ]] && add_module tracking
    [[ "${LABEL_REFINEMENT_CHANGED}" == "true" ]] && add_module label_refinement
    [[ "${OPEN_VOCAB_DISCOVERY_CHANGED}" == "true" ]] && add_module open_vocab_discovery
    [[ "${STUDENT_TRAINING_CHANGED}" == "true" ]] && add_module student_training
fi

echo "::notice:: MODIFIED_MODULES are ${MODIFIED_MODULES}"
echo "modified_modules=${MODIFIED_MODULES}" >> "${GITHUB_OUTPUT}"
