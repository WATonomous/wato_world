#!/usr/bin/env bash
set -euo pipefail

sanitize_branch_name() {
    echo "${1//[^a-zA-Z0-9._]/-}" | cut -c1-128
}

SOURCE_BRANCH_NAME="$(sanitize_branch_name "${SOURCE_BRANCH}")"
TARGET_BRANCH_NAME="$(sanitize_branch_name "${TARGET_BRANCH}")"

if [[ -z "${TARGET_BRANCH_NAME}" ]]; then
    TARGET_BRANCH_NAME="${SOURCE_BRANCH_NAME}"
fi

echo "source_branch=${SOURCE_BRANCH_NAME}" >> "${GITHUB_OUTPUT}"
echo "target_branch=${TARGET_BRANCH_NAME}" >> "${GITHUB_OUTPUT}"

echo "::notice:: Using ${SOURCE_BRANCH_NAME} as the source branch"
echo "::notice:: Using ${TARGET_BRANCH_NAME} as the target branch"
