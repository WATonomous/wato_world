#!/usr/bin/env bash
# Run every implemented stage on one bag, in order, each through watod-run.sh
# (so each stage uses its dev service when it is active as :dev, like
# `watod run`). Usage:
#   watod pipeline --bag <bag> [--bag-id <id>] [--profile <name>]
#                  [--force <stage>[,<stage>...]|all] [--from <stage>]
#                  [--to <stage>] [--dry-run]
#
# Every stage skips the work it has already done, so a re-run only does what
# is missing. --force <stage> re-runs that stage and every later one: their
# inputs changed, and a stage only checks that its outputs exist. A stage not
# active as :dev runs its image as last built, so rebuild (`watod build`)
# after changing its code.
#
# --profile <name> picks each stage's config/<stage>.<name>.yaml, falling back
# to the longest existing prefix of <name> (wato_6cam → wato), then to
# <stage>.yaml: `--profile wato_6cam` runs ingest.wato_6cam.yaml with every
# other stage's .wato.yaml.
#
# --bag is what `watod run ingest --bag` takes (a path under /data/bags); later
# stages get its bag_id, derived as ingest derives it (or --bag-id). With
# --from past ingest, --bag may be the bag_id itself.

set -euo pipefail

: "${WATO_WORLD_DIR:?WATO_WORLD_DIR must be set}"

SCRIPTS_DIR="${WATO_WORLD_DIR}/watod_scripts"

# The implemented stages, in dependency order (open_vocab_discovery and
# student_training are stubs).
STAGES=(ingest lidar_preprocessing camera_calibration detection_3d perception_2d semantic_lifting proposal_generation tracking label_refinement)

usage() {
    cat <<EOF
Usage: watod pipeline --bag <bag> [options]

Runs ${STAGES[*]}
on one bag. Each stage skips work it has already done.

Options:
  --bag BAG            Bag path as ingest takes it (/data/bags/...), or a bag_id
                       when --from is past ingest.
  --bag-id ID          Override the bag_id ingest derives from the bag's name.
  --profile NAME       Config profile, e.g. wato, wato_6cam (default: none,
                       the nuScenes configs).
  --force STAGES       Re-run these stages (comma-separated, or "all") and
                       every stage after the first of them.
  --from STAGE         Start at STAGE (default: ingest).
  --to STAGE           Stop after STAGE (default: label_refinement).
  --dry-run            Print each stage's command without running it.
  -h, --help           Show this help.

Examples:
  watod pipeline --bag /data/bags/NuScenes-v1.0-mini-scene-0061_0.mcap
  watod pipeline --bag /data/bags/sensors_2026_07_07-23_01_24_2_with_eidos.mcap \\
      --profile wato_no_lidar_nw
  watod pipeline --bag NuScenes_v1_0_mini_scene_0061_0_mcap --from tracking --force tracking
EOF
}

stage_index() {
    local i
    for i in "${!STAGES[@]}"; do
        if [[ "${STAGES[$i]}" == "$1" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "Unknown stage: $1 (stages: ${STAGES[*]})" >&2
    exit 64
}

# ingest's derive_bag_id: the bag's file or directory name, every run of
# characters other than [a-zA-Z0-9_] replaced by one "_", outer "_" stripped.
derive_bag_id() {
    local name
    name="$(basename "$1")"
    name="$(sed -E 's/[^a-zA-Z0-9_]+/_/g; s/^_+//; s/_+$//' <<<"${name}")"
    echo "${name:-bag}"
}

# config/<stage>.<profile>.yaml for the longest existing prefix of the
# profile, else <stage>.yaml; echoes the in-container path.
config_for() {
    local stage="$1" profile="$2" dir="${WATO_WORLD_DIR}/src/$1/config"
    while [[ -n "${profile}" ]]; do
        if [[ -f "${dir}/${stage}.${profile}.yaml" ]]; then
            echo "/ws/src/${stage}/config/${stage}.${profile}.yaml"
            return 0
        fi
        [[ "${profile}" == *_* ]] || break
        profile="${profile%_*}"
    done
    echo "/ws/src/${stage}/config/${stage}.yaml"
}

BAG=""
BAG_ID=""
PROFILE=""
FORCE_LIST=""
FROM="${STAGES[0]}"
TO="${STAGES[-1]}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)
            BAG="${2:?--bag needs a value}"
            shift 2
            ;;
        --bag-id)
            BAG_ID="${2:?--bag-id needs a value}"
            shift 2
            ;;
        --profile)
            PROFILE="${2:?--profile needs a value}"
            shift 2
            ;;
        --force)
            FORCE_LIST="${FORCE_LIST:+${FORCE_LIST},}${2:?--force needs a stage list}"
            shift 2
            ;;
        --from)
            FROM="${2:?--from needs a stage}"
            shift 2
            ;;
        --to)
            TO="${2:?--to needs a stage}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

if [[ -z "${BAG}" ]]; then
    usage >&2
    exit 64
fi

FROM_I="$(stage_index "${FROM}")"
TO_I="$(stage_index "${TO}")"
if ((FROM_I > TO_I)); then
    echo "--from ${FROM} comes after --to ${TO}" >&2
    exit 64
fi

# The first forced stage; every stage from it on is forced.
FORCE_I=${#STAGES[@]}
if [[ -n "${FORCE_LIST}" ]]; then
    IFS=',' read -r -a forced <<<"${FORCE_LIST}"
    for s in "${forced[@]}"; do
        [[ -z "${s}" ]] && continue
        if [[ "${s}" == "all" ]]; then
            i=0
        else
            i="$(stage_index "${s}")"
        fi
        ((i < FORCE_I)) && FORCE_I=$i
    done
fi

RESOLVED_ID="${BAG_ID:-$(derive_bag_id "${BAG}")}"

echo "pipeline: bag ${BAG} → bag_id ${RESOLVED_ID}"
echo "  stages ${STAGES[FROM_I]} → ${STAGES[TO_I]}${PROFILE:+, profile ${PROFILE}}"
if ((FORCE_I <= TO_I)); then
    echo "  forced from ${STAGES[$((FORCE_I > FROM_I ? FORCE_I : FROM_I))]} on"
fi

n=$((TO_I - FROM_I + 1))
t0=${SECONDS}
for ((i = FROM_I; i <= TO_I; i++)); do
    stage="${STAGES[$i]}"
    config="$(config_for "${stage}" "${PROFILE}")"
    if [[ "${stage}" == "ingest" ]]; then
        args=(--bag "${BAG}" --config "${config}")
        [[ -n "${BAG_ID}" ]] && args+=(--bag-id "${BAG_ID}")
    else
        args=(--bag "${RESOLVED_ID}" --config "${config}")
    fi
    ((i >= FORCE_I)) && args+=(--force)

    # watod-run.sh's choice: the dev service (the working tree, bind-mounted)
    # when the stage is active as :dev, else the image as last built.
    code="built image"
    [[ " ${SELECTED_SERVICES_STR:-} " == *" ${stage}_dev "* ]] && code="dev, working tree"

    echo ""
    echo "==> [$((i - FROM_I + 1))/${n}] ${stage} (${code}): watod run ${stage} ${args[*]}"
    if ((DRY_RUN)); then
        continue
    fi
    t_stage=${SECONDS}
    if ! bash "${SCRIPTS_DIR}/watod-run.sh" "${stage}" "${args[@]}"; then
        echo "" >&2
        echo "pipeline: ${stage} failed after $((SECONDS - t_stage)) s;" \
            "re-run with --from ${stage} once fixed" >&2
        exit 1
    fi
    echo "<== ${stage} done in $((SECONDS - t_stage)) s"
done

if ((!DRY_RUN)); then
    echo ""
    echo "pipeline: ${RESOLVED_ID} done in $((SECONDS - t0)) s"
fi
