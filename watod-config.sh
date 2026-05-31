#!/bin/bash
# watod-config.sh — user-editable defaults for the auto-labeling pipeline.
#
# HINT: copy this file to watod-config.local.sh (git-ignored) and override
#       any of the values below.

############################ ACTIVE MODULE CONFIGURATION ############################
## List of active modules to run, defined in modules/docker-compose.yaml.
## Append ":dev" to a module name to use the develop target with source
## bind-mounts (live editing inside the container).
##
## Possible values:
##   - ingest                : decode rosbags into Parquet/JSON/PNG/NPZ artifacts
##   - perception_2d         : Florence-2 + SAM 3.1 + DA-V2 + DINOv2 cross-camera merge
##   - semantic_lifting      : occlusion-aware LiDAR label lifting from 2D masks + metric depth
##   - lidar_preprocessing   : motion compensation, static/dynamic split, ground extraction
##   - proposal_generation   : LiDAR detection ensemble + Segment-Lift-Fit + fusion
##   - tracking              : 4D tracking with masklet association + DINOv2 ReID
##   - label_refinement      : multimodal LabelFormer (bootstrap → learned)
##   - open_vocab_discovery  : rare-class discovery branch
##   - student_training      : BEVFusion / TransFusion student detector training
##
## Examples:
##   ACTIVE_MODULES="ingest"                          # ingest in deploy mode
##   ACTIVE_MODULES="ingest:dev"                      # ingest in dev mode (editable)
##   ACTIVE_MODULES="ingest:dev perception_2d:dev"    # both in dev mode
##   ACTIVE_MODULES="all"                             # every module in deploy mode
##   ACTIVE_MODULES="all:dev"                         # every module in dev mode

export ACTIVE_MODULES="ingest:dev lidar_preprocessing:dev"

############################## ADVANCED CONFIGURATIONS ##############################
## Set to "true" if the host has an NVIDIA GPU + nvidia-container-toolkit.
## DEFAULT = "false"
export GPU_AVAILABLE="true"

## Install Claude Code in dev containers. Set to "false" to skip.
## DEFAULT = "false"
# export CLAUDE_CODE="false"

## Tag suffix for built images (<REGISTRY>/<module>:<TAG>).
## DEFAULT = "<your_current_git_branch>" (slashes replaced with dashes)
# export TAG=""

## Docker registry to pull/push images.
## DEFAULT = "ghcr.io/watonomous/wato_world"
# export REGISTRY=""

## Name to append to docker containers.
## DEFAULT = "<your_username>"
# export COMPOSE_PROJECT_NAME=""

############################### STORAGE ROOTS ##################################
## Where the pipeline reads/writes inside containers (bind-mounted from
## ${WATO_WORLD_DIR}/data).  Only change if you remap the host data directory.
# export ARTIFACT_ROOT_URI="file:///data/artifacts"
# export BAG_ROOT="/data/bags"
# export MODELS_ROOT="/data/models"
