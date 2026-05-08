#!/bin/bash
set -e

# Setup ROS2 environment so all ROS shared libraries and Python packages
# are visible to whatever command this container runs.
# shellcheck disable=SC1090,SC1091
source /opt/ros/"${ROS_DISTRO}"/setup.bash

exec "$@"
