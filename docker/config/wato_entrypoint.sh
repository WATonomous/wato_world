#!/bin/bash
set -e

# Source ROS2 environment only if present (wato_world has no ROS).
# shellcheck disable=SC1090,SC1091
if [ -n "${ROS_DISTRO}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source /opt/ros/"${ROS_DISTRO}"/setup.bash
fi
[ -f /opt/watonomous/setup.bash ] && source /opt/watonomous/setup.bash

exec "$@"
