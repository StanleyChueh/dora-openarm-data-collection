#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

exec /home/csl/Stanley_ws/dora-openarm-data-collection/.venv-ros2/bin/python \
  -u /home/csl/Stanley_ws/dora-openarm-data-collection/nodes/dora-openarm-ros2-bridge/bridge.py "$@"
