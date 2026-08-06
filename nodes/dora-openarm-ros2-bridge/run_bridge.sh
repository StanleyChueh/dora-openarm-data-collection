#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

<<<<<<< HEAD
exec /home/csl/Stanley_ws/dora-openarm-data-collection/.venv-ros2/bin/python \
  -u /home/csl/Stanley_ws/dora-openarm-data-collection/nodes/dora-openarm-ros2-bridge/bridge.py "$@"
=======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$REPO_ROOT/.venv-ros2/bin/python" \
  -u "$SCRIPT_DIR/bridge.py"
>>>>>>> f18fbff0fb2adaa4b146dbbd766cfa819b3d76f7
