#!/usr/bin/env bash
# Entry point for the `ros2-bridge` dora node.
#
# dora needs a single executable it can spawn, but bridge.py has to run under an
# interpreter that can see BOTH rclpy (from /opt/ros/humble, system Python 3.10)
# and dora-rs + pyarrow (from .venv-ros2, which is created with
# --system-site-packages).  This script is that shim.
#
# Paths are derived from this file's own location, so the repo can live anywhere
# and be checked out under any workspace name.
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# NOT `set -u`: /opt/ros/humble/setup.bash reads unset variables
# (AMENT_TRACE_SETUP_FILES and friends), so nounset makes sourcing it fatal.
set -eo pipefail
=======
set -euo pipefail
>>>>>>> Stashed changes
=======
set -euo pipefail
>>>>>>> Stashed changes

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV_PY="$REPO/.venv-ros2/bin/python"

source /opt/ros/humble/setup.bash

# Fail with something readable instead of bash's bare 126/127.  The venv's
# bin/python is a SYMLINK, and it is committed to git -- a checkout on a
# filesystem without symlink support turns it into a small text file containing
# the target path, which is not executable.
if [ ! -x "$VENV_PY" ]; then
  echo "run_bridge.sh: $VENV_PY is missing or not executable." >&2
  if [ -f "$VENV_PY" ]; then
    echo "  It exists but is not executable. If it is a small text file rather" >&2
    echo "  than a symlink, the venv was checked out without symlink support." >&2
    echo "  Do NOT chmod +x it -- recreate the venv:" >&2
  else
    echo "  Recreate the venv:" >&2
  fi
  echo "    rm -rf $REPO/.venv-ros2" >&2
  echo "    /usr/bin/python3.10 -m venv --system-site-packages $REPO/.venv-ros2" >&2
  echo "    $REPO/.venv-ros2/bin/python -m pip install 'dora-rs==0.5.0' pyarrow" >&2
  echo "  Use /usr/bin/python3.10 explicitly: a bare 'python3' may resolve to a" >&2
  echo "  uv-managed 3.13, whose ABI does not match Humble's rclpy .so files." >&2
  exit 1
fi

exec "$VENV_PY" -u "$HERE/bridge.py"
