# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dora node: mink-based differential IK solver for OpenArm.

Accepts end-effector pose targets and solves joint angles via mink's QP-based
differential IK. Both arms share one mink.Configuration and one QP solve per
step.

Pose convention (inputs and outputs):  float32[7] = [px, py, pz, qw, qx, qy, qz]
Inputs:
  target_right – float32[7]  right EE target pose
  target_left  – float32[7]  left  EE target pose
  position     – float32[16] current joint state right[8]+left[8] (optional sync)

Outputs:
  position_right – float32[8] solved right arm joint angles
  position_left  – float32[8] solved left arm joint angles
  status         – ["ready"] on startup
"""

from __future__ import annotations

import argparse
import time

import dora
import mujoco
import numpy as np
import pyarrow as pa

from openarm_control import (
    Kinematics,
    register_common_args,
    register_ik_args,
    ik_params_from_args,
    setup_from_args,
)


def _gripper_endpoints(model: mujoco.MjModel, side: str) -> tuple[float, float]:
    """Return one arm's (open, closed) gripper command, read off the model.

    Every OpenArm model puts the closed pose at command 0 and the open pose at
    the far end of the finger actuator's ctrlrange, but they disagree on units
    and sign: the v2 models use a hinge (right [-0.785, 0] rad, left
    [0, 0.785] rad) while v1_camera uses a prismatic joint ([0, 0.044] m on
    both sides).  Reading the endpoints keeps one trigger mapping correct for
    whichever model --xml points at; hardcoding v2's radians silently clamped
    every v1_camera right-hand command to 0, jamming that gripper shut.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger1_ctrl")
    if aid < 0:
        raise RuntimeError(f"Actuator '{side}_finger1_ctrl' not found in the IK model")
    lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
    return (lo, hi) if abs(lo) > abs(hi) else (hi, lo)


def _map_trigger_to_gripper(
    trigger: float, open_value: float, closed_value: float
) -> float:
    """trigger 0.0 (released) → open, 1.0 (fully squeezed) → closed."""
    t = min(max(trigger, 0.0), 1.0)
    return open_value + t * (closed_value - open_value)


def _run(args: argparse.Namespace) -> None:
    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    grip = {s: _gripper_endpoints(kin.setup.model, s) for s in ("right", "left")}
    for side, (open_value, closed_value) in grip.items():
        print(
            f"[ik] {side} gripper: trigger 0.0 → {open_value:+.4f} (open), "
            f"1.0 → {closed_value:+.4f} (closed)",
            flush=True,
        )

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        values = np.array(event["value"], dtype=np.float32)

        if eid == "position":
            if values.shape == (16,):
                kin.sync(values)
            continue

        if eid == "target_right" and "right" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_right[7], got {values.shape}. Skipping.")
                continue
            kin.set_target("right", values)

        elif eid == "target_left" and "left" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_left[7], got {values.shape}. Skipping.")
                continue
            kin.set_target("left", values)

        elif eid == "trigger_right":
            kin.set_gripper("right", _map_trigger_to_gripper(float(values[0]), *grip["right"]))
            continue

        elif eid == "trigger_left":
            kin.set_gripper("left", _map_trigger_to_gripper(float(values[0]), *grip["left"]))
            continue

        else:
            continue

        if not kin.ready():
            continue

        result = kin.solve()
        if result is None:
            continue

        ts = {"timestamp": time.time_ns()}
        node.send_output("position_right", pa.array(result[:8],  type=pa.float32()), ts)
        node.send_output("position_left",  pa.array(result[8:16], type=pa.float32()), ts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
