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
  button_x     – bool[1]     VR X button (anchor / reset / release a Y hold)
  button_y     – bool[1]     VR Y button (reset to the keyframe pose and hold)

Outputs:
  position_right – float32[8] solved right arm joint angles
  position_left  – float32[8] solved left arm joint angles
  status         – ["ready"] on startup

Operator anchoring (button_x / button_y)
----------------------------------------
Without ``button_x`` wired the node behaves as a plain pass-through: incoming
targets are fed to the solver verbatim.

With it wired the node starts *held*: it publishes the ``--keyframe`` (default
``home``) posture and nothing else, so the robot does not move on startup no
matter where the controllers are. From there X cycles between two states, and
the X press counter goes back to zero after the second press and whenever Y is
pressed -- so every episode is the same two presses:

  X (1st press) – *anchor and go live*. The operator's current controller pose
    is latched per arm, and from then on every incoming target is re-expressed
    as an offset from that anchor, applied on top of the EE pose the keyframe
    posture puts the arm in:

        p_cmd = p_home + (p_ctrl - p_anchor)
        R_cmd = R_home * (R_anchor^-1 * R_ctrl)

    At the instant of the press that evaluates to exactly the keyframe pose, so
    the arms stay where they are and start tracking from there without a jump.
    Re-anchoring on every cycle is what makes the operator's pose map onto the
    same MuJoCo reset pose in every episode, whatever they did in the last one.

  X (2nd press) – *reset and hold*, and the counter returns to 0, so the next X
    press anchors again.

  Y (any time) – identical to the 2nd X press, and also returns the counter to
    0. This is the discard-episode path.

Reset snaps the solver's configuration back to the keyframe posture and
publishes those joint angles immediately. Hold then freezes that exact command
and republishes it at the incoming target rate: controller poses keep being
tracked but stop driving the arms, the gripper stops following the trigger, and
downstream observations, recording and the sim keep ticking against a perfectly
constant action until the next X press.
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


def _anchored_pose(
    home: np.ndarray, anchor: np.ndarray, current: np.ndarray
) -> np.ndarray:
    """Re-express `current` as an offset from `anchor`, applied on top of `home`.

    Translation is offset in the shared origin frame so that hand motion maps to
    end-effector motion along the same axes; rotation is applied as a body-frame
    delta (the controller frame is identified with the EE frame), so that at
    ``current == anchor`` the result is exactly ``home``.
    """
    delta_quat = np.empty(4)
    inv_anchor = np.empty(4)
    mujoco.mju_negQuat(inv_anchor, np.asarray(anchor[3:7], dtype=np.float64))
    mujoco.mju_mulQuat(
        delta_quat, inv_anchor, np.asarray(current[3:7], dtype=np.float64)
    )
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, np.asarray(home[3:7], dtype=np.float64), delta_quat)
    mujoco.mju_normalize4(quat)

    pose = np.empty(7, dtype=np.float32)
    pose[:3] = np.asarray(home[:3]) + (
        np.asarray(current[:3]) - np.asarray(anchor[:3])
    )
    pose[3:7] = quat
    return pose


def _home_driver_positions(kin: Kinematics, home_qpos: np.ndarray) -> np.ndarray:
    """float32[16] driver joint state (right[8]+left[8]) for the keyframe posture."""
    resolver = kin.setup.joint_resolver
    joints_right, finger_right = resolver.get_driver(home_qpos, "right")
    joints_left, finger_left = resolver.get_driver(home_qpos, "left")
    return np.concatenate(
        [
            np.append(joints_right, float(finger_right)),
            np.append(joints_left, float(finger_left)),
        ]
    ).astype(np.float32)


def _run(args: argparse.Namespace) -> None:
    kin = Kinematics(setup_from_args(args), ik_params_from_args(args))

    grip = {s: _gripper_endpoints(kin.setup.model, s) for s in ("right", "left")}
    for side, (open_value, closed_value) in grip.items():
        print(
            f"[ik] {side} gripper: trigger 0.0 → {open_value:+.4f} (open), "
            f"1.0 → {closed_value:+.4f} (closed)",
            flush=True,
        )

    # Captured before any FK/IK call touches setup.data: ArmSetup.from_args leaves
    # it at the --keyframe posture, so these are the poses/joint angles the X/Y
    # reset returns to and the poses the operator anchor is mapped onto.
    home_qpos = kin.setup.data.qpos.copy()
    home_pose = {
        side: np.asarray(kin.setup.read_ee_pose(side), dtype=np.float64)
        for side in kin.setup.sides
    }
    home_driver = _home_driver_positions(kin, home_qpos)
    for side, pose in home_pose.items():
        print(
            f"[ik] {side} keyframe '{args.keyframe}' EE pose: "
            f"p[{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}] "
            f"q[{pose[3]:+.3f}, {pose[4]:+.3f}, {pose[5]:+.3f}, {pose[6]:+.3f}]",
            flush=True,
        )

    anchor: dict[str, np.ndarray] = {}
    latest_target: dict[str, np.ndarray] = {}
    button_prev = {"button_x": False, "button_y": False}
    # 0 = the next button_x anchors and goes live, 1 = the next one resets and
    # holds. Reset to 0 by that second press and by every button_y.
    x_presses = 0
    # Mirrors kin.set_gripper()'s value so a reset republishes the live trigger
    # command instead of snapping the fingers to the keyframe width.
    gripper = {"right": 0.0, "left": 0.0}
    # float32[16] frozen by a reset and republished verbatim until button_x
    # releases it; None means the operator is in control. A dict so the nested
    # handlers below share one piece of state.
    hold: dict[str, np.ndarray | None] = {
        "command": home_driver.copy() if args.hold_until_anchor else None
    }
    if args.hold_until_anchor:
        print(
            f"[ik] holding keyframe '{args.keyframe}' – press button_x to anchor "
            "the operator pose and start driving the arms.",
            flush=True,
        )

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))

    def _target_pose(side: str) -> np.ndarray:
        """EE target for `side`: the keyframe pose while held, else the
        (anchor-corrected, once anchored) controller pose."""
        if hold["command"] is not None:
            return home_pose[side].astype(np.float32)
        current = latest_target[side]
        if side not in anchor:
            return current.astype(np.float32)
        return _anchored_pose(home_pose[side], anchor[side], current)

    def _anchor_now() -> bool:
        missing = [side for side in kin.setup.sides if side not in latest_target]
        if missing:
            print(
                f"[ik] button_x: no controller pose yet for {', '.join(missing)} – "
                "not anchored, press again once poses are flowing.",
                flush=True,
            )
            return False
        for side in kin.setup.sides:
            anchor[side] = latest_target[side].copy()
            pose = anchor[side]
            print(
                f"[ik] anchored {side} operator pose "
                f"p[{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}] "
                f"q[{pose[3]:+.3f}, {pose[4]:+.3f}, {pose[5]:+.3f}, {pose[6]:+.3f}] "
                f"→ keyframe '{args.keyframe}' pose",
                flush=True,
            )
        return True

    def _publish(command: np.ndarray) -> None:
        ts = {"timestamp": time.time_ns()}
        node.send_output("position_right", pa.array(command[:8], type=pa.float32()), ts)
        node.send_output("position_left", pa.array(command[8:16], type=pa.float32()), ts)

    def _reset_to_home(source: str) -> np.ndarray:
        """Snap the solver to the keyframe posture, publish it, and return it."""
        kin.sync(home_driver)
        command = home_driver.copy()
        command[7] = gripper["right"]
        command[15] = gripper["left"]
        _publish(command)
        drift = ", ".join(
            f"{side} {np.linalg.norm(latest_target[side][:3] - anchor[side][:3]):.3f} m"
            for side in kin.setup.sides
            if side in anchor and side in latest_target
        )
        print(
            f"[ik] {source}: reset to keyframe '{args.keyframe}' and holding it "
            "until button_x is pressed"
            + (f" (operator drifted {drift} from the anchor)" if drift else ""),
            flush=True,
        )
        return command

    def _release_hold(source: str) -> None:
        was_held = hold["command"] is not None
        hold["command"] = None
        if was_held:
            print(
                f"[ik] {source}: hold released – the operator is driving the arms.",
                flush=True,
            )

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]

        if eid in button_prev:
            pressed = bool(np.asarray(event["value"]).reshape(-1)[0])
            rising = pressed and not button_prev[eid]
            button_prev[eid] = pressed
            if not rising:
                continue
            if eid == "button_y":
                x_presses = 0
                hold["command"] = _reset_to_home(eid)
            elif x_presses == 0:
                # First press of the cycle: re-anchor on the operator's current
                # pose and hand the arms over to them. Left at 0 if there is no
                # controller pose yet, so the next press retries the anchor.
                if _anchor_now():
                    x_presses = 1
                    _release_hold(eid)
            else:
                x_presses = 0
                hold["command"] = _reset_to_home(eid)
            continue

        values = np.array(event["value"], dtype=np.float32)

        if eid == "position":
            if values.shape == (16,):
                kin.sync(values)
            continue

        if eid == "target_right" and "right" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_right[7], got {values.shape}. Skipping.")
                continue
            latest_target["right"] = values.astype(np.float64)
            kin.set_target("right", _target_pose("right"))

        elif eid == "target_left" and "left" in kin.setup.sides:
            if values.shape != (7,):
                print(f"Warning: expected target_left[7], got {values.shape}. Skipping.")
                continue
            latest_target["left"] = values.astype(np.float64)
            kin.set_target("left", _target_pose("left"))

        elif eid == "trigger_right":
            gripper["right"] = _map_trigger_to_gripper(float(values[0]), *grip["right"])
            kin.set_gripper("right", gripper["right"])
            continue

        elif eid == "trigger_left":
            gripper["left"] = _map_trigger_to_gripper(float(values[0]), *grip["left"])
            kin.set_gripper("left", gripper["left"])
            continue

        else:
            continue

        if not kin.ready():
            continue

        result = kin.solve()
        if result is None:
            continue

        # While held the solve above only pins the configuration to the keyframe
        # (its target is the keyframe pose); the frozen command is what ships, so
        # the arms cannot creep and the recorded action stays exactly constant.
        _publish(hold["command"] if hold["command"] is not None else result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mink IK dora node – OpenArm end-effector pose → joint angles"
    )
    register_common_args(parser)
    register_ik_args(parser)
    parser.add_argument(
        "--hold-until-anchor",
        action="store_true",
        help=(
            "Start holding the --keyframe posture and ignore controller poses until"
            " button_x anchors the operator (see this module's docstring). Off by"
            " default, where the node tracks the controllers from the first packet."
            " Requires button_x to be wired, or the arms never start moving."
        ),
    )
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
