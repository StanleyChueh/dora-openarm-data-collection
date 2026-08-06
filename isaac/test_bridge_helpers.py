#!/usr/bin/env python3
"""Bring-up step V3: exercise bridge.py's pure helpers, no dora and no ROS.

    .venv-ros2/bin/python3 isaac/test_bridge_helpers.py

Imports bridge.py with rclpy/dora stubbed out, so it runs on a dev box with no
ROS installed.

Covers the two functions that were silently broken:

  extract_values        the old arrow_to_float_list() did
                        `float(item) for item in value.to_pylist()`, but every
                        dora node wraps its payload as [{"pose": [...]}], so
                        to_pylist() yields a dict and float(dict) raises
                        TypeError on the first event.

  8 vs 7 element poses  the old length check rejected anything that was not 7,
                        while pose_right/pose_left carry the gripper angle in
                        slot 7 and are 8 wide.  Both sides were dropped before
                        the TypeError could even fire.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

BRIDGE = Path(__file__).resolve().parent.parent / (
    "nodes/dora-openarm-ros2-bridge/bridge.py"
)


def load_bridge_helpers():
    class _Stub:
        def __getattr__(self, _name):
            return _Stub()

        def __call__(self, *_a, **_k):
            return _Stub()

    stubbed = []
    for name in (
        "rclpy",
        "rclpy.qos",
        "geometry_msgs",
        "geometry_msgs.msg",
        "sensor_msgs",
        "sensor_msgs.msg",
        "dora",
    ):
        if name not in sys.modules:
            sys.modules[name] = _Stub()
            stubbed.append(name)

    spec = importlib.util.spec_from_file_location("_bridge_under_test", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
    return module


def main() -> None:
    b = load_bridge_helpers()
    failures: list[str] = []

    def check(label: str, got, want, tol: float | None = None) -> None:
        if tol is None:
            ok = got == want
        else:
            ok = np.allclose(np.asarray(got, dtype=float), want, atol=tol)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}: {got!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, expected {want!r}")

    print("== extract_values ==========================================")
    pose_t = pa.struct({"pose": pa.list_(pa.float32())})
    pose_arr = pa.array([{"pose": np.arange(8, dtype=np.float32)}], type=pose_t)
    got = b.extract_values(pose_arr, "pose")
    check("StructArray pose -> shape", tuple(got.shape), (8,))
    check("StructArray pose -> values", got.tolist(), list(range(8)), tol=0)

    qpos_t = pa.struct({"qpos": pa.list_(pa.float32())})
    qpos_arr = pa.array([{"qpos": np.zeros(8, dtype=np.float32)}], type=qpos_t)
    check(
        "StructArray qpos -> shape",
        tuple(b.extract_values(qpos_arr, "qpos").shape),
        (8,),
    )

    ref_arr = pa.array([{"pose": np.arange(7, dtype=np.float32)}], type=pose_t)
    check(
        "pose_reference is 7 wide",
        tuple(b.extract_values(ref_arr, "pose").shape),
        (7,),
    )

    flat = pa.array(np.arange(8, dtype=np.float32), type=pa.float32())
    check("flat array passes through", tuple(b.extract_values(flat, "pose").shape), (8,))

    print("== gripper_rad_to_stroke ===================================")
    # _map_trigger_to_gripper: squeezed -> 0 rad, released -> +-0.785 rad.
    # openarm v1 finger range is [0, 0.044] with 0 = closed.
    check("released, left  (+0.785 rad)", b.gripper_rad_to_stroke(1.57 / 2), 0.044, tol=1e-9)
    check("released, right (-0.785 rad)", b.gripper_rad_to_stroke(-1.57 / 2), 0.044, tol=1e-9)
    check("squeezed (0 rad)", b.gripper_rad_to_stroke(0.0), 0.0, tol=1e-9)
    check("half open", b.gripper_rad_to_stroke(1.57 / 4), 0.022, tol=1e-9)
    check("clamped beyond full open", b.gripper_rad_to_stroke(10.0), 0.044, tol=1e-9)

    print("== to_base_frame ===========================================")
    pose = b.to_base_frame(
        np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5]), b.ARM_ORIGIN_TO_BASE_Z
    )
    check("drops the 8th (gripper) element", len(pose), 7)
    check("z shifted by the offset", round(pose[2], 6), round(0.3 + b.ARM_ORIGIN_TO_BASE_Z, 6))
    check("x untouched", round(pose[0], 6), 0.1)
    check("quaternion stays wxyz", pose[3:], [1.0, 0.0, 0.0, 0.0], tol=0)

    # The neutral hand position asserted in bring-up step V4.
    neutral = b.to_base_frame(
        np.array([-0.085, 0.0, -0.14, 1.0, 0.0, 0.0, 0.0, 0.0]), b.ARM_ORIGIN_TO_BASE_Z
    )
    check(
        "FRAME_OFFSET_NECK -> base_link z",
        round(neutral[2], 6),
        round(b.ARM_ORIGIN_TO_BASE_Z - 0.14, 6),
        tol=1e-9,
    )

    print("== constants (openarm v1) ==================================")
    # Cross-check ARM_ORIGIN_TO_BASE_Z against the real v2 scene with
    # isaac/check_frames.py --mjcf <v2 scene>; 0.698 is an inference from both
    # models mounting their arms at that height, not a measurement.
    check("ARM_ORIGIN_TO_BASE_Z default", b.ARM_ORIGIN_TO_BASE_Z, 0.698)
    check("GRIPPER_STROKE_M", b.GRIPPER_STROKE_M, 0.044)
    check("arm joint names", len(b.ARM_JOINT_NAMES), 14)
    check("gripper joint names", len(b.GRIPPER_JOINT_NAMES), 4)
    check(
        "arm joint naming matches the URDF",
        b.ARM_JOINT_NAMES[0],
        "openarm_left_joint1",
    )
    check(
        "gripper joint naming matches the URDF",
        b.GRIPPER_JOINT_NAMES[0],
        "openarm_left_finger_joint1",
    )

    print("== UDP payload contract ====================================")
    # What isaac/isaaclab_teleop.py's TargetReceiver parses.
    payload = {
        "t": 1,
        "frame": b.EEF_FRAME_ID,
        "right": b.to_base_frame(np.zeros(8), b.ARM_ORIGIN_TO_BASE_Z),
        "left": b.to_base_frame(np.zeros(8), b.ARM_ORIGIN_TO_BASE_Z),
        "grip_right": 0.0,
        "grip_left": 0.0,
    }
    decoded = json.loads(json.dumps(payload))
    check("json round-trips", sorted(decoded), sorted(payload))
    check("right is 7 floats", len(decoded["right"]), 7)
    check("frame is base_link", decoded["frame"], "base_link")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
