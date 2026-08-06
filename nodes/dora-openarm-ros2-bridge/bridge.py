#!/usr/bin/env python3
"""Dora -> ROS 2 bridge for the OpenArm VR teleoperation pipeline.

Published ROS 2 topics
----------------------
/openarm/eef_pose           geometry_msgs/PoseArray    [right, left] EE targets
/openarm/eef_pose/right     geometry_msgs/PoseStamped  slot 0, for rviz
/openarm/eef_pose/left      geometry_msgs/PoseStamped  slot 1, for rviz
/openarm/vr_reference_pose  geometry_msgs/PoseStamped  raw headset reference pose
/openarm/vr_joint_command   sensor_msgs/JointState     14 arm joints (dora `ik`)
/openarm/gripper_cmd        sensor_msgs/JointState     4 finger joints, metres

UDP side-channel (for Isaac Lab)
--------------------------------
The same EE targets go out as one JSON datagram per publish to
ISAAC_UDP_HOST:ISAAC_UDP_PORT.  Set ISAAC_UDP_PORT=0 to disable.

    {"t": <ns>, "frame": "base_link",
     "right": [px,py,pz,qw,qx,qy,qz], "left": [...],
     "grip_right": <m>, "grip_left": <m>}

isaac/isaaclab_teleop.py parses exactly this.

Why a side-channel at all: Isaac Lab's DifferentialIKController needs the
Jacobian out of the PhysX articulation view, so it can only run inside Isaac
Sim's interpreter -- Python 3.11 on Isaac Sim 5.x, while ROS 2 Humble's rclpy is
built against 3.10.  `import rclpy` there fails and no PYTHONPATH fixes it.  A
stdlib UDP socket has no such problem, and the payload is 16 floats at ~100 Hz.
The ROS topics stay for rviz and `ros2 topic echo`.

Frames
------
`udp-receiver` emits pose_right / pose_left in the `arm_origin` frame -- a
chest-level frame midway between the two arm mounts, defined as a site in the
MuJoCo scene the dora `ik` node loads.  Both consumers want the robot base
frame, so the translation is applied here:

    p_base = p_arm_origin + [0, 0, ARM_ORIGIN_TO_BASE_Z]
    q_base = q_arm_origin

ARM_ORIGIN_TO_BASE_Z IS NOT DERIVABLE FROM THE ISAAC ROBOT.  dora runs OpenArm
v2 (dora_openarm_mujoco/main.py:132 hardcodes `import openarm_mujoco.v2`) while
Isaac runs v1, so this number bridges two different models.  It defaults to
0.698 because both v1 and v2 mount their arms at xyz="0 +-0.031 0.698" from
openarm_body_link0 -- an inference, not a measurement.

    MEASURE IT:  python3 isaac/check_frames.py --scene demo
                 then export ARM_ORIGIN_TO_BASE_Z=<the printed z>

The rotation is assumed identity.  v1's arm mount joints carry rpy="-+1.5708 0 0"
where v2's are rpy="0 0 0", so if the two models' link7 frames differ the VR
quaternion convention (quest_receiver.py:159, tuned against v2) will not carry
over.  Bring-up step V5 in isaac/README.md is where that shows up.

`pose_reference` does NOT go through that rectification -- quest_receiver.py:164
calls pose_to_array() directly instead of _rectify() -- so it is in the
Quest-derived headset world frame, not arm_origin.  It gets its own topic, its
own frame_id, and NO z offset.  A PoseArray carries a single header, so mixing
it in with the two arm poses would be unrepresentable.

Units
-----
quest_receiver._map_trigger_to_gripper emits RADIANS, sized for OpenArm v2's
revolute fingers.  OpenArm v1's fingers are PRISMATIC with a 0.044 m stroke, so
the value is rescaled here -- see gripper_rad_to_stroke().

The rescale lives in this file rather than in quest_receiver.py for two reasons:
nodes/dora-openarm-vr is a git submodule (upstream Enactic code), and the dora
consumers (ik, mujoco-collect, recorder) read the dora stream directly and never
subscribe to a ROS topic or this UDP socket -- so a change made here
structurally cannot reach the MuJoCo/dataset path.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Optional

import numpy as np
import pyarrow as pa
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from dora import Node as DoraNode


# openarm v1's joint names already match what the dora pipeline hardcodes, and
# what the converted USD contains -- verified against
# configuration/v1_camera_isaac_physics.usd, which carries all 18 of these.
ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)] + [
    f"openarm_right_joint{i}" for i in range(1, 8)
]

# openarm v1 fingers: prismatic, range [0, +0.044] on all four, and
# finger_joint2 carries <mimic joint="finger_joint1"/> with the default
# multiplier 1 / offset 0.  So all four take the same positive value.
GRIPPER_JOINT_NAMES = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]

ARM_ORIGIN_TO_BASE_Z = float(os.environ.get("ARM_ORIGIN_TO_BASE_Z", "0.698"))
GRIPPER_RAD_FULL_OPEN = 1.57 / 2.0  # |_map_trigger_to_gripper| at trigger = 0
GRIPPER_STROKE_M = 0.044  # urdf openarm_*_finger_joint* travel

EEF_FRAME_ID = os.environ.get("EEF_FRAME_ID", "base_link")
REFERENCE_FRAME_ID = os.environ.get("REFERENCE_FRAME_ID", "quest_world")
EEF_RATE_HZ = float(os.environ.get("EEF_RATE_HZ", "100"))
REFERENCE_RATE_HZ = float(os.environ.get("REFERENCE_RATE_HZ", "10"))
EEF_STALE_SEC = float(os.environ.get("EEF_STALE_SEC", "0.25"))
GRIPPER_MODE = os.environ.get("GRIPPER_MODE", "v1_prismatic")  # or "v2_radian"
EEF_QOS = os.environ.get("EEF_QOS", "reliable")  # or "best_effort"

ISAAC_UDP_HOST = os.environ.get("ISAAC_UDP_HOST", "127.0.0.1")
ISAAC_UDP_PORT = int(os.environ.get("ISAAC_UDP_PORT", "5007"))


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is.

    Same helper as ik.py:61, fk.py:48 and mujoco/main.py:187.  The dora nodes
    wrap their payloads as [{"pose": [...]}] / [{"qpos": [...]}], so a plain
    float() over to_pylist() gets a dict and raises TypeError.
    """
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


def to_base_frame(values: np.ndarray, z_offset: float) -> list[float]:
    """[px,py,pz,qw,qx,qy,qz,...] in arm_origin -> 7 floats in base_link."""
    return [
        float(values[0]),
        float(values[1]),
        float(values[2]) + z_offset,
        float(values[3]),
        float(values[4]),
        float(values[5]),
        float(values[6]),
    ]


def fill_pose(msg: Pose, pose: list[float]) -> Pose:
    msg.position.x, msg.position.y, msg.position.z = pose[0], pose[1], pose[2]
    msg.orientation.w = pose[3]
    msg.orientation.x = pose[4]
    msg.orientation.y = pose[5]
    msg.orientation.z = pose[6]
    return msg


def gripper_rad_to_stroke(angle_rad: float) -> float:
    """v2 revolute gripper command (rad) -> v1 prismatic finger travel (m).

    _map_trigger_to_gripper maps a fully squeezed trigger to 0 rad and a
    released trigger to +-0.785 rad, so |rad| means "how open".  v1's finger
    range is [0, 0.044] with 0 = closed, so the magnitude maps straight across.
    """
    frac = min(abs(float(angle_rad)) / GRIPPER_RAD_FULL_OPEN, 1.0)
    return frac * GRIPPER_STROKE_M


def _qos(depth: int = 1) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if EEF_QOS == "best_effort"
        else ReliabilityPolicy.RELIABLE
    )
    return QoSProfile(
        reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=depth
    )


def main() -> None:
    rclpy.init()
    ros_node = rclpy.create_node("dora_openarm_ros2_bridge")
    log = ros_node.get_logger()

    joint_pub = ros_node.create_publisher(
        JointState, "/openarm/vr_joint_command", _qos()
    )
    gripper_pub = ros_node.create_publisher(JointState, "/openarm/gripper_cmd", _qos())
    eef_pub = ros_node.create_publisher(PoseArray, "/openarm/eef_pose", _qos())
    eef_right_pub = ros_node.create_publisher(
        PoseStamped, "/openarm/eef_pose/right", _qos()
    )
    eef_left_pub = ros_node.create_publisher(
        PoseStamped, "/openarm/eef_pose/left", _qos()
    )
    ref_pub = ros_node.create_publisher(
        PoseStamped, "/openarm/vr_reference_pose", _qos()
    )

    udp_sock = None
    if ISAAC_UDP_PORT:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    dora_node = DoraNode()

    latest_left: Optional[np.ndarray] = None
    latest_right: Optional[np.ndarray] = None
    left_updated = False
    right_updated = False
    first_joint_publish = True

    pose_right: Optional[np.ndarray] = None
    pose_left: Optional[np.ndarray] = None
    pose_right_t = 0.0
    pose_left_t = 0.0
    grip_right_m = 0.0
    grip_left_m = 0.0
    first_eef_publish = True
    udp_warned = False

    eef_period = 1.0 / EEF_RATE_HZ if EEF_RATE_HZ > 0.0 else 0.0
    ref_period = 1.0 / REFERENCE_RATE_HZ if REFERENCE_RATE_HZ > 0.0 else 0.0
    next_eef_publish = 0.0
    next_ref_publish = 0.0

    log.info(
        f"bridge up: eef frame_id='{EEF_FRAME_ID}', z_offset={ARM_ORIGIN_TO_BASE_Z}, "
        f"rate<={EEF_RATE_HZ:.0f} Hz, gripper={GRIPPER_MODE}, "
        f"udp={'off' if not ISAAC_UDP_PORT else f'{ISAAC_UDP_HOST}:{ISAAC_UDP_PORT}'}"
    )

    try:
        for event in dora_node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue

            input_id = event["id"]

            # ── Cartesian EE targets ─────────────────────────────────────────
            if input_id in ("pose_right", "pose_left"):
                values = extract_values(event["value"], "pose")
                # pose_right/pose_left carry the gripper angle in slot 7, so
                # they are 8 wide.  Only pose_reference is 7.
                if values.shape != (8,):
                    log.warning(
                        f"{input_id}: expected 8 values [xyz + quat + gripper], "
                        f"got {values.shape}"
                    )
                    continue

                now = time.monotonic()
                if input_id == "pose_right":
                    pose_right, pose_right_t = values, now
                    grip_right_m = gripper_rad_to_stroke(values[7])
                else:
                    pose_left, pose_left_t = values, now
                    grip_left_m = gripper_rad_to_stroke(values[7])

                # Never pad a missing side with an identity pose: identity in
                # base_link is inside the robot's own pedestal.
                if pose_right is None or pose_left is None:
                    continue
                # If a controller drops out (validity INVALID), stop publishing
                # rather than hold a ghost target.
                if (
                    now - pose_right_t > EEF_STALE_SEC
                    or now - pose_left_t > EEF_STALE_SEC
                ):
                    continue
                # udp-receiver ticks at 500 Hz per side and repeats the last UDP
                # packet when no new one arrived, so most events are duplicates.
                if now < next_eef_publish:
                    continue
                next_eef_publish = now + eef_period

                right_base = to_base_frame(pose_right, ARM_ORIGIN_TO_BASE_Z)
                left_base = to_base_frame(pose_left, ARM_ORIGIN_TO_BASE_Z)
                stamp = ros_node.get_clock().now().to_msg()

                array_msg = PoseArray()
                array_msg.header.stamp = stamp
                array_msg.header.frame_id = EEF_FRAME_ID
                array_msg.poses = [
                    fill_pose(Pose(), right_base),
                    fill_pose(Pose(), left_base),
                ]
                eef_pub.publish(array_msg)

                for pub, pose in (
                    (eef_right_pub, right_base),
                    (eef_left_pub, left_base),
                ):
                    stamped = PoseStamped()
                    stamped.header.stamp = stamp
                    stamped.header.frame_id = EEF_FRAME_ID
                    fill_pose(stamped.pose, pose)
                    pub.publish(stamped)

                rclpy.spin_once(ros_node, timeout_sec=0.0)

                if udp_sock is not None:
                    payload = {
                        "t": time.time_ns(),
                        "frame": EEF_FRAME_ID,
                        "right": right_base,
                        "left": left_base,
                        "grip_right": grip_right_m,
                        "grip_left": grip_left_m,
                    }
                    try:
                        udp_sock.sendto(
                            json.dumps(payload).encode("utf-8"),
                            (ISAAC_UDP_HOST, ISAAC_UDP_PORT),
                        )
                    except OSError as exc:
                        # A dead receiver must never take the bridge down.
                        if not udp_warned:
                            log.warning(f"isaac udp send failed: {exc}")
                            udp_warned = True

                if first_eef_publish:
                    log.info(
                        f"/openarm/eef_pose live: [right, left] in '{EEF_FRAME_ID}'"
                        + (
                            f", udp -> {ISAAC_UDP_HOST}:{ISAAC_UDP_PORT}"
                            if udp_sock is not None
                            else ""
                        )
                    )
                    first_eef_publish = False
                continue

            # ── headset reference: different frame, own topic, no z offset ───
            if input_id == "pose_reference":
                values = extract_values(event["value"], "pose")
                if values.shape != (7,):
                    log.warning(
                        f"pose_reference: expected 7 values [xyz + quat], "
                        f"got {values.shape}"
                    )
                    continue

                now = time.monotonic()
                if now < next_ref_publish:
                    continue
                next_ref_publish = now + ref_period

                ref_msg = PoseStamped()
                ref_msg.header.stamp = ros_node.get_clock().now().to_msg()
                ref_msg.header.frame_id = REFERENCE_FRAME_ID
                fill_pose(ref_msg.pose, to_base_frame(values, 0.0))
                ref_pub.publish(ref_msg)
                rclpy.spin_once(ros_node, timeout_sec=0.0)
                continue

            # ── joint solutions from the dora `ik` node ──────────────────────
            if input_id == "position_left":
                values = extract_values(event["value"], "qpos")
                if values.shape != (8,):
                    log.warning(f"position_left: expected 8, got {values.shape}")
                    continue
                latest_left, left_updated = values, True
            elif input_id == "position_right":
                values = extract_values(event["value"], "qpos")
                if values.shape != (8,):
                    log.warning(f"position_right: expected 8, got {values.shape}")
                    continue
                latest_right, right_updated = values, True
            else:
                continue

            # Publish only once both sides are fresh, so left and right in one
            # JointState always come from the same solver step.
            if (
                latest_left is None
                or latest_right is None
                or not left_updated
                or not right_updated
            ):
                continue

            stamp = ros_node.get_clock().now().to_msg()

            joint_msg = JointState()
            joint_msg.header.stamp = stamp
            joint_msg.name = list(ARM_JOINT_NAMES)
            # np.concatenate, not `+`: these are numpy arrays, so `+` would add
            # element-wise instead of concatenating.
            joint_msg.position = (
                np.concatenate((latest_left[:7], latest_right[:7]))
                .astype(float)
                .tolist()
            )
            joint_pub.publish(joint_msg)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            gripper_msg = JointState()
            gripper_msg.header.stamp = stamp
            if GRIPPER_MODE == "v2_radian":
                gripper_msg.name = ["gripper_left", "gripper_right"]
                gripper_msg.position = [
                    float(latest_left[7]),
                    float(latest_right[7]),
                ]
            else:
                stroke_l = gripper_rad_to_stroke(latest_left[7])
                stroke_r = gripper_rad_to_stroke(latest_right[7])
                gripper_msg.name = list(GRIPPER_JOINT_NAMES)
                gripper_msg.position = [stroke_l, stroke_l, stroke_r, stroke_r]
            gripper_pub.publish(gripper_msg)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            left_updated = False
            right_updated = False

            if first_joint_publish:
                log.info("First 14-axis JointState published")
                first_joint_publish = False

    finally:
        if udp_sock is not None:
            udp_sock.close()
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
