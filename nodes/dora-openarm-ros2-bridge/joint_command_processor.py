#!/usr/bin/env python3
"""ROS 2 node that post-processes /openarm/vr_joint_command for OpenArm control.

Subscribes to the 14-joint JointState published by bridge.py on
/openarm/vr_joint_command (order: openarm_left_joint1..7, openarm_right_joint1..7),
and republishes a processed copy on /openarm/vr_joint_command_processed with:

  - the arm joint values passed through unchanged. They used to be remapped here
    (joint6/joint7 swapped and negated per arm) back when the ik node still solved
    against the v2 model while Isaac Sim ran the custom v1_camera robot, whose wrist
    composes the same two perpendicular axes in the opposite order. That remap was
    never actually valid: reordering two non-commuting rotations cannot be undone by
    permuting or negating their joint values, and measuring it across random arm
    configurations showed 19-86 degrees of residual end-effector orientation error
    (it happened to be exact only when joint6 alone moved, which is presumably how it
    passed a first eyeball check). The ik node now solves against the v1_camera model
    directly (--xml, see dataflow-vr-mujoco-ros2.yaml), so its output is already in
    the convention Isaac's v1_camera USD expects and must NOT be remapped again.
  - the latest gripper_left/gripper_right (from /openarm/gripper_cmd) interleaved in,
    each right after its own arm's 7 joints, named as the robot's REAL gripper joint
    (openarm_left_finger_joint1 / openarm_right_finger_joint1 -- see
    isaaclab_assets/robots/openarm.py) rather than a synthetic "gripper_left"/
    "gripper_right" label, so Isaac Sim's Articulation Controller node (which resolves
    each Joint Names entry against the robot's actual USD joints) can find it. Output
    order is [left_joint1..7, left_finger_joint1, right_joint1..7, right_finger_joint1]
    -- 8 joints per arm -- matching record_demos_openarm.py's ActionsCfg field order
    (arm_action, gripper_action, right_arm_action, right_gripper_action).

    Both gripper values are forwarded verbatim, and are already in the finger joint's
    real prismatic travel (0 m closed .. 0.044 m open, matching
    JointMirrorBroadcaster's GRIPPER_CLOSED_VAL/GRIPPER_OPEN_VAL) because the ik node
    derives its trigger mapping from whichever model --xml loads. Two earlier
    corrections for the v2 model's hinge gripper are therefore gone: the rad->m
    rescale this docstring used to prescribe, and a negation of the right-hand value
    (v2's right finger range is [-0.785, 0], so it needed flipping; v1_camera's is
    [0, 0.044] on both sides, and negating it would drive the target below its lower
    limit and jam that gripper shut).

Runs as a dora node (like bridge.py) purely so dora schedules/manages its process and
feeds it a `tick` input to drain -- the actual work happens over ROS 2 topics, not
dora's dataflow IPC, and this node has no data dependency on any other dora node.
Draining `tick` matters: an undrained dora input queue fills up and applies backpressure
to its source (quittable-tick-leader), which stalls every other node scheduled off that
same tick (udp-receiver, ik) -- an early version of this script called plain
`rclpy.spin()` and never touched the dora API, which froze the whole dataflow after the
queue filled. Needs the same ROS 2 Humble / Python 3.10 environment as bridge.py (see
run_processor.sh), because rclpy's C extension only ships for that ABI.

If --vr-joint-udp-port is set, also fire-and-forget UDP-broadcasts the same processed
name/position arrays as JSON -- mirrors bridge.py's VrUdpBroadcaster, for the same
reason: Isaac Lab's conda env is Python 3.11, but rclpy is only built for the system's
Python 3.10, so record_demos_openarm.py's joint-space teleop device can't subscribe to
this node's ROS 2 topic directly and needs this side-channel instead.
"""

import argparse
import json
import socket
import time
from typing import Optional

import rclpy
from dora import Node as DoraNode
from rclpy.node import Node
from sensor_msgs.msg import JointState

# vr_joint_command carries 14 entries -- see bridge.py's LEFT_ARM_JOINT_NAMES +
# RIGHT_ARM_JOINT_NAMES order (7 left joints, then 7 right).
EXPECTED_LEN = 14

# Real robot joint names (isaaclab_assets/robots/openarm.py) -- used instead of the
# gripper_cmd topic's "gripper_left"/"gripper_right" labels so Isaac Sim's Articulation
# Controller can resolve them against the actual USD articulation.
LEFT_GRIPPER_JOINT_NAME = "openarm_left_finger_joint1"
RIGHT_GRIPPER_JOINT_NAME = "openarm_right_finger_joint1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process /openarm/vr_joint_command.")
    parser.add_argument(
        "--vr-joint-udp-host",
        type=str,
        default="127.0.0.1",
        help="Destination host for the optional joint-command UDP JSON side-channel.",
    )
    parser.add_argument(
        "--vr-joint-udp-port",
        type=int,
        default=0,
        help=(
            "If nonzero, best-effort UDP-broadcast the processed name/position arrays as"
            " JSON to <vr-joint-udp-host>:<vr-joint-udp-port> on every update. Off by"
            " default. Intended for an Isaac Lab joint-space teleop device that cannot"
            " import rclpy directly (Python ABI mismatch)."
        ),
    )
    return parser.parse_args()


class VrJointUdpBroadcaster:
    """Best-effort UDP JSON broadcaster of the processed joint command.

    Fire-and-forget, mirrors bridge.py's VrUdpBroadcaster: never blocks and never
    raises into the ROS 2 callback that calls it.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def broadcast(self, name: list[str], position: list[float]) -> None:
        packet = {"t": time.time(), "name": name, "position": position}
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break the node


class JointCommandProcessor(Node):
    def __init__(self, vr_joint_udp: Optional[VrJointUdpBroadcaster] = None):
        super().__init__("openarm_vr_joint_command_processor")

        self._vr_joint_udp = vr_joint_udp
        self._latest_gripper_left: float | None = None
        self._latest_gripper_right: float | None = None

        self._gripper_sub = self.create_subscription(
            JointState, "/openarm/gripper_cmd", self._on_gripper, 1
        )
        self._joint_sub = self.create_subscription(
            JointState, "/openarm/vr_joint_command", self._on_joint_command, 1
        )
        self._pub = self.create_publisher(JointState, "/openarm/vr_joint_command_processed", 1)

        self.get_logger().info(
            "Publishing /openarm/vr_joint_command_processed (arm joints passed through;"
            f" order [left_joint1..7, {LEFT_GRIPPER_JOINT_NAME}, right_joint1..7,"
            f" {RIGHT_GRIPPER_JOINT_NAME}])"
            + (f", VR joint UDP JSON -> {vr_joint_udp._addr}" if vr_joint_udp is not None else "")
        )

    def _on_gripper(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name == "gripper_left":
                self._latest_gripper_left = pos
            elif name == "gripper_right":
                self._latest_gripper_right = pos

    def _on_joint_command(self, msg: JointState) -> None:
        if len(msg.position) != EXPECTED_LEN:
            self.get_logger().warning(
                f"Expected {EXPECTED_LEN} joint positions on /openarm/vr_joint_command,"
                f" got {len(msg.position)} -- skipping"
            )
            return

        names = list(msg.name)
        positions = list(msg.position)

        # Interleave each gripper right after its own arm's 7 joints (arm_action,
        # gripper_action, right_arm_action, right_gripper_action order) instead of
        # appending both at the end. A side's gripper entry is omitted entirely if no
        # /openarm/gripper_cmd has arrived for it yet.
        out_names = names[:7]
        out_positions = positions[:7]
        if self._latest_gripper_left is not None:
            out_names.append(LEFT_GRIPPER_JOINT_NAME)
            out_positions.append(self._latest_gripper_left)
        out_names += names[7:14]
        out_positions += positions[7:14]
        if self._latest_gripper_right is not None:
            out_names.append(RIGHT_GRIPPER_JOINT_NAME)
            out_positions.append(self._latest_gripper_right)

        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = out_names
        out.position = out_positions
        self._pub.publish(out)

        if self._vr_joint_udp is not None:
            self._vr_joint_udp.broadcast(out_names, out_positions)


def main() -> None:
    args = parse_args()
    vr_joint_udp = (
        VrJointUdpBroadcaster(args.vr_joint_udp_host, args.vr_joint_udp_port)
        if args.vr_joint_udp_port
        else None
    )

    rclpy.init()
    ros_node = JointCommandProcessor(vr_joint_udp=vr_joint_udp)
    dora_node = DoraNode()

    try:
        for event in dora_node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            # `tick` carries no data we need -- receiving it is what drains dora's
            # queue so upstream doesn't back up. The actual work runs in ROS 2
            # subscription callbacks, serviced here via spin_once.
            rclpy.spin_once(ros_node, timeout_sec=0.0)
    finally:
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
