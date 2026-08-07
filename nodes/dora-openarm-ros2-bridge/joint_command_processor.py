#!/usr/bin/env python3
"""ROS 2 node that post-processes /openarm/vr_joint_command for OpenArm control.

Subscribes to the 14-joint JointState published by bridge.py on
/openarm/vr_joint_command (order: openarm_left_joint1..7, openarm_right_joint1..7),
and republishes a processed copy on /openarm/vr_joint_command_processed with:

  - joint6/joint7 swapped per arm (joint6 <- old joint7, joint7 <- old joint6)
  - both of those swapped values negated
  - the latest gripper_left/gripper_right (from /openarm/gripper_cmd) interleaved in,
    each right after its own arm's 7 joints, named as the robot's REAL gripper joint
    (openarm_left_finger_joint1 / openarm_right_finger_joint1 -- see
    isaaclab_assets/robots/openarm.py) rather than a synthetic "gripper_left"/
    "gripper_right" label, so Isaac Sim's Articulation Controller node (which resolves
    each Joint Names entry against the robot's actual USD joints) can find it. Output
    order is [left_joint1..7, left_finger_joint1, right_joint1..7, right_finger_joint1]
    -- 8 joints per arm -- matching record_demos_openarm.py's ActionsCfg field order
    (arm_action, gripper_action, right_arm_action, right_gripper_action).

    CAVEAT: the gripper value forwarded here is /openarm/gripper_cmd's raw
    trigger-mapped angle (0..0.785 rad, see record_demos_openarm.py's
    VRDualArmTeleop.GRIPPER_RAW_RANGE) -- it is NOT yet rescaled to
    openarm_left_finger_joint1's real prismatic travel (0..0.044 m, see
    OpenArmKeyboard/JointMirrorBroadcaster's GRIPPER_OPEN_VAL/GRIPPER_CLOSED_VAL).
    Isaac Lab's own BinaryJointPositionActionCfg path never hits this problem because
    it only uses the sign of a further-derived +-1 command, not this raw angle,
    directly. Feeding this raw value straight into an Articulation Controller's
    Position Command will drive the finger joint's target far past its real limit
    (clamped there by the joint's own limits, but not usefully proportional to trigger
    squeeze) until this is rescaled -- e.g. finger_target_m = (raw_rad / 0.785) * 0.044.

Runs as a dora node (like bridge.py) purely so dora schedules/manages its process and
feeds it a `tick` input to drain -- the actual work happens over ROS 2 topics, not
dora's dataflow IPC, and this node has no data dependency on any other dora node.
Draining `tick` matters: an undrained dora input queue fills up and applies backpressure
to its source (quittable-tick-leader), which stalls every other node scheduled off that
same tick (udp-receiver, ik) -- an early version of this script called plain
`rclpy.spin()` and never touched the dora API, which froze the whole dataflow after the
queue filled. Needs the same ROS 2 Humble / Python 3.10 environment as bridge.py (see
run_processor.sh), because rclpy's C extension only ships for that ABI.
"""

import rclpy
from dora import Node as DoraNode
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Indices within vr_joint_command's 14-entry name/position arrays -- see bridge.py's
# LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES order (7 left joints, then 7 right).
LEFT_JOINT6_IDX = 5
LEFT_JOINT7_IDX = 6
RIGHT_JOINT6_IDX = 12
RIGHT_JOINT7_IDX = 13
EXPECTED_LEN = 14

# Real robot joint names (isaaclab_assets/robots/openarm.py) -- used instead of the
# gripper_cmd topic's "gripper_left"/"gripper_right" labels so Isaac Sim's Articulation
# Controller can resolve them against the actual USD articulation.
LEFT_GRIPPER_JOINT_NAME = "openarm_left_finger_joint1"
RIGHT_GRIPPER_JOINT_NAME = "openarm_right_finger_joint1"


class JointCommandProcessor(Node):
    def __init__(self):
        super().__init__("openarm_vr_joint_command_processor")

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
            "Publishing /openarm/vr_joint_command_processed (joint6/joint7 swapped+negated"
            f" per arm; order [left_joint1..7, {LEFT_GRIPPER_JOINT_NAME}, right_joint1..7,"
            f" {RIGHT_GRIPPER_JOINT_NAME}])"
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

        for j6_idx, j7_idx in ((LEFT_JOINT6_IDX, LEFT_JOINT7_IDX), (RIGHT_JOINT6_IDX, RIGHT_JOINT7_IDX)):
            old_j6, old_j7 = positions[j6_idx], positions[j7_idx]
            positions[j6_idx] = -old_j7
            positions[j7_idx] = -old_j6

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
            out_positions.append(-self._latest_gripper_right)

        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = out_names
        out.position = out_positions
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    ros_node = JointCommandProcessor()
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
