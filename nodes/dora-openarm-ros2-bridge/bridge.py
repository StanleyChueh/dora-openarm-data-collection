#!/usr/bin/env python3


import argparse
import json
import socket
import time
from typing import Optional

import rclpy
from dora import Node as DoraNode
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import JointState


LEFT_ARM_JOINT_NAMES = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_ARM_JOINT_NAMES = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]


def arrow_to_float_list(value) -> list[float]:
    """將 Dora 的 Arrow Array 轉成 Python float list。"""
    return [float(item) for item in value.to_pylist()]


def pose_values_to_msg(values: list[float]) -> Pose:
    """將 [x, y, z, qw, qx, qy, qz] 轉成 geometry_msgs/Pose。"""
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = values[0], values[1], values[2]
    msg.orientation.w = values[3]
    msg.orientation.x = values[4]
    msg.orientation.y = values[5]
    msg.orientation.z = values[6]
    return msg


IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dora → ROS 2 OpenArm bridge.")
    parser.add_argument(
        "--vr-udp-host",
        type=str,
        default="127.0.0.1",
        help="Destination host for the optional VR teleop UDP JSON side-channel.",
    )
    parser.add_argument(
        "--vr-udp-port",
        type=int,
        default=0,
        help=(
            "If nonzero, best-effort UDP-broadcast the same eef_pose/gripper data as JSON"
            " to <vr-udp-host>:<vr-udp-port> on every update. Off by default. Intended to"
            " feed an external consumer (e.g. an Isaac Lab teleop device) that cannot import"
            " rclpy directly (Python ABI mismatch) -- this process never talks to that"
            " consumer except through this fire-and-forget socket."
        ),
    )
    return parser.parse_args()


class VrUdpBroadcaster:
    """Best-effort UDP JSON broadcaster of the latest eef_pose + gripper state.

    Fire-and-forget: never blocks and never raises into the dora event loop. Fields
    that haven't been received yet are sent as `null` (not a placeholder pose/value)
    so a consumer can distinguish "no VR data yet" from a real zero pose.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def broadcast(
        self,
        pose_right: Optional[list[float]],
        pose_left: Optional[list[float]],
        pose_reference: Optional[list[float]],
        gripper_right: Optional[float],
        gripper_left: Optional[float],
    ) -> None:
        packet = {
            "t": time.time(),
            "pose_right": pose_right,
            "pose_left": pose_left,
            "pose_reference": pose_reference,
            "gripper_right": gripper_right,
            "gripper_left": gripper_left,
        }
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break the bridge


def main() -> None:
    args = parse_args()

    rclpy.init()

    ros_node = rclpy.create_node("dora_openarm_ros2_bridge")
    publisher = ros_node.create_publisher(
        JointState,
        "/openarm/vr_joint_command",
        1,
    )
    eef_pose_publisher = ros_node.create_publisher(
        PoseArray,
        "/openarm/eef_pose",
        1,
    )
    gripper_publisher = ros_node.create_publisher(
        JointState,
        "/openarm/gripper_cmd",
        1,
    )

    vr_udp = VrUdpBroadcaster(args.vr_udp_host, args.vr_udp_port) if args.vr_udp_port else None

    dora_node = DoraNode()

    latest_left: Optional[list[float]] = None
    latest_right: Optional[list[float]] = None

    latest_pose_right: Optional[list[float]] = None
    latest_pose_left: Optional[list[float]] = None
    latest_pose_reference: Optional[list[float]] = None

    left_updated = False
    right_updated = False
    first_publish = True

    ros_node.get_logger().info(
        "Dora → ROS 2 bridge started: /openarm/vr_joint_command, "
        "/openarm/eef_pose (order: right, left, reference), "
        "/openarm/gripper_cmd"
        + (
            f", VR UDP JSON → {args.vr_udp_host}:{args.vr_udp_port}"
            if vr_udp is not None
            else ""
        )
    )

    try:
        for event in dora_node:
            if event["type"] == "STOP":
                break

            if event["type"] != "INPUT":
                continue

            input_id = event["id"]
            values = arrow_to_float_list(event["value"])

            if input_id == "position_left":
                if len(values) != 8:
                    ros_node.get_logger().warning(
                        f"position_left 應為 8 維，目前收到 {len(values)} 維"
                    )
                    continue

                latest_left = values
                left_updated = True

            elif input_id == "position_right":
                if len(values) != 8:
                    ros_node.get_logger().warning(
                        f"position_right 應為 8 維，目前收到 {len(values)} 維"
                    )
                    continue

                latest_right = values
                right_updated = True

            elif input_id in ("pose_right", "pose_left", "pose_reference"):
                if len(values) != 7:
                    ros_node.get_logger().warning(
                        f"{input_id} 應為 7 維 (xyz + quat)，目前收到 {len(values)} 維"
                    )
                    continue

                if input_id == "pose_right":
                    latest_pose_right = values
                elif input_id == "pose_left":
                    latest_pose_left = values
                else:
                    latest_pose_reference = values

                pose_array_msg = PoseArray()
                pose_array_msg.header.stamp = ros_node.get_clock().now().to_msg()
                # 固定順序：right, left, reference。尚未收到的一側先以原點姿態填補。
                pose_array_msg.poses = [
                    pose_values_to_msg(latest_pose_right or IDENTITY_POSE),
                    pose_values_to_msg(latest_pose_left or IDENTITY_POSE),
                    pose_values_to_msg(latest_pose_reference or IDENTITY_POSE),
                ]

                eef_pose_publisher.publish(pose_array_msg)
                rclpy.spin_once(ros_node, timeout_sec=0.0)

                if vr_udp is not None:
                    vr_udp.broadcast(
                        pose_right=latest_pose_right,
                        pose_left=latest_pose_left,
                        pose_reference=latest_pose_reference,
                        gripper_right=latest_right[7] if latest_right else None,
                        gripper_left=latest_left[7] if latest_left else None,
                    )
                continue

            else:
                continue

            # 等左右兩側都收到一筆新資料後再發布
            if (
                latest_left is None
                or latest_right is None
                or not left_updated
                or not right_updated
            ):
                continue

            msg = JointState()
            msg.header.stamp = ros_node.get_clock().now().to_msg()

            # 第一版只發布左右手臂各 7 軸。
            # Dora 每側第 8 個值是 gripper，之後再確認兩指映射。
            msg.name = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
            msg.position = latest_left[:7] + latest_right[:7]

            publisher.publish(msg)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            gripper_msg = JointState()
            gripper_msg.header.stamp = msg.header.stamp
            gripper_msg.name = ["gripper_left", "gripper_right"]
            gripper_msg.position = [latest_left[7], latest_right[7]]

            gripper_publisher.publish(gripper_msg)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            if vr_udp is not None:
                vr_udp.broadcast(
                    pose_right=latest_pose_right,
                    pose_left=latest_pose_left,
                    pose_reference=latest_pose_reference,
                    gripper_right=latest_right[7],
                    gripper_left=latest_left[7],
                )

            left_updated = False
            right_updated = False

            if first_publish:
                ros_node.get_logger().info(
                    "已發布第一筆 OpenArm 14 軸 JointState"
                )
                first_publish = False

    finally:
        ros_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()