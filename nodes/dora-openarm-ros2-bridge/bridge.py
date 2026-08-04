#!/usr/bin/env python3


from typing import Optional

import rclpy
from dora import Node as DoraNode
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


def main() -> None:
    rclpy.init()

    ros_node = rclpy.create_node("dora_openarm_ros2_bridge")
    publisher = ros_node.create_publisher(
        JointState,
        "/openarm/vr_joint_command",
        1,
    )

    dora_node = DoraNode()

    latest_left: Optional[list[float]] = None
    latest_right: Optional[list[float]] = None

    left_updated = False
    right_updated = False
    first_publish = True

    ros_node.get_logger().info(
        "Dora → ROS 2 bridge started: /openarm/vr_joint_command"
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