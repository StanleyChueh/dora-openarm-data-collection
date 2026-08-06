#!/usr/bin/env python3
"""Isaac Lab teleop: VR end-effector targets -> DifferentialIKController -> arms.

    ${ISAACLAB}/isaaclab.sh -p isaac/isaaclab_teleop.py \
        --usd ~/Ivan_ws/openarm_description-main/v1_camera_isaac.urdf.usd

Consumes the UDP side-channel the dora ros2-bridge emits (default 127.0.0.1:5007):

    {"t": <ns>, "frame": "base_link",
     "right": [px,py,pz,qw,qx,qy,qz], "left": [...],
     "grip_right": <m>, "grip_left": <m>}

Why UDP and not rclpy
---------------------
DifferentialIKController needs the Jacobian out of the PhysX articulation view,
so it can only run inside Isaac Sim's interpreter.  That is Python 3.11 on
Isaac Sim 5.x, while ROS 2 Humble's rclpy is built against 3.10, so `import
rclpy` here fails outright.  A stdlib UDP socket sidesteps it; the bridge still
publishes the equivalent ROS topics for rviz and `ros2 topic echo`.

Two controllers, not one
------------------------
A DifferentialIKController instance drives a single end-effector -- its tensors
are batched over environments, not over arms.  Bimanual therefore means one
controller per arm, which is also more predictable than a coupled solve: each
arm's redundancy is resolved independently and neither can steal error from the
other.

Frames
------
Commands must be in the ROBOT BASE frame, which is exactly what the bridge
publishes (it applies the arm_origin -> base translation).  The current EE pose
is converted world -> base with subtract_frame_transforms().

Tool offset
-----------
By default IK drives `openarm_{side}_link7` -- a real rigid body that is
guaranteed to appear in the articulation and in the Jacobian -- and the incoming
grasp-frame target is walked back along link7's +z by --tool-offset-z.  The
`openarm_{side}_grasp` frame the URDF generator appends is massless, and a
massless fixed frame may or may not survive USD import as its own body, so
relying on it for the Jacobian is a portability risk.  If your import does keep
it, `--ee-body-fmt 'openarm_{side}_grasp' --tool-offset-z 0` is equivalent and
skips the extra transform.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="USD of the OpenArm v1 robot")
parser.add_argument("--udp-host", default="0.0.0.0")
parser.add_argument("--udp-port", type=int, default=5007)
parser.add_argument(
    "--ee-body-fmt",
    default="openarm_{side}_link7",
    help="articulation body IK drives; must contain {side}",
)
parser.add_argument(
    "--tool-offset-z",
    type=float,
    default=0.165,
    help="grasp point offset along the ee body's +z (0 if --ee-body-fmt is the "
    "grasp frame itself)",
)
parser.add_argument("--lambda-val", type=float, default=0.05, help="DLS damping")
parser.add_argument(
    "--dry-run", action="store_true", help="solve and log but do not drive the arms"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The simulation app must exist before any isaaclab.* / omni.* import.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    subtract_frame_transforms,
)

SIDES = ("right", "left")

# openarm v1 joint4 has range [0, +2.4435], so an all-zero start would sit
# exactly on a limit; joint2's limits are mirrored between the arms.
HOME = {
    "openarm_left_joint2": -0.30,
    "openarm_right_joint2": 0.30,
    "openarm_left_joint4": 1.20,
    "openarm_right_joint4": 1.20,
}
GRIPPER_JOINT_RE = "openarm_{side}_finger_joint.*"
ARM_JOINT_RE = "openarm_{side}_joint.*"


class TargetReceiver:
    """Background UDP reader that keeps only the freshest datagram.

    Same shape as nodes/dora-openarm-vr/.../udp_receiver.py: a teleop loop wants
    the newest target, never a backlog, so queued datagrams are drained and
    discarded rather than processed.
    """

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._seq = 0
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[teleop] listening for EE targets on UDP {host}:{port}")

    def latest(self) -> tuple[dict | None, int]:
        with self._lock:
            return self._latest, self._seq

    def close(self) -> None:
        self._running = False
        self._sock.close()

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except (TimeoutError, OSError):
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            with self._lock:
                self._latest = msg
                self._seq += 1


@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.usd),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(HOME)),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["openarm_.*_joint[1-7]"],
                stiffness=800.0,
                damping=40.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_.*_finger_joint.*"],
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


class ArmIK:
    """One DifferentialIKController plus the index bookkeeping for one arm."""

    def __init__(self, side: str, scene: InteractiveScene, device: str) -> None:
        self.side = side
        robot: Articulation = scene["robot"]

        self.arm = SceneEntityCfg(
            "robot",
            joint_names=[ARM_JOINT_RE.format(side=side)],
            body_names=[args_cli.ee_body_fmt.format(side=side)],
        )
        self.arm.resolve(scene)
        self.fingers = SceneEntityCfg(
            "robot", joint_names=[GRIPPER_JOINT_RE.format(side=side)]
        )
        self.fingers.resolve(scene)

        # The root body is not included in the returned Jacobians, so a
        # fixed-base robot's frame index is one less than its body index.
        self.jacobi_idx = (
            self.arm.body_ids[0] - 1 if robot.is_fixed_base else self.arm.body_ids[0]
        )

        cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": args_cli.lambda_val},
        )
        self.ctrl = DifferentialIKController(
            cfg, num_envs=scene.num_envs, device=device
        )

        n = scene.num_envs
        self.tool_offset_pos = torch.zeros(n, 3, device=device)
        self.tool_offset_pos[:, 2] = -args_cli.tool_offset_z
        self.tool_offset_quat = torch.zeros(n, 4, device=device)
        self.tool_offset_quat[:, 0] = 1.0  # identity, wxyz

        print(
            f"[teleop] {side:5s} joints={len(self.arm.joint_ids)} "
            f"body={args_cli.ee_body_fmt.format(side=side)} "
            f"(id {self.arm.body_ids[0]}, jacobian {self.jacobi_idx}) "
            f"fingers={len(self.fingers.joint_ids)}"
        )

    def step(self, robot: Articulation, target: torch.Tensor) -> torch.Tensor:
        """target: (N, 7) grasp-frame pose in the robot base frame."""
        # Walk the grasp target back to the body IK actually drives.
        cmd_pos, cmd_quat = combine_frame_transforms(
            target[:, 0:3], target[:, 3:7], self.tool_offset_pos, self.tool_offset_quat
        )
        self.ctrl.set_command(torch.cat([cmd_pos, cmd_quat], dim=-1))

        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pose_w = robot.data.body_state_w[:, self.arm.body_ids[0], 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        jacobian = robot.root_physx_view.get_jacobians()[
            :, self.jacobi_idx, :, self.arm.joint_ids
        ]
        joint_pos = robot.data.joint_pos[:, self.arm.joint_ids]
        return self.ctrl.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    device = sim.device
    arms = {side: ArmIK(side, scene, device) for side in SIDES}

    receiver = TargetReceiver(args_cli.udp_host, args_cli.udp_port)
    sim_dt = sim.get_physics_dt()
    last_seq = -1
    n_steps = 0
    last_report = time.monotonic()
    warned_frame = False

    try:
        while simulation_app.is_running():
            msg, seq = receiver.latest()

            if msg is not None and seq != last_seq:
                last_seq = seq
                if msg.get("frame") != "base_link" and not warned_frame:
                    print(
                        f"[teleop] WARNING: targets tagged frame "
                        f"{msg.get('frame')!r}, expected 'base_link'; no frame "
                        "conversion is applied here"
                    )
                    warned_frame = True

                for side, arm in arms.items():
                    pose = msg.get(side)
                    if pose is None or len(pose) != 7:
                        continue
                    target = torch.tensor(
                        pose, dtype=torch.float32, device=device
                    ).repeat(scene.num_envs, 1)
                    q_des = arm.step(robot, target)
                    if not args_cli.dry_run:
                        robot.set_joint_position_target(
                            q_des, joint_ids=arm.arm.joint_ids
                        )

                    grip = float(msg.get(f"grip_{side}", 0.0))
                    if not args_cli.dry_run:
                        # All four finger joints take the same positive travel;
                        # finger_joint2 mimics finger_joint1 in the URDF.
                        finger_target = torch.full(
                            (scene.num_envs, len(arm.fingers.joint_ids)),
                            grip,
                            device=device,
                        )
                        robot.set_joint_position_target(
                            finger_target, joint_ids=arm.fingers.joint_ids
                        )

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            n_steps += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                state = "no targets yet" if msg is None else f"seq={seq}"
                print(f"[teleop] {n_steps / (now - last_report):.0f} steps/s, {state}")
                n_steps = 0
                last_report = now
    finally:
        receiver.close()


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    )
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.8])
    scene = InteractiveScene(TeleopSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    print("[teleop] scene ready")
    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
