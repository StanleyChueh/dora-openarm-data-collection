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

"""
dora-openarm-mujoco — MuJoCo simulation node for OpenArm bimanual
========================================================================

This dora node simulates the OpenArm bimanual in MuJoCo.  It replaces
the physical follower arms and cameras in a dataflow, accepting joint-position
commands and publishing arm observations and JPEG camera frames.

Dataflow configuration
----------------------
Minimal (headless, no cameras)::

    - id: openarm-mujoco
      build: pip install -e .
      path: dora-openarm-mujoco
      args: "--viewer ..."
      inputs:
        position_right: leader/follower_position_right
        position_left:  leader/follower_position_left
      outputs:
        - status
        - arm_right_observation
        - arm_left_observation

Inputs
------
position_right / position_left : float32[8] or struct{new_position: float32[8], ...}
    Target joint positions for each arm: joints 1–7 followed by the gripper
    finger joint.  Accepts either a plain float32 array or a StructArray
    with a ``new_position`` field.

pose_right / pose_left : float32[7]
    VR controller pose as [x, y, z, qw, qx, qy, qz].  Only used for the
    ``--debug-frames`` overlay; ignored otherwise.

button_x / button_y : bool[1]
    X / Y button state from the VR controller.  On press every scene joint
    (anything other than the arms) is snapped back to the ``--keyframe``
    pose (default: ``home``): freejoint objects as well as articulated
    fixtures such as drawers and doors.  With ``--reset-arms-on-button`` both
    arms are snapped back too (qpos, qvel and, in ``--ctrl`` mode, the actuator
    targets).
    The trigger is edge-detected: the button must be released before the
    next reset can fire.

Outputs
-------
status : string["ready"]
    Published once on startup so downstream nodes know the sim is live.

arm_right_observation / arm_left_observation : float32[8]
    Observed joint positions (same layout as the inputs) published in response
    to each incoming position command.

camera_wrist_right / camera_wrist_left / camera_head_left / camera_head_right / camera_ceiling : uint8[N]
    JPEG-encoded frames at ~30 Hz.  Only published when ``--render`` is set.
    Each output carries ``metadata={"encoding": "jpeg"}``.

CLI arguments (set via ``args:`` in the dataflow YAML)
--------------------------------------------------------
--xml PATH
    MJCF scene file.  Defaults to the bundled openarm_cell scene.

--scene NAME
    Bundled scene to load when --xml is not set.
    Choices: {cell, demo, pedestal, v1_camera}.

--keyframe NAME  (default: "home")
    Name of the keyframe in the MJCF to reset to on startup.

--enable-collision
    Enable contact/collision detection.  Disabled by default for speed and
    to avoid unexpected joint-locking during teleoperation.

--ctrl
    Write incoming positions to ``data.ctrl`` and advance the physics
    simulation (``mj_step``).  The default is to write directly to
    ``data.qpos`` (``mj_forward`` only), which is faster and kinematically
    exact but ignores actuator dynamics.

--viewer [FPS]
    Open the interactive MuJoCo viewer window (default: off).  When FPS is
    omitted, the target simulation loop frame rate defaults to 30 Hz.  This
    also sets viewer sync, camera publish checks, and control stepping cadence;
    in --ctrl mode the effective cadence is snapped to the MuJoCo timestep.

--render
    Enable offscreen camera rendering and publish JPEG frames.  Adds latency;
    leave off if cameras are not needed.

--debug-frames
    Draw the VR controller coordinate frames as coloured arrows in the viewer.
    Only visible when ``--viewer`` is also set.

--reset-arms-on-button
    Also snap both arms back to ``--keyframe`` when button_x/button_y is
    pressed, instead of only the scene objects.  Off by default because it is
    only safe when whatever drives ``position_left``/``position_right`` resets
    on the same button edge -- otherwise the next incoming command drags the
    arms straight back.  The ik node does exactly that when its own button_x /
    button_y inputs are wired (see dora_openarm_ik.ik), which is the setup
    dataflow-vr-mujoco-ros2.yaml uses.
"""

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback

import cv2
import dora
import mujoco
import mujoco.viewer
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
import pyarrow as pa
from openarm_mujoco.v2 import JointResolver

from dora_openarm_mujoco._draw import draw_frame, draw_world_frame


def _v1_camera_scene_xml() -> str:
    """Bundled scene for the custom v1 OpenArm with wrist-mounted cameras
    (converted from IsaacLab's v1_camera.urdf; see scenes/v1_camera/)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scenes", "v1_camera", "scene.xml"
    )


_SCENE_RESOLVERS = {
    "cell": openarm_mujoco.openarm_cell_xml,
    "demo": openarm_mujoco.openarm_demo_xml,
    "pedestal": openarm_mujoco.openarm_pedestal_xml,
    "v1_camera": _v1_camera_scene_xml,
}
_DEFAULT_SCENE = "cell"
_DEFAULT_VIEWER_FPS = 30.0

# Arm control rate (matches quittable-tick-leader: 2ms = 500Hz)
_ARM_HZ = 500
_ARM_DT = 1.0 / _ARM_HZ

# Camera rendering rate (matches quittable-tick-camera: 33ms ≈ 30Hz)
_CAM_HZ = 30
_CAM_DT = 1.0 / _CAM_HZ
_JPEG_QUALITY = 90

_CAMERAS = [
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
]

# Maps dora input IDs to arm sides for position events.
_ARM_INPUT_SIDES = {"position_right": "right", "position_left": "left"}


# ── helpers ────────────────────────────────────────────────────────────────────


def _lock(viewer, fallback: threading.Lock):
    """Return viewer.lock() when the viewer is active, otherwise the fallback lock."""
    if viewer is not None:
        return viewer.lock()
    return fallback


# ── observation extraction ─────────────────────────────────────────────────────


def _get_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    """Extract current joint positions (7 arm + 1 gripper = 8 elements)."""
    q = np.zeros(8)
    for i in range(1, 8):
        jnt_name = f"openarm_{side}_joint{i}"
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            q[i - 1] = data.qpos[model.jnt_qposadr[jnt_id]]

    grp_name = f"openarm_{side}_finger_joint1"
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, grp_name)
    if jnt_id >= 0:
        q[7] = data.qpos[model.jnt_qposadr[jnt_id]]

    return q.astype(np.float32)


# ── Isaac Sim comparison bridge ──────────────────────────────────────────────
#
# Best-effort UDP JSON broadcaster of MuJoCo's ACTUAL current state (joint
# positions, EEF pose) -- not the VR target both sims are chasing. Feeds the
# same packet shape record_demos_openarm.py's JointFeedbackReceiver /
# save_sim_vs_real_plot already expect on the Isaac Lab side (packet["t"],
# packet["joints"]), extended with "eef_left"/"eef_right" so a single Ctrl+C
# on the Isaac Sim process can plot "does Isaac actually converge to what
# MuJoCo converged to" -- not just joint angles, since IK-driven tasks can
# reach a different joint configuration for a visually-identical EEF pose.


def _relative_site_pose(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int, origin_id: int
) -> list[float] | None:
    """Return [x,y,z,qw,qx,qy,qz] of `site_id`, expressed relative to `origin_id`
    (typically the 'arm_origin' site). Mirrors openarm_control.Kinematics.read_ee_pose's
    convention so this is directly comparable to dora-openarm-kinematics' FK output.
    """
    if site_id < 0:
        return None

    site_pos = data.site_xpos[site_id].copy()
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id].reshape(9).copy())

    if origin_id < 0:
        rel_pos, rel_quat = site_pos, site_quat
    else:
        origin_pos = data.site_xpos[origin_id].copy()
        origin_quat = np.zeros(4)
        mujoco.mju_mat2Quat(origin_quat, data.site_xmat[origin_id].reshape(9).copy())
        origin_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(origin_quat_inv, origin_quat)

        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, site_pos - origin_pos, origin_quat_inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, origin_quat_inv, site_quat)

    return [float(v) for v in (*rel_pos, *rel_quat)]


class IsaacCompareBridge:
    """Fire-and-forget UDP JSON broadcaster of MuJoCo's actual arm state, for
    comparison against Isaac Sim's own state via record_demos_openarm.py's
    --mirror_feedback_port. Never blocks and never raises into the sim loop.
    """

    def __init__(self, model: mujoco.MjModel, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._origin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")
        self._ee_site_ids = {
            side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
            for side in ("right", "left")
        }
        # Running "last known" snapshot for both sides -- _handle_arm only refreshes
        # whichever side just received a new command, so each broadcast merges the
        # freshest side with the other side's last-known values (matches bridge.py's
        # VrUdpBroadcaster null-until-seen convention).
        #
        # No separate gripper-width ("mm") field: this model's openarm_{side}_finger_joint1
        # is a revolute joint in radians (~0.78 rad range, matching the VR trigger-mapping
        # constant in dora-openarm-kinematics), NOT the 0..0.044m prismatic joint Isaac Sim's
        # OpenArm gripper uses -- there's no valid shared "mm" conversion between the two.
        # finger_joint1's raw qpos is already sent below as part of `joints`, in this side's
        # own native units, which is the only honest comparison across the two joint types.
        self._joints: dict[str, float] = {}
        self._eef: dict[str, list[float] | None] = {"right": None, "left": None}
        print(f"[isaac-compare] Broadcasting actual MuJoCo state to {host}:{port}")

    def update(self, side: str, model: mujoco.MjModel, data: mujoco.MjData, obs: np.ndarray) -> None:
        for i in range(7):
            self._joints[f"openarm_{side}_joint{i + 1}"] = float(obs[i])
        self._joints[f"openarm_{side}_finger_joint1"] = float(obs[7])
        self._eef[side] = _relative_site_pose(model, data, self._ee_site_ids[side], self._origin_id)

        packet = {
            "t": time.time(),
            "joints": dict(self._joints),
            "eef_right": self._eef["right"],
            "eef_left": self._eef["left"],
        }
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # best-effort only -- never let a networking hiccup break the sim


# ── scene-object reset ─────────────────────────────────────────────────────────


# Per joint type: (qpos width, qvel width).
_JOINT_WIDTHS = {
    mujoco.mjtJoint.mjJNT_FREE: (7, 6),
    mujoco.mjtJoint.mjJNT_BALL: (4, 3),
    mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
    mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
}


def _find_scene_joint_addrs(model: mujoco.MjModel) -> list[tuple[slice, slice, str]]:
    """Find joints belonging to non-arm bodies.

    Returns a list of ``(qpos_slice, qvel_slice, name)`` covering freejoint
    objects and articulated fixtures (drawers, doors, ...).  Bodies whose name
    starts with ``openarm_`` are skipped so the arms are never teleported.
    """
    addrs: list[tuple[slice, slice, str]] = []
    for jnt_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[jnt_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("openarm_"):
            continue
        nq, nv = _JOINT_WIDTHS[mujoco.mjtJoint(model.jnt_type[jnt_id])]
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        qvel_adr = int(model.jnt_dofadr[jnt_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or body_name
        addrs.append(
            (slice(qpos_adr, qpos_adr + nq), slice(qvel_adr, qvel_adr + nv), name)
        )
    return addrs


def _find_arm_dof_addrs(model: mujoco.MjModel) -> list[int]:
    """DOF addresses of both arms' joints (7 arm + 2 finger joints per side)."""
    names = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    addrs: list[int] = []
    for side in ("right", "left"):
        for suffix in names:
            jnt_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_{suffix}"
            )
            if jnt_id >= 0:
                addrs.append(int(model.jnt_dofadr[jnt_id]))
    return addrs


def _reset_arms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    key_id: int,
    arm_dof_addrs: list[int],
) -> bool:
    """Snap both arms back to the keyframe posture, at rest.

    Writes ``data.ctrl`` as well as ``data.qpos``: in ``--ctrl`` mode the
    actuators would otherwise immediately pull the arms back to the pre-reset
    command.  The ik node performs the matching reset of its own solver
    configuration on the same button edge, so the next command it publishes is
    the keyframe posture too.
    """
    if key_id < 0:
        return False
    key_qpos = model.key_qpos[key_id]
    for side in ("right", "left"):
        joints, finger = mapper.get_driver(key_qpos, side)
        driver = np.append(np.asarray(joints, dtype=np.float64), float(finger))
        mapper.set_qpos(data.qpos, driver, side)
        mapper.set_ctrl(data.ctrl, driver, side)
    for dof in arm_dof_addrs:
        data.qvel[dof] = 0.0
        data.qacc[dof] = 0.0
    mujoco.mj_forward(model, data)
    return True


def _reset_scene_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    key_id: int,
    addrs: list[tuple[slice, slice, str]],
) -> None:
    """Snap each scene joint (arms excluded) back to its keyframe pose."""
    if key_id < 0 or not addrs:
        return
    for qpos_sl, qvel_sl, _ in addrs:
        data.qpos[qpos_sl] = model.key_qpos[key_id, qpos_sl]
        data.qvel[qvel_sl] = 0.0
    mujoco.mj_forward(model, data)


# ── offscreen camera rendering ─────────────────────────────────────────────────


class CameraRenderer:
    """Offscreen renderer for MuJoCo cameras. Renders to JPEG bytes."""

    def __init__(self, model: mujoco.MjModel, jpeg_quality: int = 90):
        self.jpeg_quality = jpeg_quality
        self.cam_ids: dict[str, int] = {}
        self.renderers: dict[str, mujoco.Renderer] = {}

        cam_resolutions: dict[str, tuple[int, int]] = {}
        for cam_name in _CAMERAS:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                print(f"[camera] Warning: camera '{cam_name}' not found in model")
                continue
            self.cam_ids[cam_name] = cam_id
            res = model.cam_resolution[cam_id]  # [width, height]
            cam_resolutions[cam_name] = (int(res[0]), int(res[1]))

        if cam_resolutions:
            max_w = max(w for w, _ in cam_resolutions.values())
            max_h = max(h for _, h in cam_resolutions.values())
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, max_w)
            model.vis.global_.offheight = max(model.vis.global_.offheight, max_h)

        for cam_name, (w, h) in cam_resolutions.items():
            try:
                self.renderers[cam_name] = mujoco.Renderer(model, height=h, width=w)
                print(f"[camera] '{cam_name}' renderer: {w}x{h}")
            except Exception as e:
                print(
                    f"[camera] ERROR: could not initialize renderer for '{cam_name}': {e}"
                )

    def render_all(self, data: mujoco.MjData) -> dict[str, bytes]:
        images = {}
        for cam_name, renderer in self.renderers.items():
            renderer.update_scene(data, camera=self.cam_ids[cam_name])
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if ok:
                images[cam_name] = buf.tobytes()
        return images

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


class CameraScheduler:
    """Throttles camera renders to _CAM_HZ and publishes JPEG frames via dora."""

    def __init__(self, renderer: CameraRenderer, node: dora.Node, data: mujoco.MjData):
        self._renderer = renderer
        self._node = node
        self._data = data
        self._next = time.perf_counter() + _CAM_DT

    def tick(self, lock_fn) -> None:
        if time.perf_counter() < self._next:
            return
        with lock_fn():
            images = self._renderer.render_all(self._data)
        for cam_name, jpeg_bytes in images.items():
            self._node.send_output(
                cam_name,
                pa.array(np.frombuffer(jpeg_bytes, dtype=np.uint8), type=pa.uint8()),
                metadata={"encoding": "jpeg"},
            )
        self._next += _CAM_DT

    def close(self) -> None:
        self._renderer.close()


# ── arm event handler ──────────────────────────────────────────────────────────


def _handle_arm(
    side: str,
    values: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    node: dora.Node,
    viewer,
    data_lock: threading.Lock,
    use_ctrl: bool,
    compare_bridge: "IsaacCompareBridge | None" = None,
) -> None:
    with _lock(viewer, data_lock):
        if use_ctrl:
            mapper.set_ctrl(data.ctrl, values, side)
        else:
            mapper.set_qpos(data.qpos, values, side)
            mujoco.mj_forward(model, data)
    obs = _get_arm_qpos(model, data, side)
    node.send_output(f"arm_{side}_observation", pa.array(obs, type=pa.float32()))
    if compare_bridge is not None:
        compare_bridge.update(side, model, data, obs)


# ── dora event loop (background thread) ───────────────────────────────────────


def _run_dora(
    node: dora.Node,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapper: JointResolver,
    viewer,
    data_lock: threading.Lock,
    stop_event: threading.Event,
    reset_key_id: int,
    object_addrs: list[tuple[slice, slice, str]],
    use_ctrl: bool = False,
    debug_frames: bool = False,
    reset_arms: bool = False,
    compare_bridge: "IsaacCompareBridge | None" = None,
) -> None:
    print("[dora] Event loop started.")
    pose_right: np.ndarray | None = None
    pose_left: np.ndarray | None = None
    button_prev = {"button_x": False, "button_y": False}
    arm_dof_addrs = _find_arm_dof_addrs(model)
    object_names = ", ".join(name for _, _, name in object_addrs) or "(none)"

    def _reset(source: str) -> None:
        with _lock(viewer, data_lock):
            _reset_scene_objects(model, data, reset_key_id, object_addrs)
            done = (
                _reset_arms(model, data, mapper, reset_key_id, arm_dof_addrs)
                if reset_arms
                else False
            )
        parts = [f"objects: {object_names}"]
        if reset_arms:
            parts.append("arms → keyframe" if done else "arms: no keyframe, skipped")
        print(f"[reset] {source} pressed → {'; '.join(parts)}", flush=True)

    try:
        for event in node:
            if stop_event.is_set():
                break
            if event["type"] != "INPUT":
                continue

            eid = event["id"]

            if eid in _ARM_INPUT_SIDES:
                value = event["value"]
                if isinstance(value, pa.StructArray):
                    value = value.field("new_position")
                    # TODO: We use this for safety check later.
                    # other_arm_position = value.field("other_arm_position")
                values = np.array(value, dtype=np.float32)
                if values.shape == (8,):
                    _handle_arm(
                        _ARM_INPUT_SIDES[eid],
                        values,
                        model,
                        data,
                        mapper,
                        node,
                        viewer,
                        data_lock,
                        use_ctrl,
                        compare_bridge,
                    )
            elif eid == "pose_right":
                pose_right = np.array(event["value"], dtype=np.float32)
            elif eid == "pose_left":
                pose_left = np.array(event["value"], dtype=np.float32)
            elif eid in button_prev:
                pressed = bool(np.asarray(event["value"]).reshape(-1)[0])
                rising = pressed and not button_prev[eid]
                button_prev[eid] = pressed
                if rising:
                    _reset(eid)

            if viewer is not None and debug_frames:
                with viewer.lock():
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    draw_world_frame(scn)
                    draw_frame(scn, pose_right)
                    draw_frame(scn, pose_left)

    finally:
        print("[dora] Event loop ended – signalling shutdown.")
        stop_event.set()


# ── physics + render loop ──────────────────────────────────────────────────────


def _run_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    lock_fn,  # callable: () → context manager
    stop_event: threading.Event,
    steps_per_frame: int,
    loop_dt: float,
    use_ctrl: bool,
    viewer=None,
    cam_scheduler: "CameraScheduler | None" = None,
) -> None:
    """Single physics/sync/camera loop used for both viewer and headless modes."""
    while not stop_event.is_set():
        if viewer is not None and not viewer.is_running():
            break
        t0 = time.perf_counter()

        if use_ctrl:
            with lock_fn():
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if cam_scheduler is not None:
            cam_scheduler.tick(lock_fn)

        elapsed = time.perf_counter() - t0
        if elapsed < loop_dt:
            time.sleep(loop_dt - elapsed)


# ── model setup ────────────────────────────────────────────────────────────────


def _setup_model(
    args,
) -> tuple[
    mujoco.MjModel, mujoco.MjData, JointResolver, int, list[tuple[slice, slice, str]]
]:
    xml_path = args.xml if args.xml is not None else _SCENE_RESOLVERS[args.scene]()
    print(f"[model] Loading scene: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mapper = JointResolver(model)

    if not args.enable_collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        print("[model] Collision (Contact) detection is DISABLED by default.")
    else:
        print("[model] Collision (Contact) detection is ENABLED.")

    cell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cell_vis")
    if cell_id >= 0:
        model.geom_rgba[cell_id, 3] = 0.2

    key_id = -1
    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            print(
                f"[model] Warning: keyframe '{args.keyframe}' not found, using defaults."
            )

    mujoco.mj_forward(model, data)

    if args.ctrl:
        mapper.set_ctrl(data.ctrl, _get_arm_qpos(model, data, "right"), "right")
        mapper.set_ctrl(data.ctrl, _get_arm_qpos(model, data, "left"), "left")

    object_addrs = _find_scene_joint_addrs(model)
    if object_addrs:
        names = ", ".join(name for _, _, name in object_addrs)
        print(f"[model] Resettable scene joints: {names}")
    else:
        print("[model] No non-arm scene joints found – button_x reset will be a no-op.")

    return model, data, mapper, key_id, object_addrs


# ── argument parsing ───────────────────────────────────────────────────────────


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Viewer dora node – MuJoCo renderer with camera output for OpenArm"
    )
    p.add_argument(
        "--xml", default=None, help="MJCF scene file. Overrides --scene when set."
    )
    p.add_argument(
        "--scene",
        choices=sorted(_SCENE_RESOLVERS),
        default=_DEFAULT_SCENE,
        help=f"Bundled scene to load when --xml is not set (default: {_DEFAULT_SCENE})",
    )
    p.add_argument(
        "--keyframe", "-k", default="home", help="Initial keyframe name (default: home)"
    )
    p.add_argument(
        "--enable-collision",
        action="store_true",
        help="Enable collision detection (default: disabled)",
    )
    p.add_argument(
        "--ctrl",
        action="store_true",
        help="Write data.ctrl targets and step physics instead of writing data.qpos directly",
    )
    p.add_argument(
        "--viewer",
        nargs="?",
        const=_DEFAULT_VIEWER_FPS,
        default=None,
        type=_positive_float,
        metavar="FPS",
        help=(
            "Open the interactive MuJoCo viewer window (default: off). "
            "Optionally set the target loop frame rate in Hz, which controls "
            "viewer sync, camera publish checks, and control stepping cadence; "
            "--ctrl mode snaps the effective cadence to the MuJoCo timestep "
            f"(default when omitted: {_DEFAULT_VIEWER_FPS:g})"
        ),
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Enable offscreen camera rendering and publish images (default: off)",
    )
    p.add_argument(
        "--debug-frames",
        action="store_true",
        help="Draw VR controller coordinate frames as overlays in the viewer (default: off)",
    )
    p.add_argument(
        "--reset-arms-on-button",
        action="store_true",
        help=(
            "On button_x/button_y also snap both arms back to --keyframe, not just the"
            " scene objects. Only safe when the node driving position_left/right resets"
            " on the same button edge (the ik node does, once its button inputs are"
            " wired) -- otherwise the next command drags the arms straight back."
        ),
    )
    p.add_argument(
        "--isaac-feedback-host",
        default="127.0.0.1",
        help=(
            "Destination host for --isaac-feedback-port (default: 127.0.0.1). Should match the"
            " --mirror_udp_host record_demos_openarm.py is listening on."
        ),
    )
    p.add_argument(
        "--isaac-feedback-port",
        type=int,
        default=0,
        help=(
            "If nonzero, best-effort UDP-broadcast MuJoCo's actual joint/EEF/gripper state (not the"
            " VR target) to <isaac-feedback-host>:<isaac-feedback-port> after every arm update. Off by"
            " default. Point this at record_demos_openarm.py's --mirror_feedback_port so Ctrl+C there"
            " saves a MuJoCo-vs-Isaac comparison plot."
        ),
    )
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    target_fps = args.viewer if args.viewer is not None else _DEFAULT_VIEWER_FPS

    stop_event = threading.Event()

    def _on_signal(sig, _frame):
        print(f"[main] Received signal {sig}, shutting down.")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    model, data, mapper, reset_key_id, object_addrs = _setup_model(args)

    compare_bridge: IsaacCompareBridge | None = None
    if args.isaac_feedback_port:
        compare_bridge = IsaacCompareBridge(model, args.isaac_feedback_host, args.isaac_feedback_port)
    frame_dt = 1.0 / target_fps
    steps_per_frame = max(1, math.ceil(frame_dt / model.opt.timestep))
    loop_dt = steps_per_frame * model.opt.timestep if args.ctrl else frame_dt
    effective_fps = 1.0 / loop_dt
    print(
        f"[loop] target_fps={target_fps:g}, effective_fps={effective_fps:g}, "
        f"model_timestep={model.opt.timestep:g}, "
        f"steps_per_frame={steps_per_frame}, loop_dt={loop_dt:g}"
    )

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))

    # Bootstrap initial arm observations so the observer can begin ticking.
    for side in ("right", "left"):
        q = _get_arm_qpos(model, data, side)
        node.send_output(f"arm_{side}_observation", pa.array(q, type=pa.float32()))

    cam_scheduler: CameraScheduler | None = None
    if args.render:
        renderer = CameraRenderer(model, _JPEG_QUALITY)
        print(f"[camera] Available cameras: {list(renderer.cam_ids.keys())}")
        cam_scheduler = CameraScheduler(renderer, node, data)

    data_lock = threading.Lock()

    if args.viewer is not None:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.5
            viewer.cam.lookat[:] = [1.3, 0, 0.6]

            dora_thread = threading.Thread(
                target=_run_dora,
                args=(
                    node,
                    model,
                    data,
                    mapper,
                    viewer,
                    data_lock,
                    stop_event,
                    reset_key_id,
                    object_addrs,
                    args.ctrl,
                    args.debug_frames,
                    args.reset_arms_on_button,
                    compare_bridge,
                ),
                daemon=True,
            )
            dora_thread.start()
            _run_loop(
                model,
                data,
                viewer.lock,
                stop_event,
                steps_per_frame,
                loop_dt,
                args.ctrl,
                viewer=viewer,
                cam_scheduler=cam_scheduler,
            )

    else:
        print("[main] Running headless (no viewer window).")
        dora_thread = threading.Thread(
            target=_run_dora,
            args=(
                node,
                model,
                data,
                mapper,
                None,
                data_lock,
                stop_event,
                reset_key_id,
                object_addrs,
                args.ctrl,
                args.debug_frames,
                args.reset_arms_on_button,
                compare_bridge,
            ),
            daemon=True,
        )
        dora_thread.start()
        _run_loop(
            model,
            data,
            lambda: data_lock,
            stop_event,
            steps_per_frame,
            loop_dt,
            args.ctrl,
            cam_scheduler=cam_scheduler,
        )

    if cam_scheduler is not None:
        cam_scheduler.close()
    stop_event.set()
    dora_thread.join(timeout=2.0)
    print("[main] Shutdown complete.")


def cli_main() -> None:
    """Console entrypoint.

    MuJoCo/GLFW can segfault during Python interpreter teardown after the viewer
    has already closed cleanly. Exit the process after a successful shutdown so
    Dora observes the real result instead of a native finalizer crash.
    """
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 1
            try:
                print(exc.code, file=sys.stderr)
            except Exception:
                pass
    except KeyboardInterrupt:
        exit_code = 130
        try:
            traceback.print_exc()
        except Exception:
            pass
    except BaseException:
        exit_code = 1
        try:
            traceback.print_exc()
        except Exception:
            pass
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
    os._exit(exit_code)


if __name__ == "__main__":
    cli_main()
