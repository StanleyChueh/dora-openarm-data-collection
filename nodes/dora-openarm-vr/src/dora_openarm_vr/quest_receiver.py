"""
Meta Quest UDP pose receiver — specification
==============================================

[1. Incoming JSON Structure]
- t:  headset monotonic timestamp (seconds, Time.realtimeSinceStartup)
- lc / rc / rf:  pose objects (left controller / right controller / reference)
    - x, y, z: Unity left-handed world coordinates (meters)
    - qx, qy, qz, qw: Unity left-handed rotation (Quaternion)
- lt / rt: left/right index trigger  0.0–1.0
- lg / rg: left/right grip           0.0–1.0
- lsx / lsy / rsx / rsy: thumbstick axes  -1.0–1.0
- a / b / x / y: buttons
- v:  overall validity   0=OK, 1=STALE, 2=INVALID
- vl: left controller validity
- vr: right controller validity

[2. Validity Handling]
- OK (0):     normal processing
- STALE (1):  HMD is sending last-good pose; pass through smoother normally
- INVALID(2): do not output pose; reset smoother so re-entry is jump-free
- buttons/triggers/grips are always forwarded regardless of pose validity

[3. Coordinate Transformation (LH to RH)]
1. Position Flip:
    p_mujoco = [x, y, -z]
2. Quaternion Flip:
    q_mujoco = [qw, -qx, -qy, qz]
3. Reference Rectification
   A saved reference pose (p_ref, R_ref) is subtracted so that the
   controller pose is expressed relative to where the operator was
   standing/looking when the reference was captured.  Two modes differ
   in which frame the relative pose is expressed in:

   NECK mode  — relative position is rotated into the HMD's frame:
     p_rel = R_ref_inv * (p_ctrl - p_ref)   (displacement in HMD axes)
     r_rel = R_ref_inv * r_ctrl             (orientation relative to HMD)

[4. Robot Workspace Mapping]
- p_out = R_FRAME * p_rel + FRAME_OFFSET_NECK
- r_out = R_FRAME * r_rel * R_FIX
    * R_FIX = Rot_z(90)
"""

import argparse
import collections
import time
from dataclasses import dataclass

import dora
import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

from .smoothing import OneEuroPoseSmoother
from .udp_receiver import JsonUdpReceiver

# ── Frame alignment — edit here to tune ──────────────────────────────────────
_FRAME_ROT: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

FRAME_OFFSET_NECK: np.ndarray = np.array([-0.2, 0, -0.3], dtype=np.float64)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 5006

VALID_OK = 0
VALID_STALE = 1
VALID_INVALID = 2
_VALID_NAMES = {VALID_OK: "OK", VALID_STALE: "STALE", VALID_INVALID: "INVALID"}

_R_FRAME = Rotation.from_matrix(_FRAME_ROT)
_IDENTITY_REF = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0,
}


def parse_lh_to_rh(c: dict) -> tuple[np.ndarray, Rotation]:
    """Convert a Unity left-handed pose dict to a right-handed (position, Rotation) pair.

    Input keys: x, y, z (meters), qx, qy, qz, qw (Unity quaternion, scalar-last).
    Flip: z → -z, qx → -qx, qy → -qy.
    """
    pos = np.array([c["x"], c["y"], -c["z"]], dtype=np.float64)
    rot = Rotation.from_quat([-c["qx"], -c["qy"], c["qz"], c["qw"]])
    return pos, rot


def pose_to_array(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    q = rot.as_quat()
    return np.array([pos[0], pos[1], pos[2], q[3], q[0], q[1], q[2]], dtype=np.float32)


@dataclass(slots=True)
class ProcessedPoses:
    """All pose representations derived from one Quest UDP packet."""

    mapped_right: np.ndarray | None
    mapped_left: np.ndarray | None
    pose_reference: np.ndarray | None
    raw_right: np.ndarray | None
    raw_left: np.ndarray | None
    raw_reference: np.ndarray | None
    relative_right: np.ndarray | None
    relative_left: np.ndarray | None
    relative_reference: np.ndarray | None


class QuestPoseProcessor:
    def process(self, msg: dict) -> ProcessedPoses:
        """Build raw, reference-relative, and robot-mapped poses.

        Naming used here:
        - raw_*: Unity world poses converted only from LH to RH coordinates.
        - relative_*: controller poses expressed in the rf/reference frame.
        - mapped_*: relative controller poses mapped into the robot arm_origin frame.
        """
        reference_packet = msg.get("rf")
        right_packet = msg.get("rc")
        left_packet = msg.get("lc")

        # Keep the original receiver behavior when rf is absent: use identity for
        # robot mapping.  The headset-relative viewer, however, only displays a
        # relative pose when an actual rf packet is present.
        p_ref, r_ref = parse_lh_to_rh(reference_packet or _IDENTITY_REF)
        r_ref_inv = r_ref.inv()
        r_fix = Rotation.from_euler("z", 90, degrees=True)

        raw_reference = (
            pose_to_array(p_ref, r_ref) if reference_packet is not None else None
        )
        relative_reference = (
            pose_to_array(np.zeros(3, dtype=np.float64), Rotation.identity())
            if reference_packet is not None
            else None
        )

        def _convert_controller(
            packet: dict | None,
        ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            if packet is None:
                return None, None, None

            p_world, r_world = parse_lh_to_rh(packet)
            raw_pose = pose_to_array(p_world, r_world)

            p_rel = r_ref_inv.apply(p_world - p_ref)
            r_rel = r_ref_inv * r_world
            relative_pose = (
                pose_to_array(p_rel, r_rel)
                if reference_packet is not None
                else None
            )

            p_out = _R_FRAME.apply(p_rel) + FRAME_OFFSET_NECK
            r_out = _R_FRAME * r_rel * r_fix
            mapped_pose = pose_to_array(p_out, r_out)
            return mapped_pose, raw_pose, relative_pose

        mapped_right, raw_right, relative_right = _convert_controller(right_packet)
        mapped_left, raw_left, relative_left = _convert_controller(left_packet)

        return ProcessedPoses(
            mapped_right=mapped_right,
            mapped_left=mapped_left,
            pose_reference=raw_reference,
            raw_right=raw_right,
            raw_left=raw_left,
            raw_reference=raw_reference,
            relative_right=relative_right,
            relative_left=relative_left,
            relative_reference=relative_reference,
        )


def _pose_rotation(pose: np.ndarray) -> Rotation:
    """Convert [x, y, z, qw, qx, qy, qz] into a SciPy Rotation."""
    return Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])


def _mapped_reference_pose() -> np.ndarray:
    """Reference anchor expressed in the same mapped frame as controller outputs."""
    r_fix = Rotation.from_euler("z", 90, degrees=True)
    mapped_rotation = _R_FRAME * r_fix
    return pose_to_array(FRAME_OFFSET_NECK.copy(), mapped_rotation)


def _compact_pose_line(label: str, pose: np.ndarray | None) -> str:
    if pose is None:
        return f"{label:<13} None"
    return (
        f"{label:<13} p[{pose[0]: .3f}, {pose[1]: .3f}, {pose[2]: .3f}] "
        f"q[{pose[3]: .3f}, {pose[4]: .3f}, {pose[5]: .3f}, {pose[6]: .3f}]"
    )


class ThreePoseViewer:
    """Reusable non-blocking 3D viewer for right, left, and reference poses."""

    _POSE_COLORS = {
        "right": "tab:red",
        "left": "tab:blue",
        "reference": "tab:green",
    }
    _AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")

    def __init__(
        self,
        *,
        window_title: str,
        figure_title: str,
        frame_description: str,
        default_center: np.ndarray,
        min_span: float = 1.5,
        axis_length: float = 0.12,
        max_fps: float = 30.0,
        trail_length: int = 60,
        labels: dict[str, str] | None = None,
    ) -> None:
        if min_span <= 0.0:
            raise ValueError("min_span must be positive")
        if axis_length <= 0.0:
            raise ValueError("axis_length must be positive")
        if max_fps <= 0.0:
            raise ValueError("max_fps must be positive")
        if trail_length < 0:
            raise ValueError("trail_length must be non-negative")

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on local GUI setup
            raise RuntimeError(
                "Matplotlib could not be imported. Install it with: uv pip install matplotlib"
            ) from exc

        self._plt = plt
        self._min_span = float(min_span)
        self._axis_length = float(axis_length)
        self._update_period = 1.0 / float(max_fps)
        self._last_draw_time = 0.0
        self._closed = False
        self._default_center = np.asarray(default_center, dtype=np.float64)
        self._labels = labels or {
            "right": "Right",
            "left": "Left",
            "reference": "Reference",
        }

        self._history = {
            name: collections.deque(maxlen=trail_length)
            for name in ("right", "left", "reference")
        }

        plt.ion()
        self._fig = plt.figure(figsize=(13.0, 7.5))
        manager = self._fig.canvas.manager
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(window_title)

        grid = self._fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 1.12))
        self._ax = self._fig.add_subplot(grid[0, :3], projection="3d")
        self._info_ax = self._fig.add_subplot(grid[0, 3])
        self._info_ax.axis("off")

        self._fig.suptitle(figure_title, fontsize=14)
        self._ax.set_xlabel("X [m]")
        self._ax.set_ylabel("Y [m]")
        self._ax.set_zlabel("Z [m]")
        self._ax.set_box_aspect((1.0, 1.0, 1.0))
        self._ax.grid(True)
        self._ax.view_init(elev=24.0, azim=-55.0)

        self._point_artists = {}
        self._label_artists = {}
        self._trail_artists = {}
        self._orientation_artists = {}

        for name, marker in (("right", "o"), ("left", "o"), ("reference", "D")):
            color = self._POSE_COLORS[name]
            point, = self._ax.plot(
                [], [], [],
                marker=marker,
                linestyle="None",
                markersize=10,
                color=color,
                label=self._labels[name],
            )
            trail, = self._ax.plot([], [], [], color=color, alpha=0.35, linewidth=1.2)
            label = self._ax.text(0.0, 0.0, 0.0, "", color=color, fontsize=9)
            axes = [
                self._ax.plot([], [], [], color=axis_color, linewidth=2.0)[0]
                for axis_color in self._AXIS_COLORS
            ]
            self._point_artists[name] = point
            self._trail_artists[name] = trail
            self._label_artists[name] = label
            self._orientation_artists[name] = axes

        self._ref_to_right, = self._ax.plot(
            [], [], [], color=self._POSE_COLORS["right"], linestyle="--", alpha=0.7
        )
        self._ref_to_left, = self._ax.plot(
            [], [], [], color=self._POSE_COLORS["left"], linestyle="--", alpha=0.7
        )
        self._left_to_right, = self._ax.plot(
            [], [], [], color="0.45", linestyle=":", alpha=0.7
        )

        origin = np.zeros(3, dtype=np.float64)
        self._origin_axes = []
        for i, color in enumerate(self._AXIS_COLORS):
            end = origin.copy()
            end[i] = self._axis_length
            line, = self._ax.plot(
                [origin[0], end[0]],
                [origin[1], end[1]],
                [origin[2], end[2]],
                color=color,
                linewidth=1.4,
                alpha=0.75,
            )
            self._origin_axes.append(line)

        self._status_text = self._info_ax.text(
            0.0,
            1.0,
            "Waiting for pose packets...",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.6,
            transform=self._info_ax.transAxes,
        )
        self._description_text = self._info_ax.text(
            0.0,
            0.02,
            frame_description,
            va="bottom",
            ha="left",
            fontsize=8.3,
            wrap=True,
            transform=self._info_ax.transAxes,
        )

        self._ax.legend(loc="upper left")
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._set_equal_limits([self._default_center])
        self._fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.show(block=False)
        plt.pause(0.001)

    @property
    def closed(self) -> bool:
        return self._closed

    def _on_close(self, _event) -> None:
        self._closed = True

    @staticmethod
    def _valid_pose(pose: np.ndarray | None) -> bool:
        return (
            pose is not None
            and pose.shape[0] >= 7
            and bool(np.all(np.isfinite(pose[:7])))
        )

    @staticmethod
    def _set_line(line, start: np.ndarray | None, end: np.ndarray | None) -> None:
        if start is None or end is None:
            line.set_data_3d([], [], [])
            return
        line.set_data_3d(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
        )

    def _set_pose_artist(self, name: str, pose: np.ndarray | None) -> None:
        point = self._point_artists[name]
        label = self._label_artists[name]
        trail = self._trail_artists[name]
        orientation_axes = self._orientation_artists[name]

        if not self._valid_pose(pose):
            self._history[name].clear()
            point.set_data_3d([], [], [])
            label.set_text("")
            label.set_position((0.0, 0.0))
            label.set_3d_properties(0.0)
            trail.set_data_3d([], [], [])
            for axis_line in orientation_axes:
                axis_line.set_data_3d([], [], [])
            return

        assert pose is not None
        position = np.asarray(pose[:3], dtype=np.float64)
        point.set_data_3d([position[0]], [position[1]], [position[2]])
        label.set_text(self._labels[name])
        label.set_position((position[0], position[1]))
        label.set_3d_properties(position[2])

        history = self._history[name]
        history.append(position.copy())
        if len(history) >= 2:
            samples = np.asarray(history)
            trail.set_data_3d(samples[:, 0], samples[:, 1], samples[:, 2])
        else:
            trail.set_data_3d([], [], [])

        rotation_matrix = _pose_rotation(pose).as_matrix()
        for axis_index, axis_line in enumerate(orientation_axes):
            endpoint = position + rotation_matrix[:, axis_index] * self._axis_length
            self._set_line(axis_line, position, endpoint)

    def _set_equal_limits(self, positions: list[np.ndarray]) -> None:
        if not positions:
            center = self._default_center
            half_span = self._min_span / 2.0
        else:
            samples = np.asarray(positions, dtype=np.float64)
            lower = samples.min(axis=0)
            upper = samples.max(axis=0)
            center = (lower + upper) / 2.0
            full_span = max(float(np.max(upper - lower)) * 1.5, self._min_span)
            half_span = full_span / 2.0

        self._ax.set_xlim(center[0] - half_span, center[0] + half_span)
        self._ax.set_ylim(center[1] - half_span, center[1] + half_span)
        self._ax.set_zlim(center[2] - half_span, center[2] + half_span)

    def _pose_text(self, name: str, pose: np.ndarray | None, validity: int) -> list[str]:
        validity_name = _VALID_NAMES.get(validity, str(validity))
        label = self._labels[name].upper()
        if pose is None:
            return [f"{label:<14} {validity_name}", "  pose: None"]
        return [
            f"{label:<14} {validity_name}",
            f"  p [{pose[0]: .3f}, {pose[1]: .3f}, {pose[2]: .3f}] m",
            f"  q [{pose[3]: .3f}, {pose[4]: .3f}, {pose[5]: .3f}, {pose[6]: .3f}]",
        ]

    def _build_status_text(
        self,
        right: np.ndarray | None,
        left: np.ndarray | None,
        reference: np.ndarray | None,
        v_right: int,
        v_left: int,
        v_reference: int,
        extra_lines: list[str] | None,
    ) -> str:
        lines = ["DISPLAYED VALUES", ""]
        lines.extend(self._pose_text("right", right, v_right))
        lines.append("")
        lines.extend(self._pose_text("left", left, v_left))
        lines.append("")
        lines.extend(self._pose_text("reference", reference, v_reference))
        lines.extend(["", "DISTANCES"])

        right_pos = right[:3] if self._valid_pose(right) else None
        left_pos = left[:3] if self._valid_pose(left) else None
        ref_pos = reference[:3] if self._valid_pose(reference) else None

        if ref_pos is not None and right_pos is not None:
            lines.append(f"  ref -> right {np.linalg.norm(right_pos - ref_pos):.3f} m")
        else:
            lines.append("  ref -> right ---")

        if ref_pos is not None and left_pos is not None:
            lines.append(f"  ref -> left  {np.linalg.norm(left_pos - ref_pos):.3f} m")
        else:
            lines.append("  ref -> left  ---")

        if left_pos is not None and right_pos is not None:
            lines.append(f"  left <-> right {np.linalg.norm(right_pos - left_pos):.3f} m")
        else:
            lines.append("  left <-> right ---")

        if extra_lines:
            lines.extend(["", *extra_lines])
        return "\n".join(lines)

    def update(
        self,
        *,
        now: float,
        right: np.ndarray | None,
        left: np.ndarray | None,
        reference: np.ndarray | None,
        v_right: int,
        v_left: int,
        v_reference: int,
        extra_lines: list[str] | None = None,
    ) -> None:
        if self._closed or now - self._last_draw_time < self._update_period:
            return
        self._last_draw_time = now

        poses = {"right": right, "left": left, "reference": reference}
        for name, pose in poses.items():
            self._set_pose_artist(name, pose)

        right_pos = right[:3] if self._valid_pose(right) else None
        left_pos = left[:3] if self._valid_pose(left) else None
        ref_pos = reference[:3] if self._valid_pose(reference) else None

        self._set_line(self._ref_to_right, ref_pos, right_pos)
        self._set_line(self._ref_to_left, ref_pos, left_pos)
        self._set_line(self._left_to_right, left_pos, right_pos)

        visible_positions = [
            np.asarray(position, dtype=np.float64)
            for position in (right_pos, left_pos, ref_pos)
            if position is not None
        ]
        self._set_equal_limits(visible_positions)
        self._status_text.set_text(
            self._build_status_text(
                right,
                left,
                reference,
                v_right,
                v_left,
                v_reference,
                extra_lines,
            )
        )

        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(0.001)
        except Exception:
            self._closed = True

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._plt.close(self._fig)


def _run(args: argparse.Namespace) -> None:
    receiver = JsonUdpReceiver(args.host, args.port)
    processor = QuestPoseProcessor()

    robot_viewer: ThreePoseViewer | None = None
    vr_viewer: ThreePoseViewer | None = None
    if args.visualize:
        try:
            robot_viewer = ThreePoseViewer(
                window_title="Robot EEF Target Poses",
                figure_title="Mapped Robot EEF Targets",
                frame_description=(
                    "Plot frame: robot arm_origin. pose_left/right are reference-"
                    "rectified and workspace-mapped targets. The green reference is "
                    "a mapped anchor by default; use --view-reference-mode output "
                    "to plot the literal pose_reference output."
                ),
                default_center=FRAME_OFFSET_NECK,
                min_span=args.view_span,
                axis_length=args.view_axis_length,
                max_fps=args.view_fps,
                trail_length=args.view_trail,
                labels={
                    "right": "pose_right",
                    "left": "pose_left",
                    "reference": "pose_reference",
                },
            )
            vr_viewer = ThreePoseViewer(
                window_title="VR Headset-Relative Raw Poses",
                figure_title="Raw Controllers Relative to Reference / Headset",
                frame_description=(
                    "Plot frame: rf/reference frame. Controller poses are computed "
                    "as R_ref^-1 * (p_controller - p_ref); the green reference is "
                    "the identity pose at the origin. Side-panel RAW RH WORLD values "
                    "are rc/lc/rf after only Unity LH-to-RH conversion."
                ),
                default_center=np.zeros(3, dtype=np.float64),
                min_span=args.view_span,
                axis_length=args.view_axis_length,
                max_fps=args.view_fps,
                trail_length=args.view_trail,
                labels={
                    "right": "right_raw_rel",
                    "left": "left_raw_rel",
                    "reference": "reference_raw",
                },
            )
            print(
                "[receiver] dual live pose viewers enabled "
                f"(robot reference_mode={args.view_reference_mode})",
                flush=True,
            )
        except Exception as exc:
            print(f"[receiver] live pose viewers disabled: {exc}", flush=True)
            if robot_viewer is not None:
                robot_viewer.close()
                robot_viewer = None
            if vr_viewer is not None:
                vr_viewer.close()
                vr_viewer = None

    smoother_right = OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5)
    smoother_left = OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5)
    smoother_reference = OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5)

    prev_v_right = VALID_OK
    prev_v_left = VALID_OK
    prev_v_overall = VALID_OK
    prev_v_reference = VALID_OK

    node = dora.Node()
    node.send_output("status", pa.array(["ready"]))

    for event in node:
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue

        recv_ts = receiver.drain_recv_timestamps()
        if recv_ts:
            node.send_output("vr_recv_ts", pa.array(recv_ts, type=pa.int64()))

        msg = receiver.latest()
        if msg is None:
            continue
        now = time.perf_counter()

        v_overall = int(msg["v"]) if "v" in msg else VALID_OK
        v_right = int(msg["vr"]) if "vr" in msg else VALID_OK
        v_left = int(msg["vl"]) if "vl" in msg else VALID_OK

        if v_overall != prev_v_overall:
            print(
                f"[receiver] validity: {_VALID_NAMES[prev_v_overall]} → {_VALID_NAMES[v_overall]} "
                f"(L={_VALID_NAMES[v_left]}, R={_VALID_NAMES[v_right]})"
            )
            prev_v_overall = v_overall

        processed = processor.process(msg)

        if v_right == VALID_INVALID:
            if prev_v_right != VALID_INVALID:
                smoother_right.reset()
            pose_right = None
        else:
            pose_right = smoother_right.smooth(now, processed.mapped_right)

        if v_left == VALID_INVALID:
            if prev_v_left != VALID_INVALID:
                smoother_left.reset()
            pose_left = None
        else:
            pose_left = smoother_left.smooth(now, processed.mapped_left)

        if v_overall == VALID_INVALID:
            if prev_v_reference != VALID_INVALID:
                smoother_reference.reset()
            pose_reference = None
        else:
            pose_reference = smoother_reference.smooth(now, processed.pose_reference)

        prev_v_right = v_right
        prev_v_left = v_left
        prev_v_reference = v_overall

        ts = {"timestamp": time.time_ns()}

        robot_reference = (
            _mapped_reference_pose()
            if args.view_reference_mode == "relationship"
            and pose_reference is not None
            else pose_reference
        )

        if robot_viewer is not None and not robot_viewer.closed:
            robot_viewer.update(
                now=now,
                right=pose_right,
                left=pose_left,
                reference=robot_reference,
                v_right=v_right,
                v_left=v_left,
                v_reference=v_overall,
                extra_lines=[
                    "DORA REFERENCE OUTPUT",
                    _compact_pose_line("pose_reference", pose_reference),
                ],
            )

        vr_right = (
            None if v_right == VALID_INVALID else processed.relative_right
        )
        vr_left = None if v_left == VALID_INVALID else processed.relative_left
        vr_reference = (
            None if v_overall == VALID_INVALID else processed.relative_reference
        )
        if vr_viewer is not None and not vr_viewer.closed:
            vr_viewer.update(
                now=now,
                right=vr_right,
                left=vr_left,
                reference=vr_reference,
                v_right=v_right,
                v_left=v_left,
                v_reference=v_overall,
                extra_lines=[
                    "RAW RH WORLD INPUT",
                    _compact_pose_line("right_raw", processed.raw_right),
                    _compact_pose_line("left_raw", processed.raw_left),
                    _compact_pose_line("reference_raw", processed.raw_reference),
                ],
            )
        
        # print(
        #     "[receiver] joystick: "
        #     f"L=({float(msg.get('lsx', 0.0)):.3f}, "
        #     f"{float(msg.get('lsy', 0.0)):.3f}) "
        #     f"R=({float(msg.get('rsx', 0.0)):.3f}, "
        #     f"{float(msg.get('rsy', 0.0)):.3f})",
        #     flush=True,
        # )

        if pose_right is not None:
            print(
                "[receiver] pose_right: "
                f"pos=({pose_right[0]:.3f}, {pose_right[1]:.3f}, {pose_right[2]:.3f}) "
                f"quat=({pose_right[3]:.3f}, {pose_right[4]:.3f}, {pose_right[5]:.3f}, {pose_right[6]:.3f})",
                flush=True,
            )
            node.send_output("pose_right", pa.array(pose_right, type=pa.float32()), ts)
        if pose_left is not None:
            print(
                "[receiver] pose_left: "
                f"pos=({pose_left[0]:.3f}, {pose_left[1]:.3f}, {pose_left[2]:.3f}) "
                f"quat=({pose_left[3]:.3f}, {pose_left[4]:.3f}, {pose_left[5]:.3f}, {pose_left[6]:.3f})",
                flush=True,
            )
            node.send_output("pose_left", pa.array(pose_left, type=pa.float32()), ts)
        if pose_reference is not None:
            print(
                "[receiver] pose_reference: "
                f"pos=({pose_reference[0]:.3f}, {pose_reference[1]:.3f}, {pose_reference[2]:.3f}) "
                f"quat=({pose_reference[3]:.3f}, {pose_reference[4]:.3f}, {pose_reference[5]:.3f}, {pose_reference[6]:.3f})",
                flush=True,
            )
            node.send_output(
                "pose_reference", pa.array(pose_reference, type=pa.float32()), ts
            )

        if "rt" in msg:
            node.send_output(
                "trigger_right", pa.array([msg["rt"]], type=pa.float32()), ts
            )
        if "lt" in msg:
            node.send_output(
                "trigger_left", pa.array([msg["lt"]], type=pa.float32()), ts
            )
        if "rg" in msg:
            node.send_output(
                "grip_right", pa.array([float(msg["rg"])], type=pa.float32()), ts
            )
        if "lg" in msg:
            node.send_output(
                "grip_left", pa.array([float(msg["lg"])], type=pa.float32()), ts
            )
        if "lsx" in msg:
            node.send_output(
                "joystick_x_left",
                pa.array([float(msg["lsx"])], type=pa.float32()),
                ts,
            )
        if "lsy" in msg:
            node.send_output(
                "joystick_y_left",
                pa.array([float(msg["lsy"])], type=pa.float32()),
                ts,
            )
        if "rsx" in msg:
            node.send_output(
                "joystick_x_right",
                pa.array([float(msg["rsx"])], type=pa.float32()),
                ts,
            )
        if "rsy" in msg:
            node.send_output(
                "joystick_y_right",
                pa.array([float(msg["rsy"])], type=pa.float32()),
                ts,
            )
        if "a" in msg:
            node.send_output(
                "button_a", pa.array([bool(msg["a"])], type=pa.bool_()), ts
            )
        if "b" in msg:
            node.send_output(
                "button_b", pa.array([bool(msg["b"])], type=pa.bool_()), ts
            )
        if "x" in msg:
            node.send_output(
                "button_x", pa.array([bool(msg["x"])], type=pa.bool_()), ts
            )
        if "y" in msg:
            node.send_output(
                "button_y", pa.array([bool(msg["y"])], type=pa.bool_()), ts
            )

    receiver.close()
    if robot_viewer is not None:
        robot_viewer.close()
    if vr_viewer is not None:
        vr_viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta Quest VR pose receiver (dora node)"
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open two non-blocking 3D windows: robot-mapped and VR-reference-relative.",
    )
    parser.add_argument(
        "--view-span",
        type=float,
        default=1.5,
        help="Minimum full width of each 3D axis in meters (default: 1.5).",
    )
    parser.add_argument(
        "--view-axis-length",
        type=float,
        default=0.12,
        help="Length of each pose orientation axis in meters (default: 0.12).",
    )
    parser.add_argument(
        "--view-fps",
        type=float,
        default=30.0,
        help="Maximum viewer refresh rate (default: 30).",
    )
    parser.add_argument(
        "--view-trail",
        type=int,
        default=60,
        help="Number of recent positions retained in each trail; 0 disables trails.",
    )
    parser.add_argument(
        "--view-reference-mode",
        choices=("relationship", "output"),
        default="relationship",
        help=(
            "relationship maps the reference into the controller workspace; "
            "output plots the literal pose_reference output."
        ),
    )
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()