"""Regenerate the v1_camera assets and *base* MJCF from the IsaacLab URDF.

Usage:
    python convert_from_urdf.py [URDF_PATH]

Pipeline: read v1_camera.urdf -> strip XML comments (they contain unexpanded
${ee_prefix} xacro leftovers) -> rebuild assets/ from the meshes the URDF
references -> rewrite package:// mesh URIs to paths relative to assets/ ->
inject a <mujoco> compiler extension -> compile with MuJoCo's native URDF
importer -> mj_saveLastXML.

MESH HANDLING -- the reason this script exists rather than a `cp`. MuJoCo
cannot load Collada, so every visual mesh has to become an STL. An .stl sits
next to each .dae in the source tree, but they are NOT reliably the same
geometry: cam.STL has cyclically permuted axes and a ~60 mm origin shift, and
finger.stl is a different, larger mesh offset by (0.9, 53.8, 74.2) mm -- using
them puts the wrist camera housing in the wrong place and buries the gripper
inside the wrist motor. link0 differs by 1 mm; the rest happen to match. So the
visual STLs are always generated here from the .dae the URDF actually names,
and the shipped .stl files next to them are never used. Collision meshes are
referenced as .stl by the URDF itself and are copied verbatim.

The committed v1_camera_robot.xml is this script's base output plus
hand-finished additions that URDF cannot express (do NOT blindly overwrite it):
  - 16 position actuators using the v2 naming JointResolver requires
    ({side}_joint{i}_ctrl / {side}_finger1_ctrl)
  - <equality> joint couplings for finger_joint2 (URDF <mimic> is ignored
    by the importer)
  - sites: arm_origin, left/right_ee_control_point, the latter two carrying a
    quat="0 1 0 0" so the EE frame matches openarm_control's v2 convention
  - <camera name="camera_wrist_left/right"> inside the camera-housing bodies.
    Their pos/quat were solved numerically at the home keyframe: optical
    center at world (0.342, +/-0.153, 0.548) -- just past the housing's front
    face, which ends at x=0.335 -- aimed at (0.455, +/-0.153, 0.425), the
    fingertip-to-workspace zone, with world +z as the image up direction; the
    world poses were then transformed into the camera_link body frame. Re-solve
    these if the cam mesh or its mount offset ever changes.
  - joint damping/frictionloss/armature defaults copied from the v2 model's
    motor classes (same DM-series motors by actuatorfrcrange)
  - geom groups remapped to the v2 convention: visual 1 -> 2, collision
    (unset, i.e. 0) -> 3. MuJoCo shows groups 0/1/2 by default, so the
    importer's own assignment drew the simplified collision hulls on top of
    the visual meshes in the viewer and in every rendered camera frame.
  - a fix for an importer quirk: openarm_right_right_finger must use the
    mirrored finger meshes (finger1/finger_col1), matching the URDF's
    negative y-scale, but the importer assigns the unmirrored ones.
"""

import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_URDF = (
    "/home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/"
    "v1_camera_isaac/urdf/v1_camera.urdf"
)

HERE = Path(__file__).resolve().parent
COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def dae_triangles(path: Path) -> np.ndarray:
    """Return a (n, 3, 3) array of triangle vertices from a Collada file."""
    root = ET.parse(path).getroot()
    tris: list[np.ndarray] = []
    for geometry in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        mesh = geometry.find("c:mesh", COLLADA_NS)
        vertices = mesh.find("c:vertices", COLLADA_NS)
        source_id = next(
            i.get("source")[1:]
            for i in vertices.findall("c:input", COLLADA_NS)
            if i.get("semantic") == "POSITION"
        )
        raw = mesh.find(f"c:source[@id='{source_id}']/c:float_array", COLLADA_NS).text
        verts = np.array(raw.split(), dtype=np.float64).reshape(-1, 3)

        primitives = mesh.findall("c:triangles", COLLADA_NS)
        primitives += mesh.findall("c:polylist", COLLADA_NS)
        for prim in primitives:
            inputs = prim.findall("c:input", COLLADA_NS)
            stride = max(int(i.get("offset", 0)) for i in inputs) + 1
            offset = next(
                int(i.get("offset", 0))
                for i in inputs
                if i.get("semantic") == "VERTEX"
                and i.get("source")[1:] == vertices.get("id")
            )
            idx = np.array(prim.find("c:p", COLLADA_NS).text.split(), dtype=np.int64)
            idx = idx.reshape(-1, stride)[:, offset]

            vcount = prim.find("c:vcount", COLLADA_NS)
            if vcount is None:  # <triangles>
                tris.extend(verts[f] for f in idx.reshape(-1, 3))
                continue
            start = 0  # <polylist>: fan-triangulate each polygon
            for n in map(int, vcount.text.split()):
                face = idx[start : start + n]
                start += n
                tris.extend(verts[[face[0], face[k], face[k + 1]]] for k in range(1, n - 1))
    return np.array(tris, dtype=np.float32)


def write_binary_stl(tris: np.ndarray, out_path: Path, header: str) -> None:
    with open(out_path, "wb") as f:
        f.write(header.encode()[:79].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else np.zeros(3)
            f.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))


def asset_name(mesh_rel: str) -> tuple[str, str]:
    """Map a URDF mesh path to (subdir, filename) under assets/."""
    base = mesh_rel.rsplit("/", 1)[-1]
    if "/visual/" in mesh_rel:
        return "visual", re.sub(r"\.(dae|stl|STL)$", ".stl", base)
    # the collision finger would otherwise collide with the visual one's name
    return "collision", "finger_col.stl" if base == "finger.stl" else base


def rebuild_assets(urdf_text: str, mesh_root: Path) -> None:
    assets = HERE / "assets"
    for sub in ("visual", "collision"):
        (assets / sub).mkdir(parents=True, exist_ok=True)

    for mesh_rel in sorted(set(re.findall(r"package://v1_camera_isaac/mesh/([^\"]+)", urdf_text))):
        sub, name = asset_name(mesh_rel)
        src, dst = mesh_root / mesh_rel, assets / sub / name
        if src.suffix.lower() == ".dae":
            tris = dae_triangles(src)
            write_binary_stl(tris, dst, f"{name} generated from {mesh_rel}")
            print(f"  {mesh_rel} -> {sub}/{name}  ({len(tris)} tris, from Collada)")
        else:
            shutil.copyfile(src, dst)
            print(f"  {mesh_rel} -> {sub}/{name}  (copied)")


def rewrite_mesh_uri(match: re.Match) -> str:
    sub, name = asset_name(match.group(1))
    return f'filename="{sub}/{name}"'


def main() -> None:
    urdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_URDF)
    urdf = re.sub(r"<!--.*?-->", "", urdf_path.read_text(), flags=re.S)

    print(f"Rebuilding assets/ from {urdf_path.parent.parent / 'mesh'}:")
    rebuild_assets(urdf, urdf_path.parent.parent / "mesh")

    urdf = re.sub(
        r'filename="package://v1_camera_isaac/mesh/([^"]+)"', rewrite_mesh_uri, urdf
    )
    assert "package://" not in urdf and ".dae" not in urdf and ".STL" not in urdf
    urdf = urdf.replace(
        '<robot name="openarm">',
        '<robot name="openarm">\n  <mujoco>\n    <compiler meshdir="assets"'
        ' balanceinertia="true" discardvisual="false" fusestatic="false"/>\n'
        "  </mujoco>",
    )

    work = HERE / "_convert_v1_camera.urdf"
    work.write_text(urdf)
    try:
        model = mujoco.MjModel.from_xml_path(str(work))
        out = HERE / "_v1_camera_base.xml"
        mujoco.mj_saveLastXML(str(out), model)
    finally:
        work.unlink(missing_ok=True)
    print(f"nq={model.nq} nbody={model.nbody} nmesh={model.nmesh} -> {out}")


if __name__ == "__main__":
    main()
