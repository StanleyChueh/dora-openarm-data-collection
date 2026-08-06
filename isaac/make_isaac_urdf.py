#!/usr/bin/env python3
"""Prepare openarm_description's v1 example URDF for import into Isaac Sim.

    python3 isaac/make_isaac_urdf.py \
        openarm_description-main/assets/robot/openarm_v1.0/urdf/example/v1_camera.urdf \
        openarm_description-main/v1_camera_isaac.urdf

Write the output at the PACKAGE ROOT (next to assets/), because mesh filenames
become `assets/...` relative to that root, and Isaac's URDF importer resolves
relative mesh paths against the URDF file's own directory.

Three transformations, none of which touches the source file:

1. `package://openarm_description/` is stripped from every <mesh filename>.
   Isaac's importer is not ROS-aware and will not resolve package:// unless the
   package happens to be on AMENT_PREFIX_PATH.

2. Physically impossible inertia tensors are repaired.
   All four finger links carry ixy = ixz = iyz = 1e-06 -- three identical round
   numbers, i.e. placeholders rather than measured products of inertia.  They
   are large enough relative to the diagonal (2.375e-06 / 7.5e-07) to break the
   triangle inequality on the tensor's eigenvalues, which makes the tensor
   physically unrealisable:

       eigenvalues 1.331e-07, 1.375e-06, 3.992e-06  ->  A + B < C

   MuJoCo refuses to load it outright.  PhysX accepts it and gives you fingers
   with nonsense rotational dynamics, which is worse -- it fails silently.

   The repair zeroes the products of inertia and keeps the authored diagonal,
   the minimal change that restores validity.  Pass --no-fix-inertia to leave
   the source values alone.

3. A real grasp frame is appended per arm.
   The stock `openarm_{side}_hand_tcp` is at link7's origin (its fixed joint is
   xyz="0 0 0"), i.e. at the WRIST, not between the fingers.  Driving IK to it
   would map the VR controller to the wrist and leave teleop feeling a
   palm-length off.

   Measured from the collision meshes (their vertices are in a shared assembly
   frame; each <collision> carries an <origin> mapping them into link
   coordinates, and the meshes are in millimetres via scale="0.001 ..."):

       link7 geometry   z in [-0.018, +0.095]   (link7 frame)
       finger mount     z  = 0.1025
       finger geometry  z in [ 0.088, +0.183]   (link7 frame)

   so the fingertips reach z = 0.183 and the pads sit around z = 0.165.

This URDF already has <mimic> on finger_joint2, TCP links and camera links, and
its joint names already match what the dora pipeline hardcodes
(openarm_{side}_joint1..7, openarm_{side}_finger_joint1), so none of those need
touching.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np

PACKAGE_PREFIX = "package://openarm_description/"

# Parent is hand_tcp (the designated tool frame, currently identity w.r.t.
# link7) rather than link7 itself, so this tracks upstream if they ever give
# hand_tcp a real offset.
GRASP_FRAMES = [
    ("openarm_left_grasp", "openarm_left_hand_tcp"),
    ("openarm_right_grasp", "openarm_right_hand_tcp"),
]
DEFAULT_GRASP_Z = 0.165

OFF_DIAGONAL = ("ixy", "ixz", "iyz")


def strip_world_link(root: ET.Element) -> str | None:
    """Drop a massless, geometry-less root `world` link and its fixed joint.

    v1_camera.urdf starts with

        <link name="world"/>
        <joint name="openarm_body_world_joint" type="fixed">
          <parent link="world"/><child link="openarm_body_link0"/>

    which is the usual ROS convention for anchoring a robot.  Isaac's URDF
    converter does not need it -- --fix-base anchors the base -- and a root link
    with no inertia, no collision and no visual gives it nothing to attach the
    articulation to at the top level, so ArticulationRootAPI can end up on a
    child prim.  Isaac Lab then fails with

        Failed to find an articulation when resolving '.../Robot'

    even though the robot itself is fine.  Removing the link makes
    openarm_body_link0 the real root.

    Returns the name of the new root, or None if there was no world link.
    """
    links = {link.get("name"): link for link in root.iter("link")}
    children = {j.find("child").get("link") for j in root.iter("joint")}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link, found {roots}")
    root_name = roots[0]

    world = links[root_name]
    # Only strip a genuinely empty anchor; a root that carries real geometry is
    # the robot itself and must stay.
    if any(world.find(tag) is not None for tag in ("inertial", "visual", "collision")):
        return None

    anchor_joints = [
        j for j in root.iter("joint") if j.find("parent").get("link") == root_name
    ]
    if len(anchor_joints) != 1 or anchor_joints[0].get("type") != "fixed":
        return None

    new_root = anchor_joints[0].find("child").get("link")
    root.remove(world)
    root.remove(anchor_joints[0])
    return new_root


def strip_package_uris(root: ET.Element, prefix: str) -> int:
    n = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(prefix):
            mesh.set("filename", filename[len(prefix) :])
            n += 1
    return n


def _is_realisable(inertia: ET.Element) -> bool:
    """Valid inertia: positive-definite, and the eigenvalues obey the triangle
    inequality (any two principal moments sum to at least the third).  Checking
    eigenvalues rather than the raw ixx/iyy/izz is what catches a tensor that
    only looks fine on the diagonal."""
    v = {k: float(inertia.get(k, 0.0)) for k in ("ixx", "iyy", "izz", *OFF_DIAGONAL)}
    tensor = np.array(
        [
            [v["ixx"], v["ixy"], v["ixz"]],
            [v["ixy"], v["iyy"], v["iyz"]],
            [v["ixz"], v["iyz"], v["izz"]],
        ]
    )
    e = np.sort(np.linalg.eigvalsh(tensor))
    return bool(e[0] > 0.0 and e[0] + e[1] >= e[2])


def fix_inertias(root: ET.Element) -> list[str]:
    fixed: list[str] = []
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        if inertia is None or _is_realisable(inertia):
            continue
        for key in OFF_DIAGONAL:
            inertia.set(key, "0")
        if not _is_realisable(inertia):
            raise SystemExit(
                f"{link.get('name')}: inertia still unrealisable after zeroing "
                "the products of inertia -- the diagonal is bad too"
            )
        fixed.append(link.get("name"))
    return fixed


def append_grasp_frames(root: ET.Element, grasp_z: float) -> None:
    existing = {link.get("name") for link in root.iter("link")}
    for grasp, parent in GRASP_FRAMES:
        if parent not in existing:
            raise SystemExit(f"parent link {parent} not found")
        if grasp in existing:
            raise SystemExit(f"link {grasp} already exists; refusing to duplicate")

        ET.SubElement(root, "link", {"name": grasp})
        joint = ET.SubElement(
            root, "joint", {"name": f"{grasp}_joint", "type": "fixed"}
        )
        # rpy stays "0 0 0" so the grasp frame keeps link7's orientation.  The
        # VR quaternion convention in quest_receiver.py was tuned against that
        # frame; any rotation here silently desyncs it.
        ET.SubElement(joint, "origin", {"xyz": f"0 0 {grasp_z}", "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": grasp})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="v1_camera.urdf from openarm_description")
    p.add_argument("output", help="generated URDF (write it at the package root)")
    p.add_argument(
        "--grasp-z",
        type=float,
        default=DEFAULT_GRASP_Z,
        help=f"grasp frame offset along link7 z, metres (default {DEFAULT_GRASP_Z}; "
        "finger mount is at 0.1025, fingertips at 0.183)",
    )
    p.add_argument("--package-prefix", default=PACKAGE_PREFIX)
    p.add_argument(
        "--no-fix-inertia",
        action="store_true",
        help="keep the source inertia tensors even if they are unrealisable",
    )
    p.add_argument(
        "--keep-world-link",
        action="store_true",
        help="keep the empty root `world` link (it usually breaks "
        "ArticulationRootAPI placement on USD import)",
    )
    args = p.parse_args()

    tree = ET.parse(args.source)
    root = tree.getroot()

    new_root = None if args.keep_world_link else strip_world_link(root)
    n_meshes = strip_package_uris(root, args.package_prefix)
    fixed = [] if args.no_fix_inertia else fix_inertias(root)
    append_grasp_frames(root, args.grasp_z)

    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    print(f"wrote {args.output}")
    if new_root:
        print(f"  root `world` link removed : new root is {new_root}")
    elif args.keep_world_link:
        print("  root `world` link         : kept (--keep-world-link)")
    else:
        print("  root `world` link         : none to remove")
    print(f"  package:// URIs rewritten : {n_meshes}")
    print(
        f"  grasp frames appended     : "
        f"{', '.join(g for g, _ in GRASP_FRAMES)} at z={args.grasp_z}"
    )
    if fixed:
        print(f"  inertia tensors repaired  : {len(fixed)} ({', '.join(fixed)})")
    elif args.no_fix_inertia:
        print("  inertia repair            : skipped (--no-fix-inertia)")
    else:
        print("  inertia tensors repaired  : none needed")


if __name__ == "__main__":
    main()
