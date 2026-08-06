#!/usr/bin/env python3
"""Bring-up steps V1/V2: measure the frames the bridge's constants depend on.

    # V1 -- measure arm_origin in the v2 scene dora actually runs
    python3 isaac/check_frames.py --scene demo
    python3 isaac/check_frames.py --mjcf /path/to/some_scene.xml

    # V2 -- where the Isaac robot's grasp frames sit
    python3 isaac/check_frames.py --urdf openarm_description-main/v1_camera_isaac.urdf

Why this exists
---------------
`ARM_ORIGIN_TO_BASE_Z` in bridge.py converts the VR poses from the `arm_origin`
frame into the robot base frame.  It CANNOT be derived from the Isaac robot,
because dora runs OpenArm **v2** (dora_openarm_mujoco/main.py:132 hardcodes
`import openarm_mujoco.v2`) while Isaac runs **v1** -- the number bridges two
different models.

`arm_origin` is a site in the v2 MuJoCo scene, so the only way to know it is to
load that scene and read it.  bridge.py defaults to 0.698 purely because both
v1 and v2 mount their arms at xyz="0 +-0.031 0.698" from openarm_body_link0.
That is an inference.  This script turns it into a measurement.

Run it on the machine where `openarm_mujoco` is installed (the dora host); it
needs no ROS, no Isaac, and no GPU.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def resolve_scene(name: str) -> str:
    """Resolve a bundled scene name through the installed openarm_mujoco."""
    try:
        import openarm_mujoco.v2 as openarm_mujoco
    except ImportError:
        raise SystemExit(
            "openarm_mujoco is not installed here.  Run this on the dora host, "
            "or pass --mjcf with an explicit path."
        )
    resolvers = {
        "cell": openarm_mujoco.openarm_cell_xml,
        "demo": openarm_mujoco.openarm_demo_xml,
        "pedestal": openarm_mujoco.openarm_pedestal_xml,
    }
    if name not in resolvers:
        raise SystemExit(f"unknown scene {name!r}; choose from {sorted(resolvers)}")
    return resolvers[name]()


def report_mjcf(path: str, keyframe: str) -> int:
    import mujoco

    print(f"== V1  MuJoCo scene: {path}")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        print(f"  (no keyframe {keyframe!r}; using the default configuration)")
    mujoco.mj_forward(model, data)

    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")
    if oid < 0:
        print("  !! this scene has no site named 'arm_origin'")
        sites = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(model.nsite)
        ]
        print(f"     sites present: {sites}")
        return 1

    pos = np.array(data.site_xpos[oid], dtype=np.float64)
    mat = np.array(data.site_xmat[oid], dtype=np.float64).reshape(3, 3)
    print(f"  arm_origin position   = {np.round(pos, 6)}")
    print(f"  arm_origin rotation   =\n{np.round(mat, 6)}")

    identity = np.allclose(mat, np.eye(3), atol=1e-6)
    centred = abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6
    print()
    print(f"  ==> set ARM_ORIGIN_TO_BASE_Z={pos[2]:.6f}")
    if not centred:
        print(
            f"  !! x/y are not zero ({pos[0]:.6f}, {pos[1]:.6f}).  bridge.py only "
            "applies a z translation, so it would need extending to carry these."
        )
    if not identity:
        print(
            "  !! rotation is NOT identity.  bridge.py passes the quaternion "
            "through unchanged, which assumes it is."
        )
    return 0 if (centred and identity) else 1


def report_urdf(path: str) -> int:
    import mujoco

    print(f"== V2  Isaac URDF: {path}")
    # MuJoCo merges massless fixed-joint links, so the grasp frames do not
    # survive as bodies here -- reconstruct them from link7, which is exactly
    # what Isaac Lab's Jacobian body will be.
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"  parses, meshes resolve: nq={model.nq} nv={model.nv}")

    for side in ("left", "right"):
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link7"
        )
        if bid < 0:
            print(f"  !! openarm_{side}_link7 not found")
            return 1
        rot = np.array(data.xmat[bid]).reshape(3, 3)
        pos = np.array(data.xpos[bid])
        print(f"  openarm_{side}_link7 @ qpos=0  pos={np.round(pos, 5)}")
        for label, z in (
            ("finger mount", 0.1025),
            ("grasp frame ", 0.165),
            ("fingertip   ", 0.183),
        ):
            print(
                f"      {label} z={z:.4f} -> {np.round(pos + rot @ np.array([0, 0, z]), 5)}"
            )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", help="bundled openarm_mujoco v2 scene: cell|demo|pedestal")
    p.add_argument("--mjcf", help="explicit MJCF path (overrides --scene)")
    p.add_argument("--keyframe", default="home")
    p.add_argument("--urdf", help="the generated Isaac URDF")
    args = p.parse_args()

    if not any((args.scene, args.mjcf, args.urdf)):
        p.error("give --scene, --mjcf and/or --urdf")

    status = 0
    if args.mjcf or args.scene:
        status |= report_mjcf(args.mjcf or resolve_scene(args.scene), args.keyframe)
    if args.urdf:
        if status:
            print()
        status |= report_urdf(args.urdf)
    sys.exit(status)


if __name__ == "__main__":
    main()
