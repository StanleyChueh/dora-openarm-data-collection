#!/usr/bin/env python3
"""Find where ArticulationRootAPI actually landed in a converted USD.

    ${ISAACLAB}/isaaclab.sh -p isaac/inspect_usd.py <abs>/v1_camera_isaac.usd

Run this when Isaac Lab says:

    RuntimeError: Failed to find an articulation when resolving
    '/World/envs/env_0/Robot'. Please ensure that the prim has
    'USD ArticulationRootAPI' applied.

Isaac Lab looks for ArticulationRootAPI at exactly the prim_path you gave it.
When the USD applies that API to a CHILD prim instead of the referenced root,
the lookup fails even though the robot is perfectly fine.  Two things cause it
here:

  * the URDF root is a massless, geometry-less `world` link, so the converter
    has no body to anchor the articulation to at the top level;
  * --fix-base already anchors the base, making that `world` link redundant.

This prints the prim tree plus every prim carrying ArticulationRootAPI, so you
can either fix the URDF (see make_isaac_urdf.py --strip-world-link, the
preferred fix) or point ArticulationCfg.prim_path at the prim it names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pxr import Usd, UsdPhysics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("usd")
    p.add_argument("--max-depth", type=int, default=4)
    args = p.parse_args()

    path = str(Path(args.usd).expanduser().resolve())
    stage = Usd.Stage.Open(path)
    if stage is None:
        sys.exit(f"could not open {path}")

    default_prim = stage.GetDefaultPrim()
    print(f"file        : {path}")
    print(f"default prim: {default_prim.GetPath() if default_prim else '(none!)'}")
    if not default_prim:
        print("  !! no default prim -- Isaac Lab's UsdFileCfg reference will be empty")

    roots, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath())
        if prim.IsA(UsdPhysics.Joint):
            joints.append((prim.GetPath(), prim.GetTypeName()))

    print()
    print("prim tree:")
    for prim in stage.Traverse():
        rel = prim.GetPath().pathString.strip("/").split("/")
        if len(rel) > args.max_depth:
            continue
        marks = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            marks.append("ARTICULATION_ROOT")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            marks.append("rigid")
        suffix = ("   <-- " + ", ".join(marks)) if marks else ""
        print(f"  {'  ' * (len(rel) - 1)}{prim.GetName()}  ({prim.GetTypeName()}){suffix}")

    print()
    print(f"joints: {len(joints)}")
    print(f"ArticulationRootAPI on {len(roots)} prim(s):")
    for r in roots:
        print(f"  {r}")

    if not roots:
        print()
        print("NONE.  The converter never applied it.  Regenerate the URDF with")
        print("  python3 isaac/make_isaac_urdf.py ... --strip-world-link")
        print("and re-convert with --fix-base.")
        sys.exit(1)

    if default_prim and roots[0] != default_prim.GetPath():
        rel = roots[0].pathString.replace(
            default_prim.GetPath().pathString, ""
        ).lstrip("/")
        print()
        print("The API is NOT on the default prim, which is why the lookup at")
        print("'{ENV_REGEX_NS}/Robot' failed.  Either:")
        print(f"  (a) set prim_path=\"{{ENV_REGEX_NS}}/Robot/{rel}\", or")
        print("  (b) preferred: regenerate the URDF with --strip-world-link so")
        print("      openarm_body_link0 becomes the real root, then re-convert.")
        sys.exit(1)

    print()
    print("OK: ArticulationRootAPI is on the default prim.")


if __name__ == "__main__":
    main()
