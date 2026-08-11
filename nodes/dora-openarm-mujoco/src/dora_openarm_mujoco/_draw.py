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

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def draw_arrow(
    scn, direction: np.ndarray, origin: np.ndarray, color: tuple, size: float
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    z = direction / (np.linalg.norm(direction) + 1e-9)
    x = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(z, x)) > 0.99:
        x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    arrow_mat = np.stack([x, y, z], axis=1)
    center = origin + direction * size / 2
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.005, 0.005, size]),
        center,
        arrow_mat.flatten(),
        np.array(color, dtype=np.float32),
    )
    g.size[:] = [0.005, 0.005, size]
    g.pos[:] = center
    g.mat[:] = arrow_mat
    scn.ngeom += 1


def draw_world_frame(scn, size: float = 0.3) -> None:
    draw_arrow(scn, np.array([1.0, 0.0, 0.0]), np.zeros(3), (1.0, 0.1, 0.1, 1.0), size)
    draw_arrow(scn, np.array([0.0, 1.0, 0.0]), np.zeros(3), (0.1, 0.8, 0.1, 1.0), size)
    draw_arrow(scn, np.array([0.0, 0.0, 1.0]), np.zeros(3), (0.1, 0.3, 1.0, 1.0), size)


def draw_frame(scn, pose: np.ndarray, size: float = 0.08) -> None:
    """Draw the pose of a VR controller as a coordinate frame."""
    if pose is None:
        return
    pos = pose[:3]
    rot = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])  # xyzw
    mat = rot.as_matrix()
    draw_arrow(scn, mat[:, 0], pos, (1.0, 0.1, 0.1, 1.0), size)  # X (Red)
    draw_arrow(scn, mat[:, 1], pos, (0.1, 0.8, 0.1, 1.0), size)  # Y (Green)
    draw_arrow(scn, mat[:, 2], pos, (0.1, 0.3, 1.0, 1.0), size)  # Z (Blue)
