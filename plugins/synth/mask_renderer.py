"""Silhouette + per-link instance mask renderer for the xarm7.

Consumes the same camera intrinsics / extrinsics the Isaac Sim render loop
writes to each view's sidecar JSON (see ``camera_calibration`` in
``render_cube.py``), so a mask rendered here should overlay the Isaac RGB
frame pixel-for-pixel.

Backend is a minimal CPU z-buffer rasterizer. It's slow but dependency-free
and correct; once the geometry pipeline is validated, swap the inner
rasterize loop for ``jaxrenderer`` or ``nvdiffrast``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy


# Link names we want to include in the mask, in the order render_cube.py
# rotates them. The ``world`` link has no geometry.
ARM_LINK_NAMES = [
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
]
GRIPPER_LINK_NAMES = [
    "xarm_gripper_base_link",
    "left_outer_knuckle", "left_finger", "left_inner_knuckle",
    "right_outer_knuckle", "right_finger", "right_inner_knuckle",
]


@dataclass(frozen=True)
class LinkMesh:
    name: str
    link_id: int
    vertices: np.ndarray  # (V, 3) float32, in the link's local frame
    triangles: np.ndarray  # (T, 3) int32


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_sidecar(cls, intr: dict) -> "Intrinsics":
        return cls(
            fx=float(intr["fx"]), fy=float(intr["fy"]),
            cx=float(intr["cx"]), cy=float(intr["cy"]),
            width=int(intr["width"]), height=int(intr["height"]),
        )


class MaskRenderer:
    """Rasterize a binary + per-link instance mask of the xarm7 robot.

    Units: meters, matching the URDF and the robot scene in render_cube.py.
    """

    def __init__(self, urdf_path: Path, include_gripper: bool = True):
        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self._link_meshes: list[LinkMesh] = []
        names = list(ARM_LINK_NAMES)
        if include_gripper:
            names += GRIPPER_LINK_NAMES
        urdf_dir = Path(urdf_path).resolve().parent
        for link_id, link_name in enumerate(names, start=1):
            link = self.urdf.link_map[link_name]
            for visual in link.visuals:
                mesh = _load_visual_mesh(visual, urdf_dir)
                if mesh is None:
                    continue
                # Bake the visual's origin into vertex positions.
                origin = _origin_matrix(visual.origin)
                verts = _transform_points(mesh.vertices, origin).astype(np.float32)
                tris = np.asarray(mesh.faces, dtype=np.int32)
                self._link_meshes.append(
                    LinkMesh(name=link_name, link_id=link_id, vertices=verts, triangles=tris)
                )
        self.link_names = names

    # ------------------------------------------------------------------
    # Public API

    def render(
        self,
        joint_angles_rad: dict[str, float],
        world_to_camera_row: np.ndarray,
        intrinsics: Intrinsics,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render (binary_mask uint8 HxW, instance_mask uint8 HxW).

        ``world_to_camera_row`` is the USD convention matrix: (4,4), applied
        as ``p_cam_row = p_world_row @ M``. Camera looks down -Z.
        """
        self.urdf.update_cfg(joint_angles_rad)

        W, H = intrinsics.width, intrinsics.height
        depth_buf = np.full((H, W), np.inf, dtype=np.float32)
        inst_buf = np.zeros((H, W), dtype=np.uint8)

        for lm in self._link_meshes:
            link_to_world = self.urdf.get_transform(lm.name)  # (4,4) column-vec
            # Convert to row-vec convention so we can chain with world_to_camera_row.
            link_to_world_row = link_to_world.T
            verts_cam = _row_transform(
                _row_transform(lm.vertices, link_to_world_row),
                world_to_camera_row,
            )
            _rasterize_mesh(
                verts_cam, lm.triangles, lm.link_id,
                intrinsics, depth_buf, inst_buf,
            )

        binary = (inst_buf > 0).astype(np.uint8) * 255
        return binary, inst_buf


# ----------------------------------------------------------------------
# Helpers

def _load_visual_mesh(visual, urdf_dir: Path) -> trimesh.Trimesh | None:
    geom = visual.geometry
    if geom.mesh is None:
        return None
    fname = urdf_dir / geom.mesh.filename
    mesh = trimesh.load(str(fname), force="mesh")
    if geom.mesh.scale is not None:
        s = np.asarray(geom.mesh.scale, dtype=np.float64)
        mesh.apply_scale(s)
    return mesh


def _origin_matrix(origin: np.ndarray | None) -> np.ndarray:
    if origin is None:
        return np.eye(4)
    return np.asarray(origin, dtype=np.float64)


def _transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Column-vector convention: p' = M @ p."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (pts_h @ M.T)[:, :3]


def _row_transform(pts: np.ndarray, M_row: np.ndarray) -> np.ndarray:
    """Row-vector convention: p' = p @ M."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (pts_h @ M_row)[:, :3]


def _rasterize_mesh(
    verts_cam: np.ndarray,   # (V, 3) in camera frame (-Z forward)
    tris: np.ndarray,        # (T, 3)
    link_id: int,
    intr: Intrinsics,
    depth_buf: np.ndarray,
    inst_buf: np.ndarray,
) -> None:
    W, H = intr.width, intr.height
    depth = -verts_cam[:, 2]                        # (V,) positive in front
    valid = depth > 1e-6
    # Project. For invalid (behind camera) vertices we fill sentinel; triangles
    # touching any invalid vertex are skipped below.
    safe_depth = np.where(valid, depth, 1.0)
    x_px = intr.fx * (verts_cam[:, 0] / safe_depth) + intr.cx
    y_px = intr.cy - intr.fy * (verts_cam[:, 1] / safe_depth)
    v2 = np.stack([x_px, y_px], axis=1).astype(np.float32)

    # Per-triangle early rejects.
    tri_valid = valid[tris].all(axis=1)
    p0 = v2[tris[:, 0]]
    p1 = v2[tris[:, 1]]
    p2 = v2[tris[:, 2]]
    d0 = depth[tris[:, 0]]
    d1 = depth[tris[:, 1]]
    d2 = depth[tris[:, 2]]

    # Signed area (in screen space) for back-face culling. Note: the Y axis
    # was flipped during projection, so positive screen-area triangles here
    # correspond to front-facing when the original mesh is CCW in world.
    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) \
          - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    # Don't backface-cull — URDF mesh winding is inconsistent across links.
    tri_valid &= np.abs(area2) > 1e-6

    # Tight screen-space bbox per triangle.
    xmin = np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    xmax = np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    ymin = np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)
    ymax = np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)
    xmin = np.clip(xmin, 0, W)
    xmax = np.clip(xmax, 0, W)
    ymin = np.clip(ymin, 0, H)
    ymax = np.clip(ymax, 0, H)
    tri_valid &= (xmax > xmin) & (ymax > ymin)

    idx = np.nonzero(tri_valid)[0]
    for i in idx:
        _rasterize_one(
            p0[i], p1[i], p2[i], d0[i], d1[i], d2[i], area2[i],
            int(xmin[i]), int(xmax[i]), int(ymin[i]), int(ymax[i]),
            link_id, depth_buf, inst_buf,
        )


def _rasterize_one(
    a, b, c, da, db, dc, area2,
    xmin, xmax, ymin, ymax,
    link_id, depth_buf, inst_buf,
) -> None:
    xs = np.arange(xmin, xmax, dtype=np.float32) + 0.5
    ys = np.arange(ymin, ymax, dtype=np.float32) + 0.5
    # (H_box, W_box) grids
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    # Barycentrics via edge functions.
    w0 = (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])
    w1 = (c[0] - b[0]) * (Y - b[1]) - (c[1] - b[1]) * (X - b[0])
    w2 = (a[0] - c[0]) * (Y - c[1]) - (a[1] - c[1]) * (X - c[0])
    # Sign depends on triangle winding. Accept either all-same-sign.
    if area2 > 0:
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    else:
        inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
    if not inside.any():
        return
    # Perspective-correct depth interpolation via 1/z.
    inv_area = 1.0 / area2
    # Match barycentric naming: l0 weights vertex A, computed from edge BC.
    l0 = ((b[0] - c[0]) * (Y - c[1]) - (b[1] - c[1]) * (X - c[0])) * inv_area
    l1 = ((c[0] - a[0]) * (Y - a[1]) - (c[1] - a[1]) * (X - a[0])) * inv_area
    l2 = 1.0 - l0 - l1
    inv_z = l0 / da + l1 / db + l2 / dc
    z = 1.0 / np.maximum(inv_z, 1e-9)
    # Write through the z-buffer.
    existing = depth_buf[ymin:ymax, xmin:xmax]
    hit = inside & (z < existing)
    if not hit.any():
        return
    existing[hit] = z[hit]
    inst_buf[ymin:ymax, xmin:xmax][hit] = np.uint8(link_id)


def joints_from_sidecar(joints: dict) -> dict[str, float]:
    """Convert the ``joints`` dict in a sidecar to radians keyed by URDF joint.

    Sidecar is degrees + ``gripper_angle`` driving the ``drive_joint``.
    """
    out: dict[str, float] = {}
    for i in range(1, 8):
        key = f"joint{i}"
        if key in joints:
            out[key] = math.radians(float(joints[key]))
    if "gripper_angle" in joints:
        out["drive_joint"] = math.radians(float(joints["gripper_angle"]))
    return out
