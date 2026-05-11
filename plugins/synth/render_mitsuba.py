from __future__ import annotations

import colorsys
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import tyro
import yourdfpy

os.environ.setdefault("DRJIT_NO_LLVM_WARNING", "1")
import mitsuba as mi
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")
mi.set_log_level(mi.LogLevel.Error)

from background import CocoBackgroundSampler
from background import composite as composite_bg
from mask_renderer import Intrinsics as MaskIntrinsics
from mask_renderer import MaskRenderer

HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "xarm7_standalone.urdf"
PLY_CACHE = HERE / "assets_ply"

_MESH_BOUNDS_CACHE: dict[str, np.ndarray] = {}


def _mesh_bounds(path: Path) -> np.ndarray:
    """Return (2, 3) bounds array for a mesh, loading from disk only once."""
    key = str(path)
    if key not in _MESH_BOUNDS_CACHE:
        _MESH_BOUNDS_CACHE[key] = np.asarray(trimesh.load(path, force="mesh").bounds)
    return _MESH_BOUNDS_CACHE[key]

ROOM_X = 4.5
ROOM_Y_NEG = -4.5
ROOM_Y_POS = 4.5
ROOM_Z_MIN = -3.0
ROOM_Z_MAX = 4.5
ROOM_MARGIN = 0.08

CAMERA_APERTURE = 20.955
FOCAL_LENGTH_BASE = 24.0
INCH_TO_METER = 0.0254

ARM_PHYSICAL_LIMITS_DEG = [
    (-360.0, 360.0),   # joint1
    (-118.0, 120.0),   # joint2
    (-360.0, 360.0),   # joint3
    (-11.0,  225.0),   # joint4
    (-360.0, 360.0),   # joint5
    (-97.0,  180.0),   # joint6
    (-360.0, 360.0),   # joint7
]
ARM_TIGHT_LIMITS_DEG = [
    (-120.0, 120.0),
    (-90.0,   60.0),
    (-150.0, 150.0),
    (-11.0,  100.0),
    (-150.0, 150.0),
    (-90.0,   90.0),
    (-175.0, 175.0),
]


def robot_visual_links(urdf: yourdfpy.URDF) -> list[str]:
    names: list[str] = []
    for link in urdf.robot.links:
        for v in link.visuals:
            if v.geometry and v.geometry.mesh and v.geometry.mesh.filename:
                names.append(link.name)
                break
    return names


KEYPOINT_JOINTS = [
    ("base", None, "link_base"),
    ("joint1", "joint1", None),
    ("joint2", "joint2", None),
    ("joint3", "joint3", None),
    ("joint4", "joint4", None),
    ("joint5", "joint5", None),
    ("joint6", "joint6", None),
    ("joint7", "joint7", None),
    ("eef", "joint_eef", None),
    ("tcp", "joint_tcp", None),
    ("gripper_drive", "drive_joint", None),
    ("gripper_left_finger", "left_finger_joint", None),
    ("gripper_left_inner", "left_inner_knuckle_joint", None),
    ("gripper_right_outer", "right_outer_knuckle_joint", None),
    ("gripper_right_finger", "right_finger_joint", None),
    ("gripper_right_inner", "right_inner_knuckle_joint", None),
]


def rand_vivid_color(rng: random.Random) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.55, 0.9), rng.uniform(0.72, 0.98))


def rand_broad_color(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    hue = rng.random()
    sat = rng.uniform(0.15, 0.95 if not floor else 0.75)
    val = rng.uniform(0.28, 0.98 if not floor else 0.85)
    return colorsys.hsv_to_rgb(hue, sat, val)


def rand_room_white(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.0, 0.12 if floor else 0.08)
    val = rng.uniform(0.72, 0.92) if floor else rng.uniform(0.88, 0.99)
    return colorsys.hsv_to_rgb(hue, sat, val)


def rand_scene_color(rng: random.Random, floor: bool = False) -> tuple[float, float, float]:
    if rng.random() < 0.10:
        return rand_room_white(rng, floor=floor)
    return rand_broad_color(rng, floor=floor)


def rand_neutral_robot(rng: random.Random) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.18), rng.uniform(0.55, 0.98))


def perturb_color(rng: random.Random, base, hue_j=0.06, sat_j=0.15, val_j=0.18) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*base)
    h = (h + rng.uniform(-hue_j, hue_j)) % 1.0
    s = max(0.0, min(1.0, s + rng.uniform(-sat_j, sat_j)))
    v = max(0.0, min(1.0, v + rng.uniform(-val_j, val_j)))
    return colorsys.hsv_to_rgb(h, s, v)


def ensure_ply_cache() -> dict[str, Path]:
    PLY_CACHE.mkdir(exist_ok=True)
    cache: dict[str, Path] = {}
    stl_dir = HERE / "assets"
    for stl in stl_dir.glob("*.stl"):
        ply = PLY_CACHE / (stl.stem + ".ply")
        if not ply.exists() or ply.stat().st_mtime < stl.stat().st_mtime:
            mesh = trimesh.load(stl, force="mesh")
            mesh.export(ply)
        cache[stl.stem] = ply
    return cache


_IK_SOLVER: dict | None = None
_IK_FAILED = False


def _get_ik_solver():
    global _IK_SOLVER, _IK_FAILED
    if _IK_SOLVER is not None or _IK_FAILED:
        return _IK_SOLVER
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk
    except Exception as exc:
        print(f"[ik] unavailable: {exc}")
        _IK_FAILED = True
        return None

    urdf = yourdfpy.URDF.load(str(URDF_PATH))
    robot = pk.Robot.from_urdf(urdf)
    tcp_idx = jnp.asarray(robot.links.names.index("link_tcp"))
    JointVar = robot.joint_var_cls

    @jax.jit
    def _solve(pos, wxyz):
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)
        jv = JointVar(0)
        costs = [
            pk.costs.pose_cost(robot, jv, pose, tcp_idx, pos_weight=5.0, ori_weight=1.0),
            pk.costs.limit_cost(robot, jv, weight=100.0),
        ]
        sol = jaxls.LeastSquaresProblem(costs, [jv]).analyze().solve(verbose=False)
        q = sol[jv]
        fk = robot.forward_kinematics(q)
        ach = jaxlie.SE3(fk[tcp_idx])
        err = jnp.linalg.norm(ach.translation() - pos)
        return q, err

    _IK_SOLVER = {"solve": _solve, "jnp": jnp}
    return _IK_SOLVER


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def sample_joints_tight(rng) -> list[float]:
    return [rng.uniform(lo, hi) for lo, hi in ARM_TIGHT_LIMITS_DEG]


def sample_joints_full(rng) -> list[float]:
    return [rng.uniform(lo, hi) for lo, hi in ARM_PHYSICAL_LIMITS_DEG]


def sample_joints_ik(rng) -> list[float] | None:
    solver = _get_ik_solver()
    if solver is None:
        return None
    jnp = solver["jnp"]
    tx = rng.uniform(-0.55, 0.55)
    ty = rng.uniform(-0.55, 0.55)
    tz = rng.uniform(0.10, 0.85)
    for _ in range(8):
        v = np.array([rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)])
        n = float(np.linalg.norm(v))
        if n > 1e-6 and (v[2] / n) < 0.3:
            d = v / n
            break
    else:
        d = np.array([0.0, 0.0, -1.0])
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, d)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(d, x_axis)
    roll = rng.uniform(-math.pi, math.pi)
    c, s = math.cos(roll), math.sin(roll)
    R = np.stack([c * x_axis + s * y_axis, -s * x_axis + c * y_axis, d], axis=1)
    wxyz = _quat_wxyz_from_R(R)
    q, err = solver["solve"](
        jnp.asarray([tx, ty, tz], dtype=jnp.float32),
        jnp.asarray(wxyz, dtype=jnp.float32),
    )
    if float(err) > 0.02:
        return None
    return [math.degrees(float(a)) for a in np.asarray(q)[:7]]


def sample_arm_angles(rng) -> tuple[list[float], str]:
    r = rng.random()
    if r < 0.45:
        return sample_joints_tight(rng), "tight"
    if r < 0.90:
        a = sample_joints_ik(rng)
        if a is not None:
            return a, "ik"
        return sample_joints_tight(rng), "tight_ik_fallback"
    return sample_joints_full(rng), "full"


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    # Mitsuba perspective looks along +Z, +Y up in camera space.
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)  # already unit
    M = np.eye(4)
    # Columns: camera-x, camera-y, camera-z, translation (camera->world)
    M[:3, 0] = right
    M[:3, 1] = cam_up
    M[:3, 2] = forward
    M[:3, 3] = eye
    return M


def rpy_offset(rpy_deg) -> np.ndarray:
    roll, pitch, yaw = [math.radians(a) for a in rpy_deg]
    # Apply in camera local frame: roll about z, pitch about x, yaw about y
    cz, sz = math.cos(roll), math.sin(roll)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Rz @ Rx @ Ry
    M = np.eye(4)
    M[:3, :3] = R
    return M


def camera_to_world(eye, target, rpy_deg) -> np.ndarray:
    return look_at_matrix(eye, target) @ rpy_offset(rpy_deg)


def project_point(world_to_cam: np.ndarray, pt_world: np.ndarray, fx, fy, cx, cy, w, h) -> dict:
    p = world_to_cam @ np.array([pt_world[0], pt_world[1], pt_world[2], 1.0])
    depth = float(p[2])  # mitsuba +Z forward
    cam_xyz = [float(p[0]), float(p[1]), float(p[2])]
    if depth <= 1e-6:
        return {"camera_xyz": cam_xyz, "pixel_xy": None, "visible": False}
    # Mitsuba perspective: camera local is right-handed with +Y up, +Z forward,
    # which makes +X point to the camera's LEFT in the rendered image.
    x_img = cx - fx * (p[0] / depth)
    y_img = cy - fy * (p[1] / depth)
    visible = bool(0.0 <= x_img < w and 0.0 <= y_img < h)
    return {"camera_xyz": cam_xyz, "pixel_xy": [float(x_img), float(y_img)], "visible": visible}


def joint_world_position(urdf: yourdfpy.URDF, joint_name: str) -> np.ndarray:
    # Joint frame origin == parent link transform * joint.origin
    joint = urdf.joint_map[joint_name]
    parent_xf = urdf.get_transform(joint.parent)
    joint_xf = parent_xf @ joint.origin
    return joint_xf[:3, 3]


def link_world_position(urdf: yourdfpy.URDF, link_name: str) -> np.ndarray:
    return urdf.get_transform(link_name)[:3, 3]


def robot_world_keypoints(urdf: yourdfpy.URDF) -> list[tuple[str, np.ndarray]]:
    out = []
    for name, joint_name, link_name in KEYPOINT_JOINTS:
        if joint_name is not None:
            out.append((name, joint_world_position(urdf, joint_name)))
        else:
            out.append((name, link_world_position(urdf, link_name)))
    return out


def robot_bbox(urdf: yourdfpy.URDF, mesh_cache: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    mn = np.array([np.inf, np.inf, np.inf])
    mx = -mn
    for link_name in robot_visual_links(urdf):
        link_xf = urdf.get_transform(link_name)
        stem = _link_mesh_stem(urdf, link_name)
        if stem is None or stem not in mesh_cache:
            continue
        pts = _mesh_bounds(mesh_cache[stem])  # (2,3) in mesh local frame
        corners = np.array([[x, y, z] for x in pts[:, 0] for y in pts[:, 1] for z in pts[:, 2]])
        corners_h = np.hstack([corners, np.ones((8, 1))])
        world = (link_xf @ corners_h.T).T[:, :3]
        mn = np.minimum(mn, world.min(axis=0))
        mx = np.maximum(mx, world.max(axis=0))
    return mn, mx


def _link_mesh_stem(urdf: yourdfpy.URDF, link_name: str) -> str | None:
    link = urdf.link_map[link_name]
    for vis in link.visuals:
        if vis.geometry and vis.geometry.mesh and vis.geometry.mesh.filename:
            return Path(vis.geometry.mesh.filename).stem
    return None


def segment_distance(a0, a1, b0, b1) -> float:
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    den = a * c - b * b
    eps = 1e-8
    if den < eps:
        s = 0.0
        t = max(0.0, min(1.0, e / c if c > eps else 0.0))
    else:
        s = max(0.0, min(1.0, (b * e - c * d) / den))
        t = max(0.0, min(1.0, (a * e - b * d) / den))
    pa = a0 + s * u
    pb = b0 + t * v
    return float(np.linalg.norm(pa - pb))


def point_segment_distance(p, a, b) -> float:
    ab = b - a
    den = float(ab @ ab)
    if den <= 1e-8:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / den))
    return float(np.linalg.norm(p - (a + t * ab)))


def robot_segments(points: list[tuple[str, np.ndarray]]):
    by = {n: p for n, p in points}
    edges = [
        ("base", "joint1"),
        ("joint1", "joint3"),
        ("joint3", "joint4"),
        ("joint4", "joint5"),
        ("joint5", "joint7"),
        ("joint7", "tcp"),
    ]
    return [(a, b, by[a], by[b]) for a, b in edges if a in by and b in by]


def robot_self_collision(points) -> bool:
    segs = robot_segments(points)
    if len(segs) < 4:
        return False
    pairs = {
        ("base", "joint1", "joint4", "joint5"),
        ("base", "joint1", "joint5", "joint7"),
        ("base", "joint1", "joint7", "tcp"),
        ("joint1", "joint3", "joint5", "joint7"),
        ("joint1", "joint3", "joint7", "tcp"),
        ("joint3", "joint4", "joint7", "tcp"),
    }
    for i, (a0n, a1n, a0, a1) in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b0n, b1n, b0, b1 = segs[j]
            if (a0n, a1n, b0n, b1n) not in pairs:
                continue
            if segment_distance(a0, a1, b0, b1) < 0.02:
                return True
    return False


def robot_penetrates_room(mn, mx) -> bool:
    return (
        mn[0] < -ROOM_X + ROOM_MARGIN
        or mx[0] > ROOM_X - ROOM_MARGIN
        or mn[1] < ROOM_Y_NEG + ROOM_MARGIN
        or mx[1] > ROOM_Y_POS - ROOM_MARGIN
        or mx[2] > ROOM_Z_MAX - ROOM_MARGIN
    )


def camera_collides(eye, bbox_mn, bbox_mx, points) -> bool:
    eye_v = np.asarray(eye)
    if (
        bbox_mn[0] - 0.12 <= eye[0] <= bbox_mx[0] + 0.12
        and bbox_mn[1] - 0.12 <= eye[1] <= bbox_mx[1] + 0.12
        and bbox_mn[2] - 0.12 <= eye[2] <= bbox_mx[2] + 0.12
    ):
        return True
    for _, p in points:
        if np.linalg.norm(eye_v - p) < 0.22:
            return True
    for _, _, a, b in robot_segments(points):
        if point_segment_distance(eye_v, a, b) < 0.16:
            return True
    return False


def camera_radius_bounds_m(radius_min_in: float, radius_max_in: float) -> tuple[float, float]:
    if radius_min_in <= 0.0 or radius_max_in <= 0.0:
        raise ValueError(f"Camera radius bounds must be positive inches, got {radius_min_in}, {radius_max_in}.")
    if radius_min_in >= radius_max_in:
        raise ValueError(f"Camera radius min must be < max (inches), got {radius_min_in} >= {radius_max_in}.")
    return radius_min_in * INCH_TO_METER, radius_max_in * INCH_TO_METER


def build_scene_dict(
    urdf: yourdfpy.URDF,
    mesh_cache: dict[str, Path],
    rng: random.Random,
    view_index: int,
    num_views: int,
    image_width: int,
    image_height: int,
    focal_length: float,
    radius_min_m: float,
    radius_max_m: float,
    *,
    max_depth: int = 8,
    force_all_emitters: bool = False,
) -> tuple[dict, dict, tuple]:
    robot_bb_mn, robot_bb_mx = robot_bbox(urdf, mesh_cache)
    wall_color = rand_scene_color(rng)
    if rng.random() < 0.5:
        robot_color = rand_neutral_robot(rng)
    else:
        robot_color = perturb_color(rng, wall_color)
    ground_color = rand_scene_color(rng, floor=True)
    backdrop_color = rand_scene_color(rng)

    dome_on = rng.random() < 0.7
    dome_intensity = rng.uniform(0.5, 3.0) if dome_on else 0.0
    dome_color = rand_scene_color(rng)
    sun_intensity = rng.uniform(3.5, 26.0)
    sun_color = rand_scene_color(rng)
    sun_rot_deg = (rng.uniform(220.0, 330.0), rng.uniform(-25.0, 25.0), 0.0)
    fill_intensity = rng.uniform(0.0, 12.0)
    fill_color = rand_scene_color(rng)

    # Camera sampling: bounded sphere around the robot base, biased to view_index azimuth.
    points = robot_world_keypoints(urdf)
    by_name = dict(points)
    if "base" not in by_name:
        raise RuntimeError("Missing 'base' keypoint for camera sampling.")
    _, lookat_point = rng.choice(points)
    sphere_point = by_name["base"]
    radius = rng.uniform(radius_min_m, radius_max_m)
    sin_el = rng.uniform(math.sin(math.radians(-25.0)), math.sin(math.radians(75.0)))
    cos_el = math.sqrt(max(0.0, 1.0 - sin_el * sin_el))
    azimuth = (360.0 * view_index / num_views) + rng.uniform(-55.0, 55.0)
    theta = math.radians(azimuth)
    eye = (
        float(sphere_point[0]) + radius * cos_el * math.cos(theta),
        float(sphere_point[1]) + radius * cos_el * math.sin(theta),
        float(sphere_point[2]) + radius * sin_el,
    )
    target = (float(lookat_point[0]), float(lookat_point[1]), float(lookat_point[2]))
    rpy_deg = (
        rng.uniform(-40.0, 40.0),
        rng.uniform(-24.0, 24.0),
        rng.uniform(-24.0, 24.0),
    )

    cam_to_world = camera_to_world(eye, target, rpy_deg)
    fov_x = math.degrees(2.0 * math.atan(CAMERA_APERTURE / (2.0 * focal_length)))

    robot_bsdf = {
        "type": "roughplastic",
        "diffuse_reflectance": {"type": "rgb", "value": list(robot_color)},
        "alpha": rng.uniform(0.12, 0.55),
    }
    ground_bsdf = {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": list(ground_color)},
    }
    backdrop_bsdf = {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": list(backdrop_color)},
    }
    def _wall_bsdf():
        return {"type": "diffuse", "reflectance": {"type": "rgb", "value": list(wall_color)}}

    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": max_depth},
        "sensor": {
            "type": "perspective",
            "fov": fov_x,
            "fov_axis": "x",
            "near_clip": 1e-3,
            "far_clip": 1e4,
            "to_world": mi.ScalarTransform4f(cam_to_world.tolist()),
            "film": {
                "type": "hdrfilm",
                "width": image_width,
                "height": image_height,
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": 32},
        },
    }

    if dome_intensity > 0 or force_all_emitters:
        scene["dome"] = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [
                    dome_color[0] * dome_intensity,
                    dome_color[1] * dome_intensity,
                    dome_color[2] * dome_intensity,
                ],
            },
        }

    # Directional lights: direction = transform of (0,0,-1) by sun rotation (ZYX-ish).
    def dir_from_rpy(rpy):
        R = rpy_offset(rpy)[:3, :3]
        return (R @ np.array([0.0, 0.0, -1.0])).tolist()

    scene["sun"] = {
        "type": "directional",
        "direction": dir_from_rpy(sun_rot_deg),
        "irradiance": {
            "type": "rgb",
            "value": [sun_color[0] * sun_intensity, sun_color[1] * sun_intensity, sun_color[2] * sun_intensity],
        },
    }
    if fill_intensity > 0:
        fill_rot = (rng.uniform(30.0, 60.0), rng.uniform(20.0, 50.0), 0.0)
        scene["fill"] = {
            "type": "directional",
            "direction": dir_from_rpy(fill_rot),
            "irradiance": {
                "type": "rgb",
                "value": [
                    fill_color[0] * fill_intensity,
                    fill_color[1] * fill_intensity,
                    fill_color[2] * fill_intensity,
                ],
            },
        }
    elif force_all_emitters:
        scene["fill"] = {
            "type": "directional",
            "direction": dir_from_rpy((45.0, 35.0, 0.0)),
            "irradiance": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
        }

    # Room as 5 rectangles. Mitsuba rectangle is the [-1,1]^2 plane in XY at Z=0, normal +Z.
    def rect_tf(center, extent_u, extent_v, normal) -> np.ndarray:
        # Build a TRS that maps the unit rect onto the desired wall.
        n = np.asarray(normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        # Choose u axis orthogonal to n
        helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(helper, n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        M = np.eye(4)
        M[:3, 0] = u * extent_u
        M[:3, 1] = v * extent_v
        M[:3, 2] = n
        M[:3, 3] = center
        return M

    # Open-top room so dome/directional lights can enter.
    walls = [
        ("ground", (0.0, 0.0, ROOM_Z_MIN), ROOM_X, ROOM_Y_POS, (0, 0, 1), ground_bsdf),
        (
            "backdrop",
            (0.0, ROOM_Y_NEG, (ROOM_Z_MIN + ROOM_Z_MAX) / 2),
            ROOM_X,
            (ROOM_Z_MAX - ROOM_Z_MIN) / 2,
            (0, 1, 0),
            backdrop_bsdf,
        ),
        (
            "left_wall",
            (-ROOM_X, 0.0, (ROOM_Z_MIN + ROOM_Z_MAX) / 2),
            ROOM_Y_POS,
            (ROOM_Z_MAX - ROOM_Z_MIN) / 2,
            (1, 0, 0),
            _wall_bsdf(),
        ),
        (
            "right_wall",
            (ROOM_X, 0.0, (ROOM_Z_MIN + ROOM_Z_MAX) / 2),
            ROOM_Y_POS,
            (ROOM_Z_MAX - ROOM_Z_MIN) / 2,
            (-1, 0, 0),
            _wall_bsdf(),
        ),
        (
            "front_wall",
            (0.0, ROOM_Y_POS, (ROOM_Z_MIN + ROOM_Z_MAX) / 2),
            ROOM_X,
            (ROOM_Z_MAX - ROOM_Z_MIN) / 2,
            (0, -1, 0),
            _wall_bsdf(),
        ),
    ]
    for name, center, eu, ev, normal, bsdf in walls:
        scene[name] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f(rect_tf(center, eu, ev, normal).tolist()),
            "bsdf": bsdf,
        }

    for link_name in robot_visual_links(urdf):
        stem = _link_mesh_stem(urdf, link_name)
        if stem is None or stem not in mesh_cache:
            continue
        # Combine link world transform with visual origin
        link = urdf.link_map[link_name]
        visual_origin = link.visuals[0].origin if link.visuals else np.eye(4)
        mesh_to_world = urdf.get_transform(link_name) @ visual_origin
        scene[f"robot_{link_name}"] = {
            "type": "ply",
            "filename": str(mesh_cache[stem]),
            "to_world": mi.ScalarTransform4f(mesh_to_world.tolist()),
            "bsdf": robot_bsdf,
        }

    meta = {
        "eye": eye,
        "target": target,
        "rpy_deg": rpy_deg,
        "focal_length": focal_length,
        "camera_radius_m": float(radius),
        "camera_radius_bounds_m": [float(radius_min_m), float(radius_max_m)],
        "camera_anchor": "base",
        "cam_to_world": cam_to_world,
        "fov_x": fov_x,
        "robot_bbox_mn": robot_bb_mn,
        "robot_bbox_mx": robot_bb_mx,
        "keypoints_world": points,
    }
    return scene, meta, (eye, target, rpy_deg)


@dataclass
class PersistentScene:
    scene: object  # mi.Scene
    params: object  # mi.SceneParameters
    link_names: list[str]
    side_wall_key: str | None  # traversal key for the merged left/right/front wall shape
    link_local_verts: dict   # link_name -> (N,3) float32 vertices in local mesh space
    link_local_normals: dict  # link_name -> (N,3) float32 normals in local mesh space


def _scene_dict_to_param_updates(scene_dict: dict, ps: "PersistentScene") -> dict:
    """Extract a flat {traverse_key: value} dict from a scene_dict for in-place updates."""
    p: dict = {}

    s = scene_dict["sensor"]
    p["sensor.to_world"] = s["to_world"]
    p["sensor.x_fov"] = s["fov"]

    p["dome.radiance.value"] = (
        scene_dict["dome"]["radiance"]["value"] if "dome" in scene_dict else [0.0, 0.0, 0.0]
    )

    # Directional emitter directions are baked at load time and not traversable;
    # only irradiance (color × intensity) is updated per frame.
    p["sun.irradiance.value"] = scene_dict["sun"]["irradiance"]["value"]

    if "fill" in scene_dict:
        p["fill.irradiance.value"] = scene_dict["fill"]["irradiance"]["value"]
    else:
        p["fill.irradiance.value"] = [0.0, 0.0, 0.0]

    for wall in ("ground", "backdrop"):
        p[f"{wall}.bsdf.reflectance.value"] = scene_dict[wall]["bsdf"]["reflectance"]["value"]

    # Mitsuba merges left/right/front walls (same wall_color → same BSDF) into one compound
    # shape with an auto-generated key; update that single shape with the shared wall color.
    if ps.side_wall_key is not None:
        p[f"{ps.side_wall_key}.bsdf.reflectance.value"] = (
            scene_dict["left_wall"]["bsdf"]["reflectance"]["value"]
        )

    # PLY shapes bake to_world into vertex_positions at load time; update by retransforming
    # the stored local-space vertices with the new per-frame joint transform.
    for link_name, local_verts in ps.link_local_verts.items():
        key = f"robot_{link_name}"
        if key not in scene_dict:
            continue
        T = np.array(scene_dict[key]["to_world"].matrix, dtype=np.float64)
        N = len(local_verts)
        verts_h = np.column_stack([local_verts.astype(np.float64), np.ones(N)])
        new_verts = (T @ verts_h.T).T[:, :3].astype(np.float32)
        p[f"{key}.vertex_positions"] = new_verts.flatten()
        local_normals = ps.link_local_normals.get(link_name)
        if local_normals is not None:
            R = T[:3, :3]
            new_normals = (R @ local_normals.astype(np.float64).T).T
            norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
            new_normals = new_normals / np.maximum(norms, 1e-8)
            p[f"{key}.vertex_normals"] = new_normals.astype(np.float32).flatten()
        bsdf = scene_dict[key]["bsdf"]
        p[f"{key}.bsdf.diffuse_reflectance.value"] = bsdf["diffuse_reflectance"]["value"]
        p[f"{key}.bsdf.alpha"] = bsdf["alpha"]

    return p


def build_persistent_scene(urdf: yourdfpy.URDF, mesh_cache: dict[str, Path], cfg) -> PersistentScene:
    """Load the Mitsuba scene once; subsequent renders update params in-place."""
    rng = random.Random(0)
    radius_min_m = cfg.camera_radius_min_in * INCH_TO_METER
    radius_max_m = cfg.camera_radius_max_in * INCH_TO_METER
    mid_focal = (cfg.fx_min + cfg.fx_max) / 2.0 * CAMERA_APERTURE / cfg.image_width
    scene_dict, _, _ = build_scene_dict(
        urdf, mesh_cache, rng, 0, max(cfg.num_views, 1),
        cfg.image_width, cfg.image_height, mid_focal,
        radius_min_m, radius_max_m,
        force_all_emitters=True,
    )
    link_names = robot_visual_links(urdf)

    # All robot links share the same robot_bsdf dict in build_scene_dict, so Mitsuba merges
    # them into one compound shape.  Apply a negligible per-link alpha offset so each link
    # gets a distinct BSDF instance and stays traversable separately.
    for i, link_name in enumerate(link_names):
        key = f"robot_{link_name}"
        if key not in scene_dict:
            continue
        orig = scene_dict[key]["bsdf"]
        scene_dict[key] = {
            **scene_dict[key],
            "bsdf": {**orig, "alpha": orig["alpha"] + (i + 1) * 1e-6},
        }

    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    # The three side walls share wall_color so Mitsuba merges them into one compound shape
    # with an auto-generated key.  Detect it: the only non-robot reflectance key that isn't
    # ground or backdrop.
    side_wall_key = None
    for k in sorted(params.keys()):
        if k.endswith(".bsdf.reflectance.value"):
            base = k[: -len(".bsdf.reflectance.value")]
            if base not in ("ground", "backdrop"):
                side_wall_key = base
                break

    # Load local-space vertices/normals from PLY files so we can retransform them per frame.
    link_local_verts: dict = {}
    link_local_normals: dict = {}
    for link_name in link_names:
        stem = _link_mesh_stem(urdf, link_name)
        if stem is None or stem not in mesh_cache:
            continue
        mesh = trimesh.load(str(mesh_cache[stem]), process=False, force="mesh")
        link_local_verts[link_name] = np.asarray(mesh.vertices, dtype=np.float32)
        link_local_normals[link_name] = np.asarray(mesh.vertex_normals, dtype=np.float32)

    return PersistentScene(
        scene=scene,
        params=params,
        link_names=link_names,
        side_wall_key=side_wall_key,
        link_local_verts=link_local_verts,
        link_local_normals=link_local_normals,
    )


def intrinsics_from_focal(focal_length: float, width: int, height: int) -> dict:
    fx = width * focal_length / CAMERA_APERTURE
    fy = height * focal_length / (CAMERA_APERTURE * height / width)
    # fy computed so vertical aperture matches height scaling
    fy = width * focal_length / CAMERA_APERTURE  # same px/mm as fx since film is square-pixel
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def mitsuba_cam_to_usd(cam_to_world_mi: np.ndarray) -> np.ndarray:
    """Convert Mitsuba cam-to-world (right=+X, up=+Y, forward=+Z) to USD-style
    cam-to-world (image-right=+X, up=+Y, forward=-Z) used by mask_renderer.

    Negate columns 0 and 2: flips the X axis (Mitsuba's X is image-left) and
    the Z axis (Mitsuba's Z is forward, USD's Z is backward).
    """
    M = cam_to_world_mi.copy()
    M[:, 0] = -M[:, 0]
    M[:, 2] = -M[:, 2]
    return M


_INST_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 60, 60],
        [60, 200, 60],
        [60, 100, 220],
        [230, 200, 40],
        [200, 60, 200],
        [60, 200, 220],
        [240, 140, 40],
        [140, 80, 220],
        [180, 60, 80],
        [80, 220, 140],
        [180, 220, 80],
        [220, 80, 160],
        [60, 140, 80],
        [220, 160, 100],
        [120, 160, 220],
        [160, 220, 220],
    ],
    dtype=np.uint8,
)


def colorize_instance(inst: np.ndarray) -> np.ndarray:
    n = _INST_COLORS.shape[0]
    return _INST_COLORS[np.clip(inst, 0, n - 1)]


def render_view(
    urdf: yourdfpy.URDF,
    mesh_cache: dict[str, Path],
    rng: random.Random,
    view_index: int,
    cfg,
    out_dir: Path,
    radius_min_m: float,
    radius_max_m: float,
    bg_sampler: CocoBackgroundSampler | None = None,
    mask_renderer: MaskRenderer | None = None,
    denoiser: object | None = None,
    persistent_scene: PersistentScene | None = None,
):
    reject = {"room": 0, "self": 0, "camera": 0, "pose": 0, "dark": 0}
    image_path = out_dir / f"view_{view_index}.png"
    json_path = out_dir / f"view_{view_index}.json"
    for _ in range(40):
        fx_sample = rng.uniform(cfg.fx_min, cfg.fx_max)
        focal_length = fx_sample * CAMERA_APERTURE / cfg.image_width
        angles_deg, sample_mode = sample_arm_angles(rng)
        grip_angle = rng.uniform(0.0, 28.0)

        joint_cfg = {
            k: np.float64(v)
            for k, v in {
                "joint1": math.radians(angles_deg[0]),
                "joint2": math.radians(angles_deg[1]),
                "joint3": math.radians(angles_deg[2]),
                "joint4": math.radians(angles_deg[3]),
                "joint5": math.radians(angles_deg[4]),
                "joint6": math.radians(angles_deg[5]),
                "joint7": math.radians(angles_deg[6]),
                "drive_joint": math.radians(grip_angle),
                "left_finger_joint": math.radians(-grip_angle),
                "left_inner_knuckle_joint": math.radians(grip_angle),
                "right_outer_knuckle_joint": math.radians(-grip_angle),
                "right_finger_joint": math.radians(grip_angle),
                "right_inner_knuckle_joint": math.radians(-grip_angle),
            }.items()
        }
        urdf.update_cfg(joint_cfg)

        mn, mx = robot_bbox(urdf, mesh_cache)
        if robot_penetrates_room(mn, mx):
            reject["room"] += 1
            continue
        points = robot_world_keypoints(urdf)
        if robot_self_collision(points):
            reject["self"] += 1
            continue

        scene_dict, meta, (eye, target, rpy_deg) = build_scene_dict(
            urdf,
            mesh_cache,
            rng,
            view_index,
            cfg.total_views,
            cfg.image_width,
            cfg.image_height,
            focal_length,
            radius_min_m,
            radius_max_m,
            max_depth=cfg.max_depth,
        )

        if camera_collides(eye, meta["robot_bbox_mn"], meta["robot_bbox_mx"], points):
            reject["camera"] += 1
            continue

        # Visibility check
        intr = intrinsics_from_focal(focal_length, cfg.image_width, cfg.image_height)
        world_to_cam = np.linalg.inv(meta["cam_to_world"])
        visible = 0
        for _, p in points:
            proj = project_point(
                world_to_cam, p, intr["fx"], intr["fy"], intr["cx"], intr["cy"], cfg.image_width, cfg.image_height
            )
            if proj["visible"]:
                visible += 1
        if visible < 1:
            reject["pose"] += 1
            continue

        if persistent_scene is not None:
            updates = _scene_dict_to_param_updates(scene_dict, persistent_scene)
            for k, v in updates.items():
                persistent_scene.params[k] = v
            persistent_scene.params.update()
            img = mi.render(persistent_scene.scene, spp=cfg.spp)
        else:
            scene = mi.load_dict(scene_dict)
            img = mi.render(scene, spp=cfg.spp)
        if cfg.denoise and denoiser is not None:
            img = denoiser(img)
        bmp = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        rgb_arr = np.asarray(bmp).copy()
        mean = float(rgb_arr.mean())
        if mean < 8.0:
            reject["dark"] += 1
            continue

        # Always render & save the mask. Train-time COCO augmentation reads it
        # from the arec record; bypassing this branch would silently strip it.
        mask_intr = MaskIntrinsics(
            fx=intr["fx"],
            fy=intr["fy"],
            cx=intr["cx"],
            cy=intr["cy"],
            width=cfg.image_width,
            height=cfg.image_height,
        )
        usd_cam_to_world = mitsuba_cam_to_usd(meta["cam_to_world"])
        world_to_cam_row = np.linalg.inv(usd_cam_to_world).T
        mask_joints = {k: float(v) for k, v in joint_cfg.items()}
        binary, instance = mask_renderer.render(mask_joints, world_to_cam_row, mask_intr)
        Image.fromarray(binary).save(out_dir / f"view_{view_index}_mask.png")
        Image.fromarray(colorize_instance(instance)).save(out_dir / f"view_{view_index}_inst.png")

        if bg_sampler is not None:
            bg, bg_name = bg_sampler.sample(cfg.image_width, cfg.image_height)
            comp = composite_bg(rgb_arr, binary, bg, erode_px=1)
            sidecar_bg = bg_name
            Image.fromarray(comp).save(image_path)
            kp_img = Image.fromarray(comp).convert("RGB")
        else:
            sidecar_bg = None
            bmp.write(str(image_path))
            kp_img = Image.open(image_path).convert("RGB")

        # Sidecar
        keypoints = []
        pixel_by_name: dict[str, tuple[float, float]] = {}
        kp_draw = ImageDraw.Draw(kp_img)
        for name, pt in points:
            proj = project_point(
                world_to_cam, pt, intr["fx"], intr["fy"], intr["cx"], intr["cy"], cfg.image_width, cfg.image_height
            )
            keypoints.append({"name": name, "world_xyz": [float(pt[0]), float(pt[1]), float(pt[2])], **proj})
            if proj["pixel_xy"]:
                pixel_by_name[name] = tuple(proj["pixel_xy"])
        skeleton_edges = [
            ("base", "joint1"),
            ("joint1", "joint2"),
            ("joint2", "joint3"),
            ("joint3", "joint4"),
            ("joint4", "joint5"),
            ("joint5", "joint6"),
            ("joint6", "joint7"),
            ("joint7", "eef"),
            ("eef", "tcp"),
        ]
        for a, b in skeleton_edges:
            if a in pixel_by_name and b in pixel_by_name:
                kp_draw.line([pixel_by_name[a], pixel_by_name[b]], fill="yellow", width=2)
        for kp in keypoints:
            if kp["pixel_xy"] and kp["visible"]:
                x, y = kp["pixel_xy"]
                kp_draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="red", outline="white", width=1)
        kp_img.save(out_dir / f"view_{view_index}_kp.png")

        sidecar = {
            "image_file": image_path.name,
            "scene_mode": "asset",
            "camera_target_world": list(target),
            "camera_rpy_deg": list(rpy_deg),
            "camera_focal_length": float(focal_length),
            "camera_distance_from_base_m": float(meta["camera_radius_m"]),
            "reject_counts": reject,
            "visible_keypoint_count": visible,
            "background": sidecar_bg,
            "joints": {
                **{f"joint{i + 1}": angles_deg[i] for i in range(7)},
                "gripper_angle": grip_angle,
                "sample_mode": sample_mode,
            },
            "keypoints": keypoints,
            "camera": {
                "intrinsics": {
                    **intr,
                    "K": [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
                },
                "extrinsics": {
                    "camera_to_world": meta["cam_to_world"].tolist(),
                    "world_to_camera": world_to_cam.tolist(),
                },
                "sampling": {
                    "anchor": meta["camera_anchor"],
                    "radius_m": float(meta["camera_radius_m"]),
                    "radius_bounds_m": list(meta["camera_radius_bounds_m"]),
                    "radius_bounds_in": [float(cfg.camera_radius_min_in), float(cfg.camera_radius_max_in)],
                },
            },
        }
        json_path.write_text(json.dumps(sidecar, indent=2))
        return reject
    raise RuntimeError(f"view_{view_index}: exhausted attempts {reject}")


@dataclass
class Config:
    """Render xarm7 views with Mitsuba 3 (path tracing)."""

    seed: int = 20260413
    num_views: int = 50
    # Index of the first view this run writes (loop iterates
    # `range(start_index, start_index + num_views)`). Used to split a single
    # output directory across multiple machines without filename collisions —
    # e.g. 4 GPUs each doing 25k of a 100k dataset:
    #   GPU 0: --start-index 0     --num-views 25000 --total-views 100000 --seed S
    #   GPU 1: --start-index 25000 --num-views 25000 --total-views 100000 --seed S+1
    #   GPU 2: --start-index 50000 --num-views 25000 --total-views 100000 --seed S+2
    #   GPU 3: --start-index 75000 --num-views 25000 --total-views 100000 --seed S+3
    start_index: int = 0
    # Total views across all workers — used only for azimuth distribution so
    # cameras spread evenly around 360°. Defaults to num_views (single-worker).
    total_views: int = 0
    output_dir: Path = Path("renders/mitsuba_views")
    # Default ON to make multi-machine writes safe — wiping the shared dir
    # from one process would clobber the other's output. Pass --clean to
    # explicitly request the wipe (e.g. for single-machine reruns).
    no_clean: bool = True
    image_width: int = 640
    image_height: int = 480
    fx_min: float = 450.0
    fx_max: float = 550.0
    camera_radius_min_in: float = 14.0  # min camera radius from robot base in inches
    camera_radius_max_in: float = 58.0  # max camera radius from robot base in inches
    spp: int = 128
    max_depth: int = 4  # path-tracer max bounce depth; 4–5 is enough with denoising
    denoise: bool = True  # apply OptiX denoiser post-render
    coco_dir: Path = Path("~/bafl/coco/train2014").expanduser()  # COCO bg images
    composite: bool = True  # composite robot over a random COCO image
    progress_file: Path | None = None  # if set, write "done total" here after each view


def main(cfg: Config) -> None:
    if cfg.total_views == 0:
        cfg.total_views = cfg.num_views
    radius_min_m, radius_max_m = camera_radius_bounds_m(cfg.camera_radius_min_in, cfg.camera_radius_max_in)
    out = cfg.output_dir.resolve()
    if out.exists() and not cfg.no_clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Writing to {out} (seed={cfg.seed}, "
        f"views={cfg.start_index}..{cfg.start_index + cfg.num_views - 1}, "
        f"spp={cfg.spp}, "
        f"camera_radius={cfg.camera_radius_min_in:.1f}-{cfg.camera_radius_max_in:.1f}in/"
        f"{radius_min_m:.3f}-{radius_max_m:.3f}m)"
    )

    mesh_cache = ensure_ply_cache()
    print(f"Prepared {len(mesh_cache)} PLY meshes in {PLY_CACHE}")

    urdf = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False, build_scene_graph=True)

    # Mask is always rendered (downstream training uses it for runtime COCO aug).
    # Compositing is the optional step that bakes a COCO background into the RGB.
    mask_renderer = MaskRenderer(URDF_PATH, include_gripper=True)
    bg_sampler: CocoBackgroundSampler | None = None
    if cfg.composite:
        bg_sampler = CocoBackgroundSampler(cfg.coco_dir, rng=random.Random(cfg.seed + 1))
        print(f"Compositing over {len(bg_sampler)} COCO images from {cfg.coco_dir}")

    denoiser = None
    if cfg.denoise:
        denoiser = mi.OptixDenoiser(input_size=[cfg.image_width, cfg.image_height])
        print("OptiX denoiser initialised.")

    ps = build_persistent_scene(urdf, mesh_cache, cfg)
    print("Persistent Mitsuba scene loaded.")

    rng = random.Random(cfg.seed)
    totals = {"room": 0, "self": 0, "camera": 0, "pose": 0, "dark": 0}
    view_range = range(cfg.start_index, cfg.start_index + cfg.num_views)
    done = 0
    use_bar = cfg.progress_file is None
    with tqdm(total=cfg.num_views, unit="view", dynamic_ncols=True, disable=not use_bar) as bar:
        for i in view_range:
            r = render_view(
                urdf,
                mesh_cache,
                rng,
                i,
                cfg,
                out,
                radius_min_m,
                radius_max_m,
                bg_sampler,
                mask_renderer,
                denoiser,
                ps,
            )
            for k, v in r.items():
                totals[k] += v
            done += 1
            if cfg.progress_file is not None:
                tmp = cfg.progress_file.with_suffix(".tmp")
                tmp.write_text(f"{done} {cfg.num_views}\n")
                tmp.replace(cfg.progress_file)
            bar.update(1)
            bar.set_postfix(rejects=sum(totals.values()))
    print(f"Wrote {cfg.num_views} renders to {out}")
    print(f"Total rejects: {totals}")


if __name__ == "__main__":
    main(tyro.cli(Config))
