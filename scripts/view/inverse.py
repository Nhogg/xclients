from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R
import tyro

from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    rr_gripper: Path = Path("~/rr_gripper")  # Directory with HT_dr.npy and metadata.json
    rr_good: Path = Path("~/rr_good")  # Fallback source records if rr_gripper/metadata.json is missing
    index: int = 0
    urdf: Path = Path("xarm7_standalone.urdf")
    w2c_name: str = "HT_dr.npy"
    camera_name: str = "rr_gripper"
    app_id: str = "inverse_view"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    hold_seconds: float = 0.0  # 0 keeps the live Rerun stream open until Ctrl-C
    keypoint_links: tuple[str, ...] = (
        "link_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link_eef",
        "link_tcp",
    )
    overlay_alpha: float = 0.5
    jax_platforms: str = "cpu"

    def __post_init__(self) -> None:
        self.rr_gripper = self.rr_gripper.expanduser().resolve()
        self.rr_good = self.rr_good.expanduser().resolve()
        self.urdf = self.urdf.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = self.rrd_path.expanduser().resolve()


@dataclass
class Record:
    path: Path
    image: np.ndarray
    joints: np.ndarray
    intrinsics: np.ndarray


def read_metadata(path: Path) -> dict:
    meta = path / "metadata.json"
    if not meta.exists():
        return {}
    with meta.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {meta}")
    return data


def record_paths(cfg: Config) -> list[Path]:
    meta = read_metadata(cfg.rr_gripper)
    paths = []
    for raw in meta.get("records", []):
        path = Path(raw).expanduser().resolve()
        if path.exists():
            paths.append(path)
            continue
        sibling = cfg.rr_gripper.parent / path.parent.name / path.name
        if sibling.exists():
            paths.append(sibling)

    if not paths:
        paths = sorted(cfg.rr_good.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No rr_good .npz records found under {cfg.rr_good}")
    return paths


def squeeze_image(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image with shape (h, w, 3), got {image.shape}")
    return image.astype(np.uint8, copy=False)


def load_record(path: Path) -> Record:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    image_raw = arrays["image_model"] if "image_model" in arrays else arrays["image"]
    joints = np.asarray(arrays["joints"], dtype=np.float32).reshape(-1)[:7]
    if not bool(np.asarray(arrays.get("joints_is_radian", True))):
        joints = np.deg2rad(joints).astype(np.float32)

    return Record(
        path=path,
        image=squeeze_image(image_raw),
        joints=joints,
        intrinsics=np.asarray(arrays["K"], dtype=np.float32),
    )


def load_pose(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            if "HT" in data.files:
                pose = data["HT"]
            elif "HT_dr" in data.files:
                pose = data["HT_dr"]
            elif "HT_rr" in data.files:
                pose = data["HT_rr"]
            elif len(data.files) == 1:
                pose = data[data.files[0]]
            else:
                raise ValueError(f"Expected {path} to contain an HT matrix; found {data.files}")
        finally:
            data.close()
    else:
        pose = data

    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    if pose.shape != (4, 4):
        raise ValueError(f"Expected {path} to contain a single (4, 4) pose, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose {path} contains non-finite values")
    return pose


def build_pyroki_q(robot: object, joints: np.ndarray) -> np.ndarray:
    joint_names = list(robot.joints.actuated_names)
    joints = np.asarray(joints, dtype=np.float32).reshape(-1)
    if len(joints) == len(joint_names):
        return joints

    q = np.zeros(len(joint_names), dtype=np.float32)
    for i, value in enumerate(joints[:7], start=1):
        name = f"joint{i}"
        if name in joint_names:
            q[joint_names.index(name)] = float(value)
    return q


def pyroki_keypoints(urdf: Path, joints: np.ndarray, link_names: tuple[str, ...], jax_platforms: str) -> tuple[np.ndarray, list[str]]:
    os.environ.setdefault("JAX_PLATFORMS", jax_platforms)
    try:
        import jax.numpy as jnp
        import jaxlie
        import pyroki as pk
        import yourdfpy
    except Exception as exc:
        raise RuntimeError("Pyroki/JAX dependencies are required for inverse.py") from exc

    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(urdf)))
    q = build_pyroki_q(robot, joints)
    fk = robot.forward_kinematics(jnp.asarray(q))

    available_links = list(robot.links.names)
    points = []
    labels = []
    for name in link_names:
        if name not in available_links:
            logging.warning("Skipping unknown Pyroki link %s", name)
            continue
        pose = jaxlie.SE3(fk[available_links.index(name)])
        points.append(np.asarray(pose.translation(), dtype=np.float32))
        labels.append(name)

    if not points:
        raise ValueError("No requested keypoint_links were present in the Pyroki robot")
    return np.stack(points, axis=0), labels


def joint_values(scene: RerunScene, joints: np.ndarray) -> dict[str, float]:
    names = [name for name in sorted(scene.joint_map) if name.startswith("joint")]
    if len(joints) != len(names):
        raise ValueError(f"Expected {len(names)} joint values for {names}, got {len(joints)}")
    return {name: float(value) for name, value in zip(names, joints, strict=True)}


def project_points(points_world: np.ndarray, w2c: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    uvw = points_cam @ k.T
    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    valid = points_cam[:, 2] > 1e-6
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, valid


def project_vertices(vertices: np.ndarray, w2c: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_cam = vertices @ w2c[:3, :3].T + w2c[:3, 3]
    uvw = points_cam @ k.T
    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    valid = points_cam[:, 2] > 1e-6
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, points_cam[:, 2]


def configured_urdf_meshes(urdf: Path, joints: np.ndarray) -> list:
    from yourdfpy import URDF

    robot = URDF.load(str(urdf))
    joint_cfg = dict.fromkeys(robot.actuated_joint_names, 0.0)
    for i, value in enumerate(np.asarray(joints, dtype=np.float32).reshape(-1)[:7], start=1):
        joint_cfg[f"joint{i}"] = float(value)
    robot.update_cfg(joint_cfg)
    return robot.scene.dump()


def render_urdf_mask(record: Record, w2c: np.ndarray, urdf: Path) -> np.ndarray:
    h, w = record.image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_items = []

    for mesh in configured_urdf_meshes(urdf, record.joints):
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        uv, z = project_vertices(vertices, w2c, record.intrinsics)
        face_z = z[faces].mean(axis=1)
        visible = np.isfinite(uv[faces]).all(axis=(1, 2)) & (z[faces] > 1e-6).all(axis=1)
        for face, depth in zip(faces[visible], face_z[visible], strict=False):
            pts = np.round(uv[face]).astype(np.int32)
            if (pts[:, 0] < 0).all() or (pts[:, 0] >= w).all() or (pts[:, 1] < 0).all() or (pts[:, 1] >= h).all():
                continue
            draw_items.append((float(depth), pts))

    for _, pts in sorted(draw_items, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(mask, pts, 255)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    foreground = mask > 0
    color = np.zeros_like(image)
    color[...] = np.array([0, 220, 80], dtype=np.uint8)
    out = image.copy()
    out[foreground] = np.clip(
        image[foreground].astype(np.float32) * (1.0 - alpha) + color[foreground].astype(np.float32) * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)
    return out


def draw_keypoints(image: np.ndarray, uv: np.ndarray, valid: np.ndarray, labels: list[str]) -> np.ndarray:
    out = image.copy()
    for point, ok, label in zip(uv, valid, labels, strict=True):
        if not ok or not np.isfinite(point).all():
            continue
        xy = tuple(np.round(point).astype(int))
        if xy[0] < 0 or xy[0] >= out.shape[1] or xy[1] < 0 or xy[1] >= out.shape[0]:
            continue
        cv2.circle(out, xy, 4, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(out, label, (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
    return out


def log_image(world_path: Path, camera_name: str, name: str, image: np.ndarray) -> None:
    path = f"{world_path}/cam/{camera_name}/{name}"
    rr.log(path, rr.CoordinateFrame(frame=f"{world_path}/cam/{camera_name}/image_plane"), static=True)
    rr.log(path, rr.Image(image, color_model="BGR").compress(jpeg_quality=90), static=True)


def log_camera(scene: RerunScene, camera_name: str, c2w: np.ndarray, k: np.ndarray, width: int, height: int) -> None:
    entity_path = f"{scene.world_path}/cam/{camera_name}"
    image_plane_frame = f"{entity_path}/image_plane"
    quat_xyzw = R.from_matrix(np.asarray(c2w[:3, :3], dtype=np.float64)).as_quat()
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=c2w[:3, 3],
            quaternion=quat_xyzw,
            parent_frame=str(scene.world_path),
            child_frame=entity_path,
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[float(k[0, 0]), float(k[1, 1])],
            principal_point=[float(k[0, 2]), float(k[1, 2])],
            camera_xyz=rr.ViewCoordinates.RDF,
            parent_frame=entity_path,
            child_frame=image_plane_frame,
            image_plane_distance=0.1,
            color=[255, 0, 255],
            line_width=0.002,
        ),
        static=True,
    )


def send_blueprint(scene: RerunScene, camera_name: str) -> None:
    cam = f"{scene.world_path}/cam/{camera_name}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["+ /robot/**", f"+ {scene.world_path}/**"]),
            rrb.Grid(
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/image"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/urdf_keypoints_overlay"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/urdf_mask"]),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def main(cfg: Config) -> None:
    paths = record_paths(cfg)
    record = load_record(paths[cfg.index])
    w2c_path = cfg.rr_gripper / cfg.w2c_name
    w2c = load_pose(w2c_path)
    c2w = np.linalg.inv(w2c)
    points_world, labels = pyroki_keypoints(cfg.urdf, record.joints, cfg.keypoint_links, cfg.jax_platforms)
    uv, valid = project_points(points_world, w2c, record.intrinsics)
    h, w = record.image.shape[:2]

    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )
    scene.set_cameras([cfg.camera_name])
    send_blueprint(scene, cfg.camera_name)

    rr.set_time("step", sequence=0)
    scene.log_joints(joint_values(scene, record.joints), step=0)
    log_camera(scene, cfg.camera_name, c2w, record.intrinsics, w, h)
    log_image(scene.world_path, cfg.camera_name, "image", record.image)

    mask = render_urdf_mask(record, w2c, cfg.urdf)
    overlay = overlay_mask(record.image, mask, cfg.overlay_alpha)
    overlay = draw_keypoints(overlay, uv, valid, labels)
    log_image(scene.world_path, cfg.camera_name, "urdf_mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    log_image(scene.world_path, cfg.camera_name, "urdf_keypoints_overlay", overlay)

    scene.log_points3d(
        points_world,
        colors=np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(points_world), 1)),
        radii=0.018,
        labels=labels,
        path=f"{scene.world_path}/scene/pyroki_keypoints",
        parent_frame=str(scene.world_path),
        static=True,
    )
    scene.log_points2d(
        cfg.camera_name,
        uv[valid],
        colors=np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (int(valid.sum()), 1)),
        radii=4.0,
        labels=[label for label, ok in zip(labels, valid, strict=True) if ok],
        path=f"{scene.world_path}/cam/{cfg.camera_name}/pyroki_keypoints",
        static=True,
    )

    logging.info("Record: %s", record.path)
    logging.info("World-to-camera: %s", w2c_path)
    logging.info("camera_to_world:\n%s", np.array2string(c2w, precision=5))
    logging.info("Projected %d/%d Pyroki keypoints into the camera image", int(valid.sum()), len(valid))

    if cfg.rrd_path is None:
        if cfg.hold_seconds > 0.0:
            logging.info("Holding Rerun stream open for %.1f seconds", cfg.hold_seconds)
            time.sleep(cfg.hold_seconds)
        else:
            logging.info("Holding Rerun stream open; press Ctrl-C to exit")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logging.info("Exiting on Ctrl-C")


if __name__ == "__main__":
    main(tyro.cli(Config))
