from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

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
    rr_good: Path = Path("~/rr_good")  # Source records with image, joints, K, and DREAM w2c
    rr_gripper: Path = Path("~/rr_gripper")  # DR output directory with HT_dr.npy and computed masks/renders
    index: int = 0  # Record index from rr_good to visualize
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "dream_rr_debug"
    camera_name: str = "rr_gripper"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    depth_stride: int = 4
    max_depth: float = 2.0
    render_robot: bool = True  # Compute a fresh robot render using the first rr_good pose and rr_gripper HT_dr

    def __post_init__(self) -> None:
        self.rr_good = self.rr_good.expanduser().resolve()
        self.rr_gripper = self.rr_gripper.expanduser().resolve()
        self.urdf = self.urdf.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = self.rrd_path.expanduser().resolve()


@dataclass
class Record:
    stem: str
    path: Path
    image: np.ndarray
    image_model: np.ndarray
    joints: np.ndarray
    intrinsics: np.ndarray
    dream_w2c: np.ndarray | None
    arrays: dict[str, np.ndarray]


def ensure_plugin_src() -> None:
    plugin_src = Path(__file__).resolve().parents[2] / "plugins/server_roboreg/src"
    if str(plugin_src) not in sys.path:
        sys.path.insert(0, str(plugin_src))


def read_metadata(path: Path) -> dict:
    meta = path / "metadata.json"
    if not meta.exists():
        return {}
    with meta.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {meta}")
    return data


def load_pose(path: Path) -> np.ndarray:
    pose = np.asarray(np.load(path), dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    if pose.shape != (4, 4):
        raise ValueError(f"Expected {path} to contain a single (4, 4) pose, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose {path} contains non-finite values")
    return pose


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
    image = squeeze_image(arrays["image"])
    image_model = squeeze_image(arrays.get("image_model", image))
    return Record(
        stem=path.stem,
        path=path,
        image=image,
        image_model=image_model,
        joints=np.asarray(arrays["joints"], dtype=np.float32).reshape(-1)[:7],
        intrinsics=np.asarray(arrays["K"], dtype=np.float32),
        dream_w2c=np.asarray(arrays["w2c"], dtype=np.float32) if "w2c" in arrays else None,
        arrays=arrays,
    )


def record_paths(cfg: Config) -> list[Path]:
    meta = read_metadata(cfg.rr_gripper)
    paths = [Path(raw).expanduser().resolve() for raw in meta.get("records", [])]
    paths = [path for path in paths if path.exists()]
    if not paths:
        paths = sorted(cfg.rr_good.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No rr_good .npz records found under {cfg.rr_good}")
    return paths


def read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image {path}")
    return image


def log_image(scene: RerunScene, camera_name: str, name: str, image: np.ndarray, *, static: bool = False) -> None:
    path = f"{scene.world_path}/cam/{camera_name}/{name}"
    rr.log(path, rr.CoordinateFrame(frame=f"{scene.world_path}/cam/{camera_name}/image_plane"), static=True)
    rr.log(path, rr.Image(image, color_model="BGR").compress(jpeg_quality=85), static=static)


def send_blueprint(scene: RerunScene, camera_name: str) -> None:
    cam = f"{scene.world_path}/cam/{camera_name}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["+ /robot/**", f"+ {scene.world_path}/**"]),
            rrb.Grid(
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/image_model", "+ /robot/**"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/computed_render"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/rr_gripper_render"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/rr_gripper_mask"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/rr_good_raster"]),
                rrb.Spatial2DView(origin=cam, contents=["+ $origin/difference"]),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def log_camera_pose(
    scene: RerunScene, camera_name: str, w2c: np.ndarray, k: np.ndarray, width: int, height: int
) -> None:
    entity_path = f"{scene.world_path}/cam/{camera_name}"
    image_plane_frame = f"{entity_path}/image_plane"
    rot = np.asarray(w2c[:3, :3], dtype=np.float64)
    quat_xyzw = R.from_matrix(rot).as_quat()
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=w2c[:3, 3],
            quaternion=quat_xyzw,
            parent_frame=str(scene.world_path),
            child_frame=entity_path,
            relation=rr.TransformRelation.ChildFromParent,
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
            color=[255, 128, 0],
            line_width=0.002,
        ),
        static=True,
    )


def joint_values(scene: RerunScene, joints: np.ndarray) -> dict[str, float]:
    names = [name for name in sorted(scene.joint_map) if name.startswith("joint")]
    if len(joints) != len(names):
        raise ValueError(f"Expected {len(names)} joint values for {names}, got {len(joints)}")
    return {name: float(value) for name, value in zip(names, joints, strict=True)}


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    mask2 = np.asarray(mask)
    if mask2.ndim == 3:
        mask2 = mask2[..., 0]
    if mask2.shape[:2] != image.shape[:2]:
        mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    alpha = (mask2 > 0).astype(np.float32)[..., None] * 0.5
    color_image = np.zeros_like(image)
    color_image[...] = np.asarray(color, dtype=np.uint8)
    return np.clip(image.astype(np.float32) * (1.0 - alpha) + color_image.astype(np.float32) * alpha, 0, 255).astype(
        np.uint8
    )


def render_robot(record: Record, final_w2c: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    ensure_plugin_src()
    from server_roboreg.common import HydraConfig
    from server_roboreg.dr import render_cv_w2c
    from server_roboreg.render import Renderer, RendererConfig
    import torch

    h, w = record.image_model.shape[:2]
    hcfg = HydraConfig(
        urdf=cfg.urdf,
        root_link_name="link_base",
        end_link_name="link_eef",
        collision_meshes=False,
    )
    renderer = Renderer(
        hcfg,
        RendererConfig(batch_size=1),
        height=h,
        width=w,
        intr=record.intrinsics,
    )
    joints = torch.tensor(record.joints[None], dtype=torch.float32, device=renderer.device)
    w2c = torch.tensor(final_w2c[None], dtype=torch.float32, device=renderer.device)
    intr = torch.tensor(record.intrinsics, dtype=torch.float32, device=renderer.device)
    render = render_cv_w2c(renderer, joints, w2c, intr, h, w)[0, ..., 0].detach().cpu().numpy()
    render_u8 = np.clip(render * 255.0, 0, 255).astype(np.uint8)

    renderer.scene.robot.configure(joints)
    vertices = renderer.scene.robot.configured_vertices[0].detach().cpu().numpy().astype(np.float32)
    return render_u8, vertices


def unproject_depth(
    depth: np.ndarray, k: np.ndarray, image: np.ndarray, stride: int, max_depth: float
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype=np.float32).squeeze()
    if z.ndim != 2:
        raise ValueError(f"Expected depth shape (h, w), got {z.shape}")
    if image.shape[:2] != z.shape:
        image = cv2.resize(image, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_LINEAR)

    step = max(int(stride), 1)
    ys, xs = np.mgrid[0 : z.shape[0] : step, 0 : z.shape[1] : step]
    zs = z[::step, ::step]
    valid = np.isfinite(zs) & (zs > 0.0)
    if max_depth > 0:
        valid &= zs <= float(max_depth)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)
    if len(zs) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    points = np.column_stack(
        [
            (xs - float(k[0, 2])) * zs / float(k[0, 0]),
            (ys - float(k[1, 2])) * zs / float(k[1, 1]),
            zs,
        ]
    ).astype(np.float32)
    colors = image[::step, ::step][valid, ::-1].astype(np.uint8)
    return points, colors


def log_optional_observed_cloud(scene: RerunScene, record: Record, final_w2c: np.ndarray, cfg: Config) -> None:
    points = None
    colors = None
    if "points" in record.arrays:
        points = np.asarray(record.arrays["points"], dtype=np.float32).reshape(-1, 3)
        if "colors" in record.arrays:
            colors = np.asarray(record.arrays["colors"], dtype=np.uint8).reshape(-1, 3)
    elif "depth" in record.arrays:
        points_cam, colors = unproject_depth(
            record.arrays["depth"], record.intrinsics, record.image_model, cfg.depth_stride, cfg.max_depth
        )
        points = points_cam @ final_w2c[:3, :3].T + final_w2c[:3, 3]

    if points is None:
        logging.warning("No depth/points in %s; no observed point cloud source is available.", record.path)
        return
    scene.log_points3d(
        points,
        colors=colors,
        radii=0.004,
        path=f"{scene.world_path}/scene/observed_point_cloud",
        parent_frame=str(scene.world_path),
        static=True,
    )


def main(cfg: Config) -> None:
    paths = record_paths(cfg)
    record = load_record(paths[cfg.index])
    final_w2c = load_pose(cfg.rr_gripper / "HT_dr.npy")
    initial_w2c = load_pose(cfg.rr_gripper / "HT_initial.npy")
    h, w = record.image_model.shape[:2]

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
    log_camera_pose(scene, cfg.camera_name, final_w2c, record.intrinsics, w, h)
    log_camera_pose(scene, f"{cfg.camera_name}_initial", initial_w2c, record.intrinsics, w, h)
    if record.dream_w2c is not None:
        log_camera_pose(scene, f"{cfg.camera_name}_dream", record.dream_w2c, record.intrinsics, w, h)

    log_image(scene, cfg.camera_name, "image_model", record.image_model, static=True)
    for name, path in {
        "rr_good_raster": cfg.rr_good / f"{record.stem}_raster.png",
        "rr_good_mask": cfg.rr_good / f"{record.stem}_mask.png",
        "rr_gripper_mask": cfg.rr_gripper / "masks" / f"{record.stem}_mask.png",
        "rr_gripper_render": cfg.rr_gripper / "renders" / f"{record.stem}_renders.png",
        "rr_gripper_overlay": cfg.rr_gripper / "overlays" / f"{record.stem}_overlays.png",
        "render_overlay": cfg.rr_gripper / "render_overlays" / f"{record.stem}_render_overlays.png",
        "difference": cfg.rr_gripper / "difference" / f"{record.stem}_difference.png",
    }.items():
        if (image := read_image(path)) is not None:
            log_image(scene, cfg.camera_name, name, image, static=True)

    if cfg.render_robot:
        try:
            render_u8, vertices = render_robot(record, final_w2c, cfg)
        except ModuleNotFoundError as exc:
            logging.warning("Skipping fresh roboreg render because a dependency is missing: %s", exc)
            render_u8, vertices = None, None
        if render_u8 is not None and vertices is not None:
            computed = overlay_mask(record.image_model, render_u8)
            log_image(scene, cfg.camera_name, "computed_render", computed, static=True)
            scene.log_points3d(
                vertices,
                colors=np.tile(np.array([[255, 128, 0]], dtype=np.uint8), (len(vertices), 1)),
                radii=0.002,
                path=f"{scene.world_path}/scene/robot_vertices_first_pose",
                parent_frame=str(scene.world_path),
                static=True,
            )

    log_optional_observed_cloud(scene, record, final_w2c, cfg)
    if record.dream_w2c is not None:
        logging.info("First rr_good DREAM w2c:\n%s", np.array2string(record.dream_w2c, precision=5))
    logging.info("rr_gripper HT_initial:\n%s", np.array2string(initial_w2c, precision=5))
    logging.info("rr_gripper HT_dr used for final point cloud/camera:\n%s", np.array2string(final_w2c, precision=5))
    logging.info("Visualized %s", record.path)


if __name__ == "__main__":
    main(tyro.cli(Config))
