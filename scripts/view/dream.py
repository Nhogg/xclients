from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R
import tyro
from webpolicy.client import Client

from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    da_host: str | None = None
    da_port: int | None = None
    call_dream: bool = False  # Call Dream for masks/raster/w2c; disabled by default for precomputed HT viewing
    camera: str | int = 0  # OpenCV camera index to poll from
    camera_name: str = "dream"  # Rerun camera entity name
    image_size: int = 200  # Resize frames to a square image before sending to Dream
    fx: float = 515.0  # Focal length in pixels along x for the payload K matrix
    fy: float = 515.0  # Focal length in pixels along y for the payload K matrix
    q: list[float] = field(default_factory=lambda: [0.0] * 7)  # Joint vector sent to Dream and logged to Rerun
    deg2rad: bool = False  # Convert cfg.q from degrees to radians before sending
    ht_path: Path | None = None  # Optional .npy/.npz file containing a precomputed HT extrinsics matrix
    ht_convention: Literal["cv-w2c", "camera-file"] = "cv-w2c"  # Convention for --ht-path matrices
    record_index: int = 0  # Saved record index from ht_path/metadata.json or sorted sibling rr_good records
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "dream_view"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    limit: int | None = None
    history: int = 100  # Number of recent camera centers to keep as 3D points
    max_camera_distance: float = 3.0  # Skip Dream poses whose camera center is farther than this many meters
    depth_stride: int = 4  # Subsample factor when converting depth maps to 3D points
    max_depth: float = 2.0  # Maximum depth in meters to include in the point cloud
    depth_scale: float = 1.0  # Scale factor applied to DA depth before unprojection
    depth_history: int = 10  # Number of recent DA point clouds to keep, fading older clouds by opacity
    show: bool = False  # Also show the local OpenCV window

    def __post_init__(self) -> None:
        self.urdf = self.urdf.expanduser().resolve()
        if self.ht_path is not None:
            self.ht_path = Path(self.ht_path).expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = Path(self.rrd_path).expanduser().resolve()
        if (self.da_host is None) != (self.da_port is None):
            raise ValueError("Pass both da_host and da_port, or neither.")


def scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float32).copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def draw_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask with shape (h, w) or (1, h, w), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def draw_output_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None

    arr = np.asarray(image)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected raster image with shape (h, w), (h, w, c), or batched variants, got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def draw_depth_image(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None

    arr = np.asarray(depth)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected depth with shape (h, w) or singleton-batched variants, got {arr.shape}")

    arr = arr.astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)

    lo = float(arr[valid].min())
    hi = float(arr[valid].max())
    norm = np.zeros_like(arr, dtype=np.float32)
    if hi > lo:
        norm[valid] = (arr[valid] - lo) / (hi - lo)
    depth_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


@dataclass
class RecordFrame:
    path: Path
    frame: np.ndarray
    intrinsics: np.ndarray
    joints: np.ndarray
    joints_are_degrees: bool


def read_metadata(path: Path) -> dict:
    meta = path / "metadata.json"
    if not meta.exists():
        return {}
    with meta.open() as f:
        import json

        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {meta}")
    return data


def resolve_record_path(raw: object, base: Path) -> Path | None:
    if not isinstance(raw, str):
        return None
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base / path)
    candidates.append(base.parent / path.parent.name / path.name)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def record_paths_from_ht_dir(path: Path) -> list[Path]:
    meta = read_metadata(path)
    records = []
    for raw in meta.get("records", []):
        record = resolve_record_path(raw, path)
        if record is not None:
            records.append(record)
    if records:
        return records

    rr_good = path.parent / "rr_good"
    if rr_good.exists():
        return sorted(rr_good.glob("*.npz"))
    return []


def squeeze_record_image(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected saved record image with shape (h, w, 3), got {image.shape}")
    return image.astype(np.uint8, copy=False)


def load_record_frame(path: Path) -> RecordFrame:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    image_raw = arrays["image_model"] if "image_model" in arrays else arrays["image"]
    image = squeeze_record_image(image_raw)
    joints = np.asarray(arrays["joints"], dtype=np.float32).reshape(-1)[:7]
    joints_are_degrees = not bool(np.asarray(arrays.get("joints_is_radian", True)))
    return RecordFrame(
        path=path,
        frame=image,
        intrinsics=np.asarray(arrays["K"], dtype=np.float32),
        joints=joints,
        joints_are_degrees=joints_are_degrees,
    )


def load_config_record_frame(cfg: Config) -> RecordFrame | None:
    if cfg.ht_path is None:
        return None
    record_base = cfg.ht_path if cfg.ht_path.is_dir() else cfg.ht_path.parent
    paths = record_paths_from_ht_dir(record_base)
    if not paths:
        return None
    if cfg.record_index < 0 or cfg.record_index >= len(paths):
        raise IndexError(f"record_index={cfg.record_index} outside saved record range 0..{len(paths) - 1}")
    return load_record_frame(paths[cfg.record_index])


def ensure_bgr_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    return arr


def log_aux_image(scene: RerunScene, camera_name: str, path: str, image: np.ndarray) -> None:
    rr.log(path, rr.CoordinateFrame(frame=f"{scene.world_path}/cam/{camera_name}/image_plane"), static=True)
    rr.log(path, rr.Image(image, color_model="BGR").compress(jpeg_quality=75), static=False)


def send_dream_blueprint(scene: RerunScene, camera_name: str) -> None:
    cam_root = f"{scene.world_path}/cam/{camera_name}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=[
                    "+ /robot/**",
                    f"+ {scene.world_path}/**",
                ],
            ),
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/image", "+ /robot/**"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/mask"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/raster"]),
                    rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/depth"]),
                ]
            ),
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def coerce_w2c_pose(w2c: np.ndarray | None) -> np.ndarray | None:
    if w2c is None:
        return None

    pose = np.asarray(w2c, dtype=np.float64)
    while pose.ndim > 2 and pose.shape[0] == 1:
        pose = pose[0]

    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])

    if pose.shape != (4, 4):
        raise ValueError(f"Expected w2c with shape (4, 4), got {pose.shape}")

    return pose


def resolve_ht_path(path: Path) -> Path:
    if path.is_dir():
        for name in ("HT_dr.npy", "HT_rr.npz", "HT_rr.npy", "HT_initial.npy", "HT_dream.npy"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No HT file found in {path}; expected HT_dr.npy, HT_rr.npz, or similar")
    return path


def load_ht(path: Path) -> tuple[np.ndarray, Path]:
    source = resolve_ht_path(path)
    data = np.load(source)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            if "HT" in data.files:
                ht = data["HT"]
            elif "HT_dr" in data.files:
                ht = data["HT_dr"]
            elif "HT_rr" in data.files:
                ht = data["HT_rr"]
            elif len(data.files) == 1:
                ht = data[data.files[0]]
            else:
                raise ValueError(f"Expected {source} to contain an HT array; found {data.files}")
        finally:
            data.close()
    else:
        ht = data

    pose = coerce_w2c_pose(ht)
    if pose is None:
        raise ValueError(f"Expected {source} to contain an HT pose.")
    if not np.isfinite(pose).all():
        raise ValueError(f"HT pose from {source} has non-finite values: {pose!r}")
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError(f"Expected HT pose bottom row to be [0, 0, 0, 1], got {pose[3]}")
    return pose, source


def ht_to_rerun_extrinsics(ht: np.ndarray) -> np.ndarray:
    return ht


def camera_file_ht_to_rerun_extrinsics(ht: np.ndarray) -> np.ndarray:
    from xclients.core import tf as xctf

    return xctf.RDF2FLU @ np.linalg.inv(ht)


def dream_camera_calibration(k: np.ndarray, w2c: np.ndarray, width: int, height: int) -> dict[str, np.ndarray | int]:
    return {
        "intrinsics": np.asarray(k, dtype=np.float32),
        "extrinsics": np.asarray(w2c, dtype=np.float32),
        "width": int(width),
        "height": int(height),
    }


def log_dynamic_camera(scene: RerunScene, camera_name: str, calibration: dict[str, np.ndarray | int]) -> None:
    entity_path = f"{scene.world_path}/cam/{camera_name}"
    image_plane_frame = f"{entity_path}/image_plane"
    extrinsic = np.asarray(calibration["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
    width = int(calibration["width"])
    height = int(calibration["height"])

    rot = extrinsic[:3, :3]
    if not np.isfinite(rot).all():
        raise ValueError(f"Camera rotation has non-finite values: {rot!r}")
    if not np.isfinite(extrinsic[:3, 3]).all():
        raise ValueError(f"Camera translation has non-finite values: {extrinsic[:3, 3]!r}")
    quat_xyzw = R.from_matrix(rot).as_quat()
    t = extrinsic[:3, 3].astype(np.float32)

    rr.log(
        entity_path,
        rr.Transform3D(
            translation=t,
            quaternion=quat_xyzw,
            parent_frame=str(scene.world_path),
            child_frame=entity_path,
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=False,
    )
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[float(intrinsics[0, 0]), float(intrinsics[1, 1])],
            principal_point=[float(intrinsics[0, 2]), float(intrinsics[1, 2])],
            camera_xyz=rr.ViewCoordinates.RDF,
            parent_frame=entity_path,
            child_frame=image_plane_frame,
            image_plane_distance=0.1,
            color=[255, 128, 0],
            line_width=0.002,
        ),
        static=False,
    )


def joint_values_from_q(scene: RerunScene, q: np.ndarray) -> dict[str, float]:
    if q.ndim != 1:
        q = q.reshape(-1)

    arm_joint_names = [name for name in sorted(scene.joint_map) if name.startswith("joint")]
    if len(q) != len(arm_joint_names):
        raise ValueError(f"Expected {len(arm_joint_names)} joint values for {arm_joint_names}, got {len(q)}")

    return {name: float(value) for name, value in zip(arm_joint_names, q, strict=True)}


def camera_world_position(extrinsic: np.ndarray) -> np.ndarray:
    rot = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    t = np.asarray(extrinsic[:3, 3], dtype=np.float64)
    return (-rot.T @ t).astype(np.float32)


def history_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 4), dtype=np.uint8)
    if count == 1:
        return np.array([[255, 128, 0, 255]], dtype=np.uint8)

    oldest = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    newest = np.array([255.0, 128.0, 0.0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    rgb = oldest * (1.0 - t) + newest * t
    alpha = (64.0 + 191.0 * t).reshape(-1, 1)
    colors = np.concatenate([rgb, alpha], axis=1)
    return np.round(colors).astype(np.uint8)


def add_alpha(colors: np.ndarray, alpha: float) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.uint8)
    if rgb.ndim != 2 or rgb.shape[1] not in (3, 4):
        raise ValueError(f"Expected colors with shape (n, 3) or (n, 4), got {rgb.shape}")
    if rgb.shape[1] == 4:
        rgb = rgb[:, :3]
    alpha_col = np.full((len(rgb), 1), round(np.clip(alpha, 0.0, 255.0)), dtype=np.uint8)
    return np.concatenate([rgb, alpha_col], axis=1)


def depth_history_cloud(history: deque[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.uint8)

    count = len(history)
    points = []
    colors = []
    for i, (cloud_points, cloud_colors) in enumerate(history):
        alpha = 64.0 if count == 1 else 32.0 + 223.0 * (i / (count - 1))
        points.append(cloud_points)
        colors.append(add_alpha(cloud_colors, alpha))
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    rot = np.asarray(transform[:3, :3], dtype=np.float32)
    t = np.asarray(transform[:3, 3], dtype=np.float32)
    return (points @ rot.T + t).astype(np.float32)


def first_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def unproject_depth_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    bgr_image: np.ndarray,
    *,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected depth map with shape (h, w), got {z.shape}")

    k = np.asarray(intrinsics, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (3, 3), got {k.shape}")

    if bgr_image.shape[:2] != z.shape:
        bgr_image = cv2.resize(bgr_image, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_LINEAR)

    step = max(int(stride), 1)
    ys, xs = np.mgrid[0 : z.shape[0] : step, 0 : z.shape[1] : step]
    zs = z[::step, ::step]

    valid = np.isfinite(zs) & (zs > 0.0)
    if max_depth > 0.0:
        valid &= zs <= float(max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points = np.stack([x, y, zs], axis=1)

    colors_bgr = bgr_image[::step, ::step][valid]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)
    return points, colors_rgb


def main(cfg: Config) -> None:
    client = Client(cfg.host, cfg.port) if cfg.call_dream else None
    da_client = Client(cfg.da_host, cfg.da_port) if cfg.da_host is not None and cfg.da_port is not None else None
    record_frame = load_config_record_frame(cfg)
    ht_source = None
    ht_payload = None
    if cfg.ht_path is not None:
        ht_payload, ht_source = load_ht(cfg.ht_path)
    ht_rerun_pose = None
    if ht_payload is not None:
        ht_rerun_pose = (
            ht_to_rerun_extrinsics(ht_payload)
            if cfg.ht_convention == "cv-w2c"
            else camera_file_ht_to_rerun_extrinsics(ht_payload)
        )
    if ht_payload is not None:
        logging.info("Loaded precomputed HT extrinsics from %s using %s convention", ht_source, cfg.ht_convention)
    if client is None and ht_rerun_pose is None:
        raise ValueError("Pass --ht-path when using --no-call-dream; otherwise no camera extrinsics are available.")
    if da_client is None:
        logging.warning("No DA client configured; depth point cloud logging is disabled.")

    cap = None
    if record_frame is None:
        cap = cv2.VideoCapture(cfg.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {cfg.camera}")
    else:
        logging.info("Rendering saved record %s; live camera is disabled", record_frame.path)

    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )
    mask_path = f"{scene.world_path}/cam/{cfg.camera_name}/mask"
    raster_path = f"{scene.world_path}/cam/{cfg.camera_name}/raster"
    depth_path = f"{scene.world_path}/cam/{cfg.camera_name}/depth"
    depth_points_path = f"{scene.world_path}/scene/{cfg.camera_name}_depth_points"
    scene.set_cameras([cfg.camera_name])
    send_dream_blueprint(scene, cfg.camera_name)

    q_cfg = np.asarray(record_frame.joints if record_frame is not None else cfg.q, dtype=np.float32)
    q_degrees = bool(record_frame.joints_are_degrees) if record_frame is not None else bool(cfg.deg2rad)
    q_payload = np.deg2rad(q_cfg) if q_degrees else q_cfg

    start = time.monotonic()
    step = 0
    camera_history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.history)))
    depth_history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max(1, int(cfg.depth_history)))
    if client is not None:
        logging.info("Polling camera %s and sending frames to Dream at %s:%s", cfg.camera, cfg.host, cfg.port)
    elif record_frame is not None:
        logging.info("Rendering saved record with fixed HT extrinsics and no Dream calls")
    else:
        logging.info("Polling camera %s with fixed HT extrinsics and no Dream calls", cfg.camera)
    run_limit = 1 if record_frame is not None and cfg.limit is None else cfg.limit
    while run_limit is None or step < run_limit:
        if record_frame is not None:
            frame = record_frame.frame
            k_orig = record_frame.intrinsics
        else:
            assert cap is not None
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.camera)
                continue
            h, w = frame.shape[:2]
            k_orig = np.array(
                [
                    [cfg.fx, 0.0, w / 2.0],
                    [0.0, cfg.fy, h / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        h, w = frame.shape[:2]
        frame_model = cv2.resize(frame, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        k_model = scale_intrinsics(k_orig, cfg.image_size / float(w), cfg.image_size / float(h))

        rr.set_time("step", sequence=step)
        rr.set_time("time", duration=time.monotonic() - start)
        scene.log_camera_images({cfg.camera_name: frame})
        scene.log_joints(joint_values_from_q(scene, q_cfg), step=step, degrees=q_degrees)

        out = None
        if client is not None:
            payload = {
                "image": frame_model,
                "type": "image",
                "q": q_payload,
                "K": k_model,
            }
            if ht_payload is not None:
                payload["HT"] = ht_payload.astype(np.float32)
            out = client.step(payload)

        da_out = None
        if da_client is not None:
            da_payload = {
                "image": [frame],
                "intrinsics": np.array([k_orig], dtype=np.float32),
            }
            try:
                da_out = da_client.step(da_payload)
            except Exception as exc:
                logging.warning("Skipping Depth Anything point cloud at step %d: %s", step, exc)
                da_client = None

        mask = draw_mask(out.get("mask") if out else None)
        raster_raw = out.get("raster_image") if out else None
        if raster_raw is None and out is not None:
            raster_raw = out.get("rast_image")
        raster = draw_output_image(raster_raw)
        depth_raw = first_array(da_out.get("depth") if da_out else None)
        if depth_raw is not None:
            depth_raw = np.asarray(depth_raw, dtype=np.float32) * float(cfg.depth_scale)
            valid_depth = np.isfinite(depth_raw) & (depth_raw > 0.0)
            if step == 0 and np.any(valid_depth):
                logging.info(
                    "DA depth stats after depth_scale=%.4f: min=%.4f median=%.4f max=%.4f is_metric=%s scale_factor=%s",
                    cfg.depth_scale,
                    float(np.nanmin(depth_raw[valid_depth])),
                    float(np.nanmedian(depth_raw[valid_depth])),
                    float(np.nanmax(depth_raw[valid_depth])),
                    da_out.get("is_metric") if da_out else None,
                    da_out.get("scale_factor") if da_out else None,
                )
        depth_vis = draw_depth_image(depth_raw)
        if mask is not None:
            log_aux_image(scene, cfg.camera_name, mask_path, ensure_bgr_image(mask))
        if raster is not None:
            log_aux_image(scene, cfg.camera_name, raster_path, ensure_bgr_image(raster))
        if depth_vis is not None:
            log_aux_image(scene, cfg.camera_name, depth_path, depth_vis)

        dream_pose = coerce_w2c_pose(out.get("w2c") if out else None)
        pose = ht_rerun_pose if ht_rerun_pose is not None else dream_pose
        calibration = None
        if pose is not None:
            calibration = dream_camera_calibration(k_orig, pose, w, h)
            calibration["extrinsics"] = pose
        if depth_raw is not None and calibration is not None:
            depth_intr = first_array(da_out.get("intrinsics") if da_out else None)
            if depth_intr is None:
                dh, dw = depth_raw.shape
                depth_intr = scale_intrinsics(k_orig, dw / float(w), dh / float(h))
            try:
                points_cam, colors_rgb = unproject_depth_points(
                    depth_raw,
                    np.asarray(depth_intr, dtype=np.float32),
                    frame,
                    stride=cfg.depth_stride,
                    max_depth=cfg.max_depth,
                )
                camera_to_world = np.linalg.inv(np.asarray(calibration["extrinsics"], dtype=np.float64))
                points_world = transform_points(camera_to_world, points_cam)
                depth_history.append((points_world, colors_rgb))
                history_points, history_colors_rgba = depth_history_cloud(depth_history)
                scene.log_points3d(
                    history_points,
                    colors=history_colors_rgba,
                    radii=0.005,
                    path=depth_points_path,
                    parent_frame=str(scene.world_path),
                )
            except ValueError as exc:
                logging.warning("Skipping DA depth point cloud at step %d: %s", step, exc)
        elif depth_raw is not None:
            logging.warning("Skipping DA depth point cloud at step %d because Dream pose is unavailable", step)

        if pose is not None:
            try:
                assert calibration is not None

                camera_position = camera_world_position(np.asarray(calibration["extrinsics"], dtype=np.float64))
                if not np.isfinite(camera_position).all():
                    raise ValueError(f"Camera center has non-finite values: {camera_position!r}")
                camera_distance = float(np.linalg.norm(camera_position))
                if camera_distance > cfg.max_camera_distance:
                    raise ValueError(
                        f"Camera center distance {camera_distance:.3f} m exceeds max_camera_distance={cfg.max_camera_distance:.3f} m"
                    )

                log_dynamic_camera(scene, cfg.camera_name, calibration)
                camera_history.append(camera_position)
                history_points = np.stack(camera_history, axis=0)
                scene.log_points3d(
                    history_points,
                    colors=history_colors(len(history_points)),
                    radii=0.01,
                    path=f"{scene.world_path}/scene/{cfg.camera_name}_history",
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logging.warning("Skipping invalid Dream camera pose at step %d: %s", step, exc)
        else:
            logging.warning("No precomputed HT or valid Dream w2c pose available at step %d", step)

        if cfg.show:
            cv2.imshow(f"Dream {cfg.camera}", frame)
            if mask is not None:
                cv2.imshow(f"Dream {cfg.camera} Mask", mask)
            if raster is not None:
                cv2.imshow(f"Dream {cfg.camera} Raster", raster)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        step += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Config))
