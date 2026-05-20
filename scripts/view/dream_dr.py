from __future__ import annotations

from dataclasses import dataclass
import json
import logging
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
    result_dir: Path = Path("mytmp")  # Directory written by scripts/dream_dr.py
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "dream_dr_view"
    camera_name: str = "dream_dr"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    dt: float = 0.25  # Delay between frames when replaying
    loop: bool = False
    pose: str = "dr"  # Pose to visualize: dr, initial, or both
    static_images: bool = False  # Log all image artifacts as static data

    def __post_init__(self) -> None:
        self.result_dir = self.result_dir.expanduser().resolve()
        self.urdf = self.urdf.expanduser().resolve()
        if self.rrd_path is not None:
            self.rrd_path = self.rrd_path.expanduser().resolve()
        if self.pose not in {"dr", "initial", "both"}:
            raise ValueError("pose must be one of: dr, initial, both")


@dataclass
class Record:
    stem: str
    image: np.ndarray | None
    joints: np.ndarray | None
    intrinsics: np.ndarray | None


def read_metadata(result_dir: Path) -> dict:
    path = result_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata object in {path}")
    return data


def record_paths(metadata: dict) -> list[Path]:
    paths = []
    for raw in metadata.get("records", []):
        path = Path(raw).expanduser()
        if path.exists():
            paths.append(path.resolve())
        else:
            logging.warning("Metadata record does not exist: %s", path)
    return paths


def load_records(cfg: Config, metadata: dict) -> list[Record]:
    records = [load_record(path) for path in record_paths(metadata)]
    if records:
        return records

    image_dir = cfg.result_dir / "images"
    image_paths = sorted(image_dir.glob("*_image.png"))
    if not image_paths:
        raise FileNotFoundError(
            f"No metadata records or image artifacts found. Expected {cfg.result_dir / 'metadata.json'} "
            f"or files under {image_dir}"
        )
    return [load_image_record(path) for path in image_paths]


def load_record(path: Path) -> Record:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        image = np.asarray(data["image_model" if "image_model" in keys else "image"], dtype=np.uint8)
        while image.ndim > 3 and image.shape[0] == 1:
            image = image[0]
        return Record(
            stem=path.stem,
            image=image,
            joints=np.asarray(data["joints"], dtype=np.float32).reshape(-1)[:7] if "joints" in keys else None,
            intrinsics=np.asarray(data["K"], dtype=np.float32) if "K" in keys else None,
        )


def load_image_record(path: Path) -> Record:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image {path}")
    stem = path.name.removesuffix("_image.png")
    return Record(stem=stem, image=image, joints=None, intrinsics=None)


def read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image {path}")
    return image


def load_pose(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    pose = np.asarray(np.load(path), dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    if pose.shape == (4, 4):
        return pose
    if pose.ndim == 3 and pose.shape[1:] == (4, 4):
        return pose
    raise ValueError(f"Expected {path} to contain shape (4, 4) or (n, 4, 4), got {pose.shape}")


def pose_for_step(pose: np.ndarray | None, step: int) -> np.ndarray | None:
    if pose is None:
        return None
    if pose.ndim == 2:
        return pose
    return pose[min(step, len(pose) - 1)]


def send_blueprint(scene: RerunScene, camera_name: str) -> None:
    cam_root = f"{scene.world_path}/cam/{camera_name}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["+ /robot/**", f"+ {scene.world_path}/**"]),
            rrb.Grid(
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/image"]),
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/mask"]),
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/renders"]),
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/overlays"]),
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/render_overlays"]),
                rrb.Spatial2DView(origin=cam_root, contents=["+ $origin/difference"]),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def log_image(scene: RerunScene, camera_name: str, name: str, image: np.ndarray, *, static: bool) -> None:
    path = f"{scene.world_path}/cam/{camera_name}/{name}"
    rr.log(path, rr.CoordinateFrame(frame=f"{scene.world_path}/cam/{camera_name}/image_plane"), static=True)
    rr.log(path, rr.Image(image, color_model="BGR").compress(jpeg_quality=85), static=static)


def log_pose(scene: RerunScene, entity_suffix: str, pose: np.ndarray, k: np.ndarray, width: int, height: int) -> None:
    entity_path = f"{scene.world_path}/cam/{entity_suffix}"
    image_plane_frame = f"{entity_path}/image_plane"
    rot = np.asarray(pose[:3, :3], dtype=np.float64)
    quat_xyzw = R.from_matrix(rot).as_quat()
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=pose[:3, 3],
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
            focal_length=[float(k[0, 0]), float(k[1, 1])],
            principal_point=[float(k[0, 2]), float(k[1, 2])],
            camera_xyz=rr.ViewCoordinates.RDF,
            parent_frame=entity_path,
            child_frame=image_plane_frame,
            image_plane_distance=0.1,
        ),
        static=False,
    )


def joint_values(scene: RerunScene, joints: np.ndarray | None) -> dict[str, float]:
    if joints is None:
        return {}
    names = [name for name in sorted(scene.joint_map) if name.startswith("joint")]
    return {name: float(value) for name, value in zip(names, joints.reshape(-1), strict=False)}


def fallback_intrinsics(record: Record) -> np.ndarray:
    if record.intrinsics is not None:
        return record.intrinsics
    if record.image is None:
        raise ValueError(f"Record {record.stem} has no image for fallback intrinsics")
    h, w = record.image.shape[:2]
    return np.array([[515.0, 0.0, w / 2.0], [0.0, 515.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def artifact_paths(result_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "image": result_dir / "images" / f"{stem}_image.png",
        "mask": result_dir / "masks" / f"{stem}_mask.png",
        "renders": result_dir / "renders" / f"{stem}_renders.png",
        "overlays": result_dir / "overlays" / f"{stem}_overlays.png",
        "render_overlays": result_dir / "render_overlays" / f"{stem}_render_overlays.png",
        "difference": result_dir / "difference" / f"{stem}_difference.png",
    }


def replay(
    cfg: Config, scene: RerunScene, records: list[Record], initial: np.ndarray | None, dr: np.ndarray | None
) -> None:
    step = 0
    while True:
        for record in records:
            rr.set_time("step", sequence=step)
            scene.log_joints(joint_values(scene, record.joints), step=step)

            for name, path in artifact_paths(cfg.result_dir, record.stem).items():
                image = read_image(path)
                if image is None and name == "image":
                    image = record.image
                if image is not None:
                    log_image(scene, cfg.camera_name, name, image, static=cfg.static_images)

            image = record.image
            if image is not None:
                h, w = image.shape[:2]
                k = fallback_intrinsics(record)
                if cfg.pose in {"initial", "both"} and (pose := pose_for_step(initial, step)) is not None:
                    log_pose(scene, f"{cfg.camera_name}_initial", pose, k, w, h)
                if cfg.pose in {"dr", "both"} and (pose := pose_for_step(dr, step)) is not None:
                    log_pose(scene, f"{cfg.camera_name}_dr", pose, k, w, h)

            step += 1
            if cfg.dt > 0.0 and cfg.rrd_path is None:
                time.sleep(cfg.dt)
        if not cfg.loop:
            return


def main(cfg: Config) -> None:
    metadata = read_metadata(cfg.result_dir)
    records = load_records(cfg, metadata)
    initial = load_pose(cfg.result_dir / "HT_initial.npy")
    dr = load_pose(cfg.result_dir / "HT_dr.npy")
    if cfg.pose in {"dr", "both"} and dr is None:
        logging.warning("No HT_dr.npy found in %s; DR camera pose will not be shown", cfg.result_dir)

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
    replay(cfg, scene, records, initial, dr)
    logging.info("Logged %d records from %s", len(records), cfg.result_dir)


if __name__ == "__main__":
    main(tyro.cli(Config))
