from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

DEFAULT_K = np.array(
    [[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# ROS REP-103 (X=fwd, Y=left, Z=up) -> OpenCV world (X=right, Y=down, Z=fwd)
R_ROS2CV = np.array(
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    dtype=np.float32,
)
FLU2RDF = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
RDF2FLU = np.linalg.inv(FLU2RDF).astype(np.float32)

MANO_JOINT_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class VizConfig:
    data_dir: Path = Path("~/crossformer_data").expanduser()
    extr_dir: Path = Path("/home/nh/crossformer_data/extr/cam")
    start: int = 1
    batch_size: int = 4
    future_steps: int = 6
    sample_index: int = 0
    image_key: str = "low"
    camera_view: str = "low"
    extrinsics_mode: Literal["w2c", "c2w"] = "w2c"
    extrinsics_style: Literal["preprocess", "direct"] = "preprocess"
    camera_axes: Literal["RDF", "DRF"] = "RDF"
    projection_mode: Literal["world_extr", "camera_direct"] = "world_extr"
    ros_to_opencv: bool = True
    spawn: bool = True
    save_rrd: Path | None = None
    train_step: int = 0


def load_camera_ht(view: str, extr_dir: Path) -> np.ndarray:
    path = extr_dir.expanduser() / view / "HT.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing camera extrinsics: {path}")
    return np.load(path)["HT"].astype(np.float32)


def world_to_camera_extrinsics(
    ht: np.ndarray,
    extrinsics_mode: Literal["w2c", "c2w"],
    extrinsics_style: Literal["preprocess", "direct"],
    ros_to_opencv: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h_w2c = ht if extrinsics_mode == "w2c" else np.linalg.inv(ht)
    if extrinsics_style == "preprocess":
        # Matches xclients preprocess.py load_extr convention.
        e = RDF2FLU @ np.linalg.inv(h_w2c)
        return e[:3, :3].astype(np.float32), e[:3, 3].astype(np.float32)

    r = h_w2c[:3, :3]
    t = h_w2c[:3, 3]
    if ros_to_opencv:
        r = r @ R_ROS2CV
    return r.astype(np.float32), t.astype(np.float32)


def _uv_from_cam_xy(x_cam: np.ndarray, y_cam: np.ndarray, z_cam: np.ndarray, k_scaled: np.ndarray) -> np.ndarray:
    u = k_scaled[0, 0] * (x_cam / z_cam) + k_scaled[0, 2]
    v = k_scaled[1, 1] * (y_cam / z_cam) + k_scaled[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def _ndarray_from_serializable(d: dict[str, Any]) -> np.ndarray:
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"])


def _default_unpack(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return _ndarray_from_serializable(obj)
    return obj


def _decode_record(buf: bytes) -> Any:
    try:
        import msgpack
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing 'msgpack'. Install with `uv pip install msgpack` "
            "or run this script from the crossformer uv environment."
        ) from exc
    return msgpack.unpackb(buf, object_hook=_default_unpack, raw=False)


class DecodedArrayRecord:
    def __init__(self, shards: list[Path]) -> None:
        try:
            from array_record.python.array_record_data_source import ArrayRecordDataSource
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing 'array_record'. Install with `uv pip install array-record` "
                "or run this script from the crossformer uv environment."
            ) from exc
        self._ds = ArrayRecordDataSource([str(p) for p in sorted(shards)])

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int):
        return _decode_record(self._ds[index])


def load_test_viz_callback_style_data(cfg: VizConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = sorted(cfg.data_dir.expanduser().glob("*.arrayrecord"))
    if not shards:
        raise FileNotFoundError(f"No .arrayrecord shards found in {cfg.data_dir}")

    records = DecodedArrayRecord(shards)
    n = cfg.batch_size + cfg.future_steps
    if cfg.start < 0 or cfg.start + n > len(records):
        raise ValueError(f"Requested range [{cfg.start}, {cfg.start + n}) exceeds dataset length {len(records)}")

    steps = [records[cfg.start + i] for i in range(n)]
    k3ds = np.stack([s["observation"]["proprio"]["k3ds"][:, :3] for s in steps])  # (N, 21, 3)
    imgs = np.stack([s["observation"]["image"][cfg.image_key] for s in steps])  # (N, H, W, 3)

    joints_ft = np.stack([k3ds[i : i + cfg.future_steps] for i in range(cfg.batch_size)])  # (B, ft, 21, 3)
    imgs_ft = np.stack([imgs[i : i + cfg.future_steps] for i in range(cfg.batch_size)])  # (B, ft, H, W, 3)
    batch_imgs = imgs[: cfg.batch_size]  # (B, H, W, 3), kept for compatibility/debug
    return joints_ft, imgs_ft, batch_imgs


def _scaled_intrinsics(k: np.ndarray, h: int, w: int) -> np.ndarray:
    k_scaled = k.copy()
    k_scaled[0] *= w / (k[0, 2] * 2)
    k_scaled[1] *= h / (k[1, 2] * 2)
    return k_scaled


def project_from_world(
    joints_world: np.ndarray,
    k_scaled: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    camera_axes: Literal["RDF", "DRF"],
    eps: float = 1e-8,
) -> np.ndarray:
    x_cam = (joints_world @ r_w2c.T) + t_w2c
    z = np.maximum(x_cam[:, 2], eps)
    if camera_axes == "DRF":
        return _uv_from_cam_xy(x_cam=x_cam[:, 1], y_cam=x_cam[:, 0], z_cam=z, k_scaled=k_scaled)
    return _uv_from_cam_xy(x_cam=x_cam[:, 0], y_cam=x_cam[:, 1], z_cam=z, k_scaled=k_scaled)


def project_camera_direct(
    joints_cam: np.ndarray,
    k_scaled: np.ndarray,
    camera_axes: Literal["RDF", "DRF"],
    eps: float = 1e-8,
) -> np.ndarray:
    z = np.maximum(joints_cam[:, 2], eps)
    if camera_axes == "DRF":
        return _uv_from_cam_xy(x_cam=joints_cam[:, 1], y_cam=joints_cam[:, 0], z_cam=z, k_scaled=k_scaled)
    return _uv_from_cam_xy(x_cam=joints_cam[:, 0], y_cam=joints_cam[:, 1], z_cam=z, k_scaled=k_scaled)


def log_camera_pose(
    entity: str,
    image: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    k: np.ndarray,
    camera_axes: Literal["RDF", "DRF"],
    rr: Any,
) -> None:
    rr.log(f"{entity}/world", rr.ViewCoordinates.FLU, static=True)
    r_c2w = r_w2c.T
    t_c2w = -r_c2w @ t_w2c
    rr.log(
        f"{entity}/world/cam",
        rr.Transform3D(
            translation=t_c2w.tolist(),
            mat3x3=r_c2w.tolist(),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    h, w = image.shape[:2]
    rr.log(
        f"{entity}/world/cam",
        rr.Pinhole(
            image_from_camera=_scaled_intrinsics(k, h, w),
            width=w,
            height=h,
            camera_xyz=getattr(rr.ViewCoordinates, camera_axes),
        ),
        static=True,
    )


def overlay_wireframe(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    import cv2

    overlay = image.copy()
    edges = np.array(MANO_JOINT_PAIRS, dtype=np.int64)
    for seg in uv[edges]:
        p0 = tuple(np.round(seg[0]).astype(np.int32))
        p1 = tuple(np.round(seg[1]).astype(np.int32))
        cv2.line(overlay, p0, p1, (0, 255, 0), 2)
    for p in uv:
        xy = tuple(np.round(p).astype(np.int32))
        cv2.circle(overlay, xy, 3, (255, 0, 0), -1)
    return overlay


def main(cfg: VizConfig) -> None:
    import rerun as rr

    joints_ft, imgs_ft, batch_imgs = load_test_viz_callback_style_data(cfg)
    if cfg.sample_index < 0 or cfg.sample_index >= joints_ft.shape[0]:
        raise ValueError(f"sample_index={cfg.sample_index} is out of range [0, {joints_ft.shape[0] - 1}]")

    rr.init("crossformer_viz", spawn=cfg.spawn)
    if cfg.save_rrd is not None:
        rr.save(str(cfg.save_rrd.expanduser()))

    img0 = batch_imgs[cfg.sample_index]
    h, w = img0.shape[:2]
    k_scaled = _scaled_intrinsics(DEFAULT_K, h, w)
    entity = "sweep_mano/text_conditioned"

    r_w2c, t_w2c = None, None
    try:
        ht = load_camera_ht(cfg.camera_view, cfg.extr_dir)
        r_w2c, t_w2c = world_to_camera_extrinsics(
            ht=ht,
            extrinsics_mode=cfg.extrinsics_mode,
            extrinsics_style=cfg.extrinsics_style,
            ros_to_opencv=cfg.ros_to_opencv,
        )

        log_camera_pose(
            entity=entity,
            image=img0,
            r_w2c=r_w2c,
            t_w2c=t_w2c,
            k=DEFAULT_K,
            camera_axes=cfg.camera_axes,
            rr=rr,
        )
    except FileNotFoundError:
        if cfg.projection_mode == "world_extr":
            raise FileNotFoundError(
                "projection_mode=world_extr requires extrinsics. "
                f"Expected under {cfg.extr_dir}/{cfg.camera_view}/HT.npz"
            )

    rr.set_time_sequence("train_step", cfg.train_step)
    edges = np.array(MANO_JOINT_PAIRS, dtype=np.int64)
    joints_seq = joints_ft[cfg.sample_index]
    for ft_idx in range(joints_seq.shape[0]):
        rr.set_time_sequence("ft", ft_idx)
        joints_world = joints_seq[ft_idx]  # robot/world frame
        rr.log(f"{entity}/world/hand/joints", rr.Points3D(joints_world, radii=0.003))
        rr.log(f"{entity}/world/hand/skeleton", rr.LineStrips3D(joints_world[edges]))

        if cfg.projection_mode == "camera_direct":
            uv = project_camera_direct(joints_world, k_scaled=k_scaled, camera_axes=cfg.camera_axes)
        else:
            if r_w2c is None or t_w2c is None:
                raise ValueError("Missing extrinsics for world_extr projection")
            uv = project_from_world(
                joints_world,
                k_scaled=k_scaled,
                r_w2c=r_w2c,
                t_w2c=t_w2c,
                camera_axes=cfg.camera_axes,
            )

        img_t = imgs_ft[cfg.sample_index][ft_idx]
        rr.log(f"{entity}/world/cam/image_raw", rr.Image(img_t))
        rr.log(f"{entity}/world/cam/image", rr.Image(overlay_wireframe(img_t, uv)))
        rr.log(f"{entity}/world/cam/hand_points", rr.Points2D(uv, radii=3.0))
        rr.log(f"{entity}/world/cam/hand_skeleton", rr.LineStrips2D(uv[edges]))

    print(
        f"overlay projection={cfg.projection_mode} ros_to_opencv={cfg.ros_to_opencv} "
        f"extrinsics_mode={cfg.extrinsics_mode} extrinsics_style={cfg.extrinsics_style} "
        f"camera_axes={cfg.camera_axes}"
    )


if __name__ == "__main__":
    main(tyro.cli(VizConfig))
