from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Literal

import cv2
import numpy as np
import torch
import tyro
import webpolicy.client as webpolicy_client
from webpolicy.client import Client


@dataclass
class Config:
    data_dir: Path = Path("~/rr_good")  # Directory containing record_data.py npz files
    data_select: list[int] = field(default_factory=lambda: [-1])  # select records idx or -1 for all
    output_dir: Path | None = None  # Run output directory. Defaults under data_dir.
    image_size: int = 200  # Square size for SAM, Dream, and DR
    max_records: int | None = None  # Optional cap for quick tests
    extrinsics_path: Path | None = None  # Optional file/dir with w2c, HT, or extrinsics
    record_w2c_index: int | None = None  # Optional record index for static w2c; defaults to scored best
    intrinsics_path: Path | None = None  # Optional file/dir with K or intrinsics

    sam_host: str = "127.0.0.1"  # SAM3 server host
    sam_port: int = 8080  # SAM3 server port
    sam_prompt: str = "robot arm"  # SAM3 text prompt
    sam_confidence: float = 0.5  # SAM3 confidence threshold
    sam_raw_webpolicy: bool = True  # Send raw payloads for older SAM3 webpolicy servers

    dream_host: str = "127.0.0.1"  # Crossformer DREAM server host
    dream_port: int = 8002  # Crossformer DREAM server port
    dream_joint_units: Literal["deg", "rad"] = "deg"  # DREAM server converts deg to rad internally
    call_dream: bool = False  # Call Dream for initial w2c instead of using extrinsics_path

    run_dr: bool = True  # Run server_roboreg DR after cached masks/Dream poses are ready
    inspect: bool = False  # Only print record/cache state; do not call servers or DR
    refresh_cache: bool = False  # Recompute SAM/Dream outputs even when cache files exist

    dr_optimizer: str = "Adam"  # torch.optim optimizer name
    dr_lr: float = 3e-3  # DR optimizer learning rate
    dr_max_iterations: int = 1000  # DR optimization iterations
    dr_step_size: int = 100  # LR scheduler step size
    dr_gamma: float = 0.8  # LR scheduler gamma
    dr_mode: Literal["distance-function", "segmentation"] = "segmentation"  # DR loss target

    ros_package: str = "xarm_description"  # Robot description package for roboreg
    xacro_path: str = "urdf/xarm_device.urdf.xacro"  # Xacro path relative to ros_package
    urdf_path: Path | None = None  # Direct URDF path; defaults to server_roboreg bundled xArm URDF
    root_link_name: str = "link_base"  # Robot root link
    end_link_name: str = "link7"  # Robot end link
    collision_meshes: bool = False  # Use collision meshes instead of visual meshes

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        if self.extrinsics_path is not None:
            self.extrinsics_path = self.extrinsics_path.expanduser().resolve()
        if self.intrinsics_path is not None:
            self.intrinsics_path = self.intrinsics_path.expanduser().resolve()
        if self.urdf_path is not None:
            self.urdf_path = self.urdf_path.expanduser().resolve()
        if self.output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.data_dir / f"dream_dr_{stamp}"
        self.output_dir = self.output_dir.expanduser().resolve()


@dataclass
class Record:
    stem: str
    path: Path
    image: np.ndarray
    joints: np.ndarray
    intrinsics: np.ndarray
    w2c: np.ndarray | None


class RawWebPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.uri = f"ws://{host}:{port}"
        logging.info("Waiting for server at %s...", self.uri)
        self.ws = webpolicy_client.websockets.sync.client.connect(self.uri, compression=None, max_size=None)
        self.packer = webpolicy_client.msgpack_numpy.Packer()
        self.metadata = webpolicy_client.msgpack_numpy.unpackb(self.ws.recv())

    def step(self, obs: dict) -> dict:
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpacked = webpolicy_client.msgpack_numpy.unpackb(response)
        if isinstance(unpacked, dict) and "action" in unpacked:
            return unpacked["action"]
        return unpacked


def scale_intrinsics(k: np.ndarray, old_h: int, old_w: int, new_h: int, new_w: int) -> np.ndarray:
    out = np.asarray(k, dtype=np.float32).copy()
    out[0, :] *= new_w / float(old_w)
    out[1, :] *= new_h / float(old_h)
    return out


def model_image(data: dict[str, np.ndarray], size: int) -> tuple[np.ndarray, np.ndarray]:
    if "image_model" in data:
        image = np.asarray(data["image_model"])
        while image.ndim > 3 and image.shape[0] == 1:
            image = image[0]
    else:
        image = np.asarray(data["image"])

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB/BGR image shape (h, w, 3), got {image.shape}")

    image = image.astype(np.uint8, copy=False)
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image, np.asarray(data["K"], dtype=np.float32)

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, scale_intrinsics(data["K"], h, w, size, size)


def load_records(cfg: Config) -> list[Record]:
    paths = sorted(cfg.data_dir.glob("*.npz"))
    if cfg.data_select != [-1]:
        selected = []
        for index in cfg.data_select:
            try:
                selected.append(paths[index])
            except IndexError as exc:
                raise IndexError(
                    f"data_select index {index} is out of range for {len(paths)} .npz files under {cfg.data_dir}"
                ) from exc
        paths = selected
    if cfg.max_records is not None:
        paths = paths[: cfg.max_records]
    if not paths:
        raise FileNotFoundError(f"No .npz records found under {cfg.data_dir}")

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        image, k = model_image(arrays, cfg.image_size)
        records.append(
            Record(
                stem=path.stem,
                path=path,
                image=image,
                joints=np.asarray(arrays["joints"], dtype=np.float32).reshape(-1)[:7],
                intrinsics=k,
                w2c=np.asarray(arrays["w2c"], dtype=np.float32) if "w2c" in arrays else None,
            )
        )
    return records


def latest_npz(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")
    return files[-1]


def load_first_array(path: Path, keys: tuple[str, ...]) -> np.ndarray | None:
    npz = latest_npz(path)
    loaded = np.load(npz, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float32)
    with loaded as data:
        for key in keys:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
        logging.warning("No keys %s found in %s. Available keys: %s", keys, npz, data.files)
    return None


def load_intrinsics_override(cfg: Config) -> np.ndarray | None:
    if cfg.intrinsics_path is None:
        return None
    if not cfg.intrinsics_path.exists():
        logging.warning("Intrinsics path does not exist: %s", cfg.intrinsics_path)
        return None
    intr = load_first_array(cfg.intrinsics_path, ("K", "intrinsics", "camera_matrix"))
    if intr is None:
        return None
    while intr.ndim > 2 and intr.shape[0] == 1:
        intr = intr[0]
    if intr.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3), got {intr.shape} from {cfg.intrinsics_path}")
    return intr.astype(np.float32)


def apply_intrinsics_override(records: list[Record], intrinsics: np.ndarray | None) -> None:
    if intrinsics is None:
        return
    for record in records:
        record.intrinsics = intrinsics.copy()


def mask_from_sam(out: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = out.get("masks")
    if masks is None:
        raise KeyError(f"SAM response has no masks key. Keys: {sorted(out)}")

    arr = np.asarray(masks)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        mask = np.any(arr > 0, axis=0)
    elif arr.ndim == 2:
        mask = arr > 0
    else:
        raise ValueError(f"Unsupported SAM masks shape {arr.shape}")

    out_mask = mask.astype(np.uint8) * 255
    if out_mask.shape != shape:
        out_mask = cv2.resize(out_mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return out_mask


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_sam_masks(cfg: Config, records: list[Record]) -> np.ndarray:
    mask_dir = cfg.output_dir / "masks"
    masks = []
    missing = []
    for record in records:
        path = mask_dir / f"{record.stem}_mask.png"
        if path.exists() and not cfg.refresh_cache:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read cached mask {path}")
            masks.append(mask)
        else:
            missing.append(record)

    if missing:
        client = (
            RawWebPolicyClient(cfg.sam_host, cfg.sam_port)
            if cfg.sam_raw_webpolicy
            else Client(cfg.sam_host, cfg.sam_port)
        )
        for record in missing:
            logging.info("Requesting SAM mask for %s", record.stem)
            arm_out = client.step(
                {
                    "type": "image",
                    "image": record.image,
                    "text": cfg.sam_prompt,
                    "confidence": cfg.sam_confidence,
                }
            )
            gripper_out = client.step(
                {
                    "type": "image",
                    "image": record.image,
                    "text": "end effector",
                    "confidence": 0.3,
                }
            )
            arm_mask = mask_from_sam(arm_out, record.image.shape[:2])
            gripper_mask = mask_from_sam(gripper_out, record.image.shape[:2])
            binarize = lambda arr: (arr / 255) > 0.5
            arm_mask, gripper_mask = binarize(arm_mask), binarize(gripper_mask)
            tocv2 = lambda arr: (arr * 255).astype(np.uint8)
            mask = np.logical_and(arm_mask, np.logical_not(gripper_mask)).astype(np.uint8)
            write_image(mask_dir / f"{record.stem}_mask.png", tocv2(mask))
            write_image(mask_dir / f"{record.stem}_grippermask.png", tocv2(gripper_mask))
            write_image(mask_dir / f"{record.stem}_arm-mask.png", tocv2(arm_mask))

    if missing:
        return collect_sam_masks(Config(**(asdict(cfg) | {"refresh_cache": False})), records)
    return np.stack(masks).astype(np.uint8)


def dream_joints(joints: np.ndarray, units: str) -> np.ndarray:
    if units == "deg":
        return np.rad2deg(joints).astype(np.float32)
    if units == "rad":
        return joints.astype(np.float32)
    raise ValueError(f"Unsupported dream_joint_units={units}")


def collect_dream_pose(cfg: Config, records: list[Record], masks: np.ndarray) -> dict:
    cache = cfg.output_dir / "dream_outputs.npz"
    if cache.exists() and not cfg.refresh_cache:
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    client = Client(cfg.dream_host, cfg.dream_port)
    images = np.stack([record.image for record in records]).astype(np.uint8)
    joints = np.stack([dream_joints(record.joints, cfg.dream_joint_units) for record in records])
    intrinsics = np.stack([record.intrinsics for record in records]).astype(np.float32)

    out = client.step({"image": images, "q": joints, "K": intrinsics, "mask": masks})
    if "w2c" not in out:
        raise KeyError(f"Dream response has no w2c key. Keys: {sorted(out)}")

    wanted = {
        key: np.asarray(value)
        for key, value in out.items()
        if key
        in {
            "w2c",
            "K",
            "pnp_success",
            "pnp_reproj_px",
            "mask_iou",
            "mask_iou_reject",
            "dr_success",
            "dr_loss_init",
            "dr_loss_final",
            "dr_cxy_delta",
        }
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **wanted)
    return wanted


def collect_initial_pose(cfg: Config, records: list[Record], masks: np.ndarray) -> dict:
    if cfg.call_dream:
        out = collect_dream_pose(cfg, records, masks)
        out["pose_source"] = np.asarray("dream")
        return out

    if cfg.extrinsics_path is not None:
        ht = load_first_array(cfg.extrinsics_path, ("w2c", "HT", "extrinsics"))
        if ht is None:
            raise ValueError(f"No w2c/HT/extrinsics found in {cfg.extrinsics_path}")
        return {"w2c": apply_w2c_adjustments(cfg, ht), "pose_source": np.asarray(str(cfg.extrinsics_path))}

    record, w2c = select_record_w2c(cfg, records, masks)
    if w2c is not None:
        logging.info("Using static initial w2c from record %s", record.path)
        return {"w2c": w2c, "pose_source": np.asarray(str(record.path))}
    raise ValueError("No record has w2c. Pass --extrinsics-path or use --call-dream to get initial extrinsics.")


def apply_w2c_adjustments(_cfg: Config, ht: np.ndarray) -> np.ndarray:
    return np.asarray(ht, dtype=np.float32).copy()


def select_record_w2c(cfg: Config, records: list[Record], masks: np.ndarray) -> tuple[Record, np.ndarray | None]:
    candidates = [(i, record, record.w2c) for i, record in enumerate(records) if record.w2c is not None]
    if not candidates:
        return records[0], None

    if cfg.record_w2c_index is not None:
        if cfg.record_w2c_index < 0 or cfg.record_w2c_index >= len(records):
            raise ValueError(f"record_w2c_index must be in [0, {len(records) - 1}], got {cfg.record_w2c_index}")
        record = records[cfg.record_w2c_index]
        if record.w2c is None:
            raise ValueError(f"Record {record.path} has no w2c")
        return record, apply_w2c_adjustments(cfg, record.w2c)

    if len(candidates) == 1:
        _, record, w2c = candidates[0]
        return record, apply_w2c_adjustments(cfg, w2c)

    try:
        return score_record_w2c(cfg, records, masks, candidates)
    except Exception:
        logging.exception("Failed to score record w2c candidates; falling back to first record w2c")
        _, record, w2c = candidates[0]
        return record, w2c


def score_record_w2c(
    cfg: Config,
    records: list[Record],
    masks: np.ndarray,
    candidates: list[tuple[int, Record, np.ndarray]],
) -> tuple[Record, np.ndarray]:
    ensure_plugin_src()

    from server_roboreg.common import HydraConfig
    from server_roboreg.render import Renderer, RendererConfig

    bundled_urdf = Path(__file__).resolve().parents[1] / "plugins/server_roboreg/xarm7_standalone.urdf"
    hcfg = HydraConfig(
        ros_package=cfg.ros_package,
        xacro_path=cfg.xacro_path,
        urdf=cfg.urdf_path or bundled_urdf,
        root_link_name=cfg.root_link_name,
        end_link_name=cfg.end_link_name,
        collision_meshes=cfg.collision_meshes,
    )
    renderer = Renderer(
        hcfg,
        RendererConfig(batch_size=len(records)),
        height=masks[0].shape[0],
        width=masks[0].shape[1],
        intr=np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
    )
    joints = torch.tensor(np.stack([record.joints for record in records]), dtype=torch.float32, device=renderer.device)
    mask_bin = masks > 0
    intr = torch.tensor(
        np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
        dtype=torch.float32,
        device=renderer.device,
    )

    best_score = -1.0
    best_record = candidates[0][1]
    best_w2c = candidates[0][2]
    for index, record, w2c in candidates:
        adjusted_w2c = apply_w2c_adjustments(cfg, w2c)
        ht = np.repeat(adjusted_w2c[None], len(records), axis=0)
        render = render_cv_w2c(
            renderer,
            joints,
            torch.tensor(ht, dtype=torch.float32, device=renderer.device),
            intr,
            masks[0].shape[0],
            masks[0].shape[1],
        )
        # render: torch size B,W,H,C=1
        render_bin = render.detach().cpu().numpy()[..., 0] > 0.5  # np B,W,H
        print(type(render_bin))
        print("render bin shape", render_bin.shape)
        intersection = np.logical_and(render_bin, mask_bin).sum()
        union = np.logical_or(render_bin, mask_bin).sum()
        render_area = render_bin.sum()
        print("render bin shape", render_bin.shape)
        print(intersection, union)
        print("iou", intersection / union)
        print("render bin mean", render_bin.mean())
        mask_area = mask_bin.sum()
        area_ratio = render_area / float(mask_area) if mask_area > 0 else 0.0
        area_penalty = min(area_ratio, 1.0 / area_ratio) if area_ratio > 0.0 else 0.0
        iou = float(intersection / union) if union > 0 else 0.0
        score = iou * area_penalty
        logging.info(
            "record w2c candidate %d score=%.6f iou=%.6f render_area=%d mask_area=%d area_ratio=%.3f",
            index,
            score,
            iou,
            render_area,
            mask_area,
            area_ratio,
        )
        if score > best_score or (score == best_score and render_area > 0):
            best_score = score
            best_record = record
            best_w2c = adjusted_w2c
    return best_record, best_w2c


def opencv_projection(intr: torch.Tensor, width: int, height: int) -> torch.Tensor:

    projection = torch.zeros(4, 4, dtype=intr.dtype, device=intr.device)
    znear, zfar = 0.01, 10.0
    projection[0, 0] = 2.0 * intr[0, 0] / width
    projection[1, 1] = 2.0 * intr[1, 1] / height
    projection[0, 2] = 1.0 - 2.0 * intr[0, 2] / width
    projection[1, 2] = 2.0 * intr[1, 2] / height - 1.0
    projection[2, 2] = -(zfar + znear) / (zfar - znear)
    projection[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    projection[3, 2] = -1.0
    return projection


def render_cv_w2c(
    renderer,
    joints: torch.Tensor,
    w2c: torch.Tensor,
    intr: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:

    renderer.scene.robot.configure(joints)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=w2c.dtype, device=w2c.device))
    mvp = opencv_projection(intr, width, height) @ (flip @ w2c)
    observed_vertices = torch.matmul(renderer.scene.robot.configured_vertices, mvp.transpose(-1, -2))
    render = renderer.scene.renderer.constant_color(
        observed_vertices,
        renderer.scene.robot.faces,
        renderer.scene.cameras[renderer.camera_name].resolution,
    )
    return torch.flip(render, dims=[1])


def assert_dream_pose(out: dict, n: int) -> np.ndarray:
    ht = np.asarray(out["w2c"], dtype=np.float32)
    if ht.shape != (4, 4) and ht.shape != (n, 4, 4):
        raise ValueError(f"Expected w2c shape (4, 4) or ({n}, 4, 4), got {ht.shape}")
    axes = (0, 1) if ht.shape == (4, 4) else (1, 2)
    bad = np.flatnonzero(~np.isfinite(ht).all(axis=axes))
    if bad.size:
        raise ValueError(f"Dream produced non-finite w2c for record indices {bad.tolist()}")
    return ht


def ensure_plugin_src() -> None:
    script_dir = Path(__file__).resolve().parent
    sys.path = [path for path in sys.path if Path(path or ".").resolve() != script_dir]

    module = sys.modules.get("roboreg")
    module_file = Path(getattr(module, "__file__", "")).resolve() if module is not None else None
    if module_file == script_dir / "roboreg.py":
        sys.modules.pop("roboreg", None)

    plugin_src = Path(__file__).resolve().parents[1] / "plugins/server_roboreg/src"
    if str(plugin_src) not in sys.path:
        sys.path.insert(0, str(plugin_src))


def run_dr(cfg: Config, records: list[Record], masks: np.ndarray, ht: np.ndarray) -> dict:
    ensure_plugin_src()
    from server_roboreg.common import DRConfig, HydraConfig, REGISTRATION_MODE
    from server_roboreg.dr import DR

    bundled_urdf = Path(__file__).resolve().parents[1] / "plugins/server_roboreg/xarm7_standalone.urdf"
    hcfg = HydraConfig(
        ros_package=cfg.ros_package,
        xacro_path=cfg.xacro_path,
        urdf=cfg.urdf_path or bundled_urdf,
        root_link_name=cfg.root_link_name,
        end_link_name=cfg.end_link_name,
        collision_meshes=cfg.collision_meshes,
    )
    hcfg.dr = DRConfig(
        optimizer=cfg.dr_optimizer,
        lr=cfg.dr_lr,
        max_iterations=cfg.dr_max_iterations,
        step_size=cfg.dr_step_size,
        gamma=cfg.dr_gamma,
        mode=REGISTRATION_MODE(cfg.dr_mode),
    )

    payload = {
        "images": np.stack([record.image for record in records]).astype(np.uint8),
        "joints": np.stack([record.joints for record in records]).astype(np.float32),
        "mask": masks.astype(np.uint8),
        "intrinsics": np.stack([record.intrinsics for record in records]).astype(np.float32)[0],
        "HT": ht.astype(np.float32),
        "ht_is_cv_w2c": True,
    }
    return DR(hcfg.dr, hcfg).step(payload)


def save_outputs(cfg: Config, records: list[Record], masks: np.ndarray, initial: dict, dr_out: dict | None) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "HT_initial.npy", np.asarray(initial["w2c"], dtype=np.float32))
    if cfg.call_dream:
        np.save(cfg.output_dir / "HT_dream.npy", np.asarray(initial["w2c"], dtype=np.float32))
    if dr_out is not None:
        np.save(cfg.output_dir / "HT_dr.npy", np.asarray(dr_out["HT"], dtype=np.float32))
        for key in ("overlays", "difference", "renders", "render_overlays"):
            if key not in dr_out:
                continue
            for record, image in zip(records, np.asarray(dr_out[key]), strict=True):
                write_image(cfg.output_dir / key / f"{record.stem}_{key}.png", image)

    metadata = {
        "records": [str(record.path) for record in records],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "mask_shape": list(masks.shape),
        "initial_pose_keys": sorted(initial),
        "dr_keys": [] if dr_out is None else sorted(dr_out),
    }
    with (cfg.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def print_inspect(records: list[Record], cfg: Config) -> None:
    print(f"records: {len(records)}")
    print(f"data_dir: {cfg.data_dir}")
    print(f"output_dir: {cfg.output_dir}")
    joints = np.stack([record.joints for record in records])
    if np.allclose(joints, 0.0):
        logging.warning("All loaded joint vectors are zero. DR renders will use the same robot pose for every record.")
    elif np.allclose(joints, joints[0]):
        logging.warning("All loaded joint vectors are identical. DR renders will not reflect per-record robot motion.")
    for record in records:
        print(
            f"{record.stem}: image={record.image.shape} joints={record.joints.shape} "
            f"K_fx_fy=({record.intrinsics[0, 0]:.3f}, {record.intrinsics[1, 1]:.3f}) "
            f"w2c={'yes' if record.w2c is not None else 'no'}"
        )
    warn_w2c_consistency(records)


def warn_w2c_consistency(records: list[Record]) -> None:
    poses = [(record, record.w2c) for record in records if record.w2c is not None]
    if len(poses) < 2:
        return
    base = poses[0][1]
    trans_deltas = []
    angle_deltas = []
    for _, pose in poses[1:]:
        delta = pose @ np.linalg.inv(base)
        trans_deltas.append(float(np.linalg.norm(delta[:3, 3])))
        cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle_deltas.append(float(np.rad2deg(np.arccos(cos_angle))))

    max_trans = max(trans_deltas, default=0.0)
    max_angle = max(angle_deltas, default=0.0)
    if max_trans > 0.05 or max_angle > 5.0:
        logging.warning(
            "Stored record w2c poses are inconsistent for a static camera: "
            "max relative translation %.3f m, max relative rotation %.1f deg. "
            "Treat them as per-frame DREAM/PnP estimates, not calibrated extrinsics.",
            max_trans,
            max_angle,
        )


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    records = load_records(cfg)
    apply_intrinsics_override(records, load_intrinsics_override(cfg))
    print_inspect(records, cfg)
    if cfg.inspect:
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_image(cfg.output_dir / "images" / f"{record.stem}_image.png", record.image)

    masks = collect_sam_masks(cfg, records)
    initial = collect_initial_pose(cfg, records, masks)
    ht = assert_dream_pose(initial, len(records))

    dr_out = run_dr(cfg, records, masks, ht) if cfg.run_dr else None
    print(dr_out)
    save_outputs(cfg, records, masks, initial, dr_out)
    logging.info("Wrote outputs to %s", cfg.output_dir)


if __name__ == "__main__":
    main(tyro.cli(Config))
