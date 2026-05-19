from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

import cv2
import numpy as np
import tyro
from webpolicy.client import Client


@dataclass
class Config:
    host: str = "127.0.0.1"  # Dream server host
    port: int = 8082  # Dream server port
    camera: str | int = 16  # OpenCV camera index or path
    robot_ip: str | None = None  # xArm IP. If unset, cfg.q is used.
    q: list[float] = field(default_factory=lambda: [0.0] * 7)  # Fallback joint vector
    deg2rad: bool = False  # Convert fallback q from degrees to radians
    image_size: int = 200  # Dream input image size
    fx: float = 515.0  # Payload focal length in pixels along x
    fy: float = 515.0  # Payload focal length in pixels along y
    save_dir: Path | None = None  # Optional directory for image/rast/joints records

    def __post_init__(self) -> None:
        if self.save_dir is not None:
            self.save_dir = self.save_dir.expanduser().resolve()


def open_arm(robot_ip: str | None):
    if robot_ip is None:
        return None

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    arm.motion_enable(True)
    return arm


def read_joints(arm, cfg: Config) -> np.ndarray:
    if arm is not None:
        return np.asarray(arm.angles, dtype=np.float32).reshape(-1)

    q = np.asarray(cfg.q, dtype=np.float32)
    return np.deg2rad(q).astype(np.float32) if cfg.deg2rad else q


def camera_source(camera: str | int) -> str | int:
    return int(camera) if isinstance(camera, str) and camera.isdigit() else camera


def intrinsics(width: int, height: int, cfg: Config) -> np.ndarray:
    k = np.array(
        [
            [cfg.fx, 0.0, width / 2.0],
            [0.0, cfg.fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sx = cfg.image_size / float(width)
    sy = cfg.image_size / float(height)
    k[0, :] *= sx
    k[1, :] *= sy
    return k


def display_image(value: object) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and float(image.max()) <= 1.0:
            image *= 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3:
        raise ValueError(f"Expected display image, got shape {image.shape}")
    return image


def first_image(value: object) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def optional_display_image(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return display_image(value)


def dream_raster(out: dict) -> np.ndarray | None:
    for key in ("raster_image", "rast_image", "rast"):
        image = optional_display_image(out.get(key))
        if image is not None:
            return image
    return None


<<<<<<< HEAD
=======
def dream_extrinsics(out: dict) -> np.ndarray | None:
    for key in ("w2c", "HT", "extrinsics"):
        if key in out:
            pose = np.asarray(out[key], dtype=np.float32)
            while pose.ndim > 2 and pose.shape[0] == 1:
                pose = pose[0]
            if pose.shape == (4, 4):
                return pose
            logging.warning("Ignoring Dream %s with unexpected shape %s", key, pose.shape)
    return None


>>>>>>> 194541670afeac8fa10bd04960d10778a15b93ca
def raster_overlay(frame: np.ndarray, raster: np.ndarray | None) -> np.ndarray | None:
    if raster is None:
        return None

    resized = resize_for_panel(raster, frame)
    foreground = np.max(resized, axis=2) > 5
    if not np.any(foreground):
        return resized

    overlay = frame.copy()
    blended = cv2.addWeighted(frame, 0.35, resized, 0.65, 0.0)
    overlay[foreground] = blended[foreground]
    return overlay


def dream_mask(mask: object | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = first_image(mask).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask with shape (h, w) or batched singleton variants, got {arr.shape}")

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)

    lo = float(finite.min())
    hi = float(finite.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    elif hi <= 1.0:
        arr = arr.copy()
    else:
        arr = arr / 255.0
    mask_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(mask_u8, cv2.COLORMAP_TURBO)


<<<<<<< HEAD
=======
def dream_mask_raw(mask: object | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = first_image(mask).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask with shape (h, w) or batched singleton variants, got {arr.shape}")
    if arr.size and float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


>>>>>>> 194541670afeac8fa10bd04960d10778a15b93ca
def mask_overlay(frame: np.ndarray, mask: object | None) -> np.ndarray | None:
    if mask is None:
        return None

    arr = first_image(mask).astype(np.float32)
    if arr.ndim != 2:
        return None
    if arr.size and float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    mask_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    mask_u8 = cv2.resize(mask_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(frame)
    color[..., 2] = mask_u8
    return cv2.addWeighted(frame, 0.7, color, 0.3, 0.0)


def resize_for_panel(image: np.ndarray, frame: np.ndarray) -> np.ndarray:
    if image.shape[:2] != frame.shape[:2]:
        return cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image


def output_panel(frame: np.ndarray, raster: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray:
    images = [frame]
    if raster is not None:
        images.append(resize_for_panel(raster, frame))
    if mask is not None:
        images.append(resize_for_panel(mask, frame))
    return np.concatenate(images, axis=1)


def response_shapes(out: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    shapes = {}
    for key, value in out.items():
        arr = np.asarray(value)
        shapes[key] = (arr.shape, str(arr.dtype))
    return shapes


def response_stats(out: dict) -> dict[str, dict[str, float | bool]]:
    stats = {}
    for key in ("mask", "raster_image", "rast_image", "rast", "dr_success", "mask_iou", "mask_iou_reject"):
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.dtype == np.bool_:
            stats[key] = {"any": bool(arr.any()), "all": bool(arr.all())}
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        stats[key] = {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }
    return stats


def save_record(
    cfg: Config,
    frame: np.ndarray,
    model_frames: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    raster: np.ndarray | None,
    mask: np.ndarray | None,
    mask_raw: np.ndarray | None,
    extrinsics: np.ndarray | None,
) -> None:
    if cfg.save_dir is None:
        return

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(cfg.save_dir / f"{stem}_image.png"), frame)
    if raster is not None:
        cv2.imwrite(str(cfg.save_dir / f"{stem}_raster.png"), raster)
    if mask is not None:
        cv2.imwrite(str(cfg.save_dir / f"{stem}_mask.png"), mask)
    np.savez(cfg.save_dir / f"{stem}.npz", image=frame, image_model=model_frames, joints=q, K=k, raster=raster, mask=mask)
    if mask_raw is not None:
        cv2.imwrite(str(cfg.save_dir / f"{stem}_mask_raw.png"), mask_raw)

    arrays = {
        "image": frame,
        "image_model": model_frames,
        "joints": q,
        "K": k,
    }
    if raster is not None:
        arrays["raster"] = raster
    if mask is not None:
        arrays["mask"] = mask
    if mask_raw is not None:
        arrays["mask_raw"] = mask_raw
    if extrinsics is not None:
        arrays["w2c"] = extrinsics
    np.savez(cfg.save_dir / f"{stem}.npz", **arrays)


def main(cfg: Config) -> None:
    client = Client(cfg.host, cfg.port)
    arm = open_arm(cfg.robot_ip)
    cap = cv2.VideoCapture(camera_source(cfg.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.camera}")

    logging.info("Press y to capture/render/save, n to capture/render without saving, q or Esc to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.camera)
                continue

            cv2.imshow("live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 255:
                continue
            if key not in (ord("y"), ord("n")):
                logging.info("Ignoring key %s. Press y to save or n to render without saving.", key)
                continue
            save = key == ord("y")

            model_frame = cv2.resize(frame, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
            model_frames = np.stack([model_frame], axis=0)
            q = read_joints(arm, cfg)
            k = intrinsics(frame.shape[1], frame.shape[0], cfg)
            out = client.step({"image": model_frames, "type": "image", "q": q, "K": k})
            logging.info("Dream response shapes=%s", response_shapes(out))
            logging.info("Dream response stats=%s", response_stats(out))
            raster = dream_raster(out)
            if raster is not None:
                logging.info("Dream raster display shape=%s dtype=%s", raster.shape, raster.dtype)
            mask = dream_mask(out.get("mask"))
<<<<<<< HEAD
=======
            mask_raw = dream_mask_raw(out.get("mask"))
            extrinsics = dream_extrinsics(out)
            if extrinsics is not None:
                logging.info("Dream extrinsics w2c=%s", extrinsics)
>>>>>>> 194541670afeac8fa10bd04960d10778a15b93ca
            raster_on_frame = raster_overlay(frame, raster)
            overlay = mask_overlay(frame, out.get("mask"))
            if raster is None and mask is None:
                raise KeyError(f"Dream response has no raster or mask image. Keys: {sorted(out)}")

            cv2.imshow("input | raster | mask", output_panel(frame, raster, mask))
            if raster is not None:
                cv2.imshow("raster", resize_for_panel(raster, frame))
            if raster_on_frame is not None:
                cv2.imshow("raster overlay", raster_on_frame)
            if mask is not None:
                cv2.imshow("mask", mask)
            if overlay is not None:
                cv2.imshow("mask overlay", overlay)
            if save:
<<<<<<< HEAD
                save_record(cfg, frame, model_frames, q, k, raster, mask)
=======
                save_record(cfg, frame, model_frames, q, k, raster, mask, mask_raw, extrinsics)
>>>>>>> 194541670afeac8fa10bd04960d10778a15b93ca
            logging.info("Captured save=%s q=%s Dream keys=%s", save, q.tolist(), sorted(out))
    finally:
        cap.release()
        if arm is not None and hasattr(arm, "disconnect"):
            arm.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Config))
