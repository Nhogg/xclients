from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tyro

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    camera: str | int = 0  # OpenCV camera index or path
    output_dir: Path = Path("data/dream_records")  # Directory for captured images and joint metadata
    robot_host: str | None = "192.168.1.231"  # Set to None to record cfg.q instead of live xArm joints
    q: list[float] = field(default_factory=lambda: [0.0] * 7)  # Fallback joint vector when robot_host is None
    deg2rad: bool = False  # Treat cfg.q as degrees; xArm readings are always radians
    limit: int | None = None  # Maximum number of captures, or None for no limit
    window_name: str = "Record Dream"

    def __post_init__(self) -> None:
        self.output_dir = self.output_dir.expanduser().resolve()


def connect_xarm(host: str | None):
    if host is None:
        logging.info("robot_host is None; recording fallback cfg.q joint values")
        return None

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(host, is_radian=True)
    ret = arm.connect()
    if ret not in (None, 0):
        raise RuntimeError(f"Failed to connect to xArm at {host}: {ret}")
    logging.info("Connected to xArm at %s", host)
    return arm


def read_joint_positions(arm: object | None, cfg: Config) -> np.ndarray:
    if arm is None:
        q = np.asarray(cfg.q, dtype=np.float32).reshape(-1)
        return np.deg2rad(q).astype(np.float32) if cfg.deg2rad else q

    angles = getattr(arm, "angles", None)
    if angles is None or len(angles) == 0:
        ret, angles = arm.get_servo_angle(is_radian=True)
        if ret != 0:
            raise RuntimeError(f"Failed to read xArm joint angles: {ret}")

    q = np.asarray(angles, dtype=np.float32).reshape(-1)
    if q.size < 7:
        raise ValueError(f"Expected at least 7 xArm joint angles, got {q.size}: {q}")
    return q[:7]


def save_capture(cfg: Config, *, frame: np.ndarray, q_rad: np.ndarray, capture_idx: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = cfg.output_dir / f"dream_{stamp}"
    image_path = stem.with_suffix(".png")
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")

    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to write image to {image_path}")

    np.savez(npz_path, image=frame, xarm_joints=q_rad)
    metadata = {
        "timestamp": stamp,
        "capture_idx": capture_idx,
        "camera": cfg.camera,
        "image": str(image_path),
        "npz": str(npz_path),
        "xarm_joints": q_rad.astype(float).tolist(),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return image_path


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cfg.camera}")

    arm = connect_xarm(cfg.robot_host)
    capture_idx = 0
    window = f"{cfg.window_name} {cfg.camera}"
    logging.info("Showing camera %s. Press space to capture, q or Esc to quit.", cfg.camera)

    try:
        while cfg.limit is None or capture_idx < cfg.limit:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera %s", cfg.camera)
                continue

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue

            q_rad = read_joint_positions(arm, cfg)
            image_path = save_capture(cfg, frame=frame, q_rad=q_rad, capture_idx=capture_idx)
            logging.info("Captured %s with joints %s", image_path, np.round(q_rad, 4).tolist())
            capture_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main(tyro.cli(Config))
