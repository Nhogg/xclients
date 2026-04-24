"""Overlay a MaskRenderer silhouette on an existing Isaac Sim render.

Reads ``view_i.json`` + ``view_i.png`` from a renders dir, rasterizes the
robot mask using the same camera K / extrinsics / joints, and writes
``view_i_overlay.png`` / ``view_i_mask.png`` / ``view_i_instmask.png`` into
an output dir for eyeball alignment checks.

Run:
    uv run python smoke_test_mask.py \
        --renders-dir /home/mhyatt/media/renders/robot_vga_fx50 \
        --out-dir /tmp/mask_smoke \
        --indices 0 5 12
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from background import CocoBackgroundSampler, composite
from mask_renderer import Intrinsics, MaskRenderer, joints_from_sidecar


@dataclass
class Config:
    renders_dir: Path = Path("/home/mhyatt/media/renders/robot_vga_fx50")
    out_dir: Path = Path("/tmp/mask_smoke")
    indices: list[int] = field(default_factory=lambda: [0, 5, 12])
    urdf: Path = Path(__file__).resolve().parent / "xarm7_standalone.urdf"
    include_gripper: bool = True
    coco_dir: Path = Path.home() / "bafl" / "coco" / "train2014"
    seed: int = 0
    erode_px: int = 1


def _overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 32, 32), alpha=0.45) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = (mask > 0).astype(np.float32)[..., None]
    tint = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + tint * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _colorize_instances(inst: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[1:] = rng.integers(30, 255, size=(255, 3), dtype=np.uint8)
    return palette[inst]


def _project_keypoints(sidecar: dict, mask: np.ndarray) -> tuple[int, int]:
    """Rough sanity stat: how many sidecar keypoints fall inside the mask."""
    intr = sidecar["camera"]["intrinsics"]
    H = int(intr["height"])
    W = int(intr["width"])
    inside = 0
    total = 0
    for kp in sidecar.get("keypoints", []):
        px = kp.get("pixel_xy")
        if px is None or not kp.get("visible", False):
            continue
        total += 1
        x, y = int(px[0]), int(px[1])
        if 0 <= x < W and 0 <= y < H and mask[y, x] > 0:
            inside += 1
    return inside, total


def run(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    renderer = MaskRenderer(cfg.urdf, include_gripper=cfg.include_gripper)
    rng = random.Random(cfg.seed)
    coco = CocoBackgroundSampler(cfg.coco_dir, rng=rng) if cfg.coco_dir.exists() else None
    if coco is None:
        print(f"[warn] coco_dir {cfg.coco_dir} missing — skipping compositing")
    else:
        print(f"[info] using {len(coco)} coco backgrounds from {cfg.coco_dir}")

    for i in cfg.indices:
        side_path = cfg.renders_dir / f"view_{i}.json"
        img_path = cfg.renders_dir / f"view_{i}.png"
        if not (side_path.exists() and img_path.exists()):
            print(f"[skip] view_{i}: missing file")
            continue

        sidecar = json.loads(side_path.read_text())
        intr = Intrinsics.from_sidecar(sidecar["camera"]["intrinsics"])
        w2c = np.asarray(sidecar["camera"]["extrinsics"]["world_to_camera"], dtype=np.float64)
        joints_rad = joints_from_sidecar(sidecar["joints"])

        binary, inst = renderer.render(joints_rad, w2c, intr)

        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        if rgb.shape[:2] != binary.shape:
            raise RuntimeError(
                f"image shape {rgb.shape[:2]} != mask shape {binary.shape}"
            )
        overlay = _overlay(rgb, binary)
        inst_rgb = _colorize_instances(inst)

        inside, total = _project_keypoints(sidecar, binary)
        cov = float((binary > 0).mean())
        print(
            f"view_{i}: mask coverage={cov:.3%} "
            f"keypoints_in_mask={inside}/{total}"
        )

        Image.fromarray(binary).save(cfg.out_dir / f"view_{i}_mask.png")
        Image.fromarray(inst_rgb).save(cfg.out_dir / f"view_{i}_instmask.png")
        Image.fromarray(overlay).save(cfg.out_dir / f"view_{i}_overlay.png")

        if coco is not None:
            bg, bg_name = coco.sample(intr.width, intr.height)
            comp = composite(rgb, binary, bg, erode_px=cfg.erode_px)
            Image.fromarray(comp).save(cfg.out_dir / f"view_{i}_composite.png")
            print(f"  composited onto {bg_name}")

    print(f"wrote debug images to {cfg.out_dir}")


if __name__ == "__main__":
    run(tyro.cli(Config))
