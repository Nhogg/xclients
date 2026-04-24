"""Random background sampling from a local COCO image dir."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image


class CocoBackgroundSampler:
    """Serves random COCO images resized/cropped to a target (W, H)."""

    def __init__(self, image_dir: Path, rng: random.Random | None = None):
        image_dir = Path(image_dir)
        self._files = sorted(image_dir.glob("*.jpg"))
        if not self._files:
            raise FileNotFoundError(f"no .jpg files under {image_dir}")
        self._rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self._files)

    def sample(self, width: int, height: int) -> tuple[np.ndarray, str]:
        path = self._rng.choice(self._files)
        img = Image.open(path).convert("RGB")
        img = _resize_cover(img, width, height)
        return np.asarray(img, dtype=np.uint8), path.name


def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale so the image covers the target, then center-crop."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def composite(rgb: np.ndarray, mask: np.ndarray, background: np.ndarray, erode_px: int = 1) -> np.ndarray:
    """Alpha-composite ``rgb`` over ``background`` using ``mask`` (0 or 255).

    ``erode_px`` shrinks the mask before compositing to hide the 1-px halo
    Isaac's anti-aliased edges leave against the black backdrop.
    """
    m = (mask > 0).astype(np.uint8)
    if erode_px > 0:
        m = _erode(m, erode_px)
    alpha = m.astype(np.float32)[..., None]
    out = rgb.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary erosion via min-pooling with a (2k+1) square kernel."""
    from numpy.lib.stride_tricks import sliding_window_view

    H, W = mask.shape
    pad = np.pad(mask, k, mode="constant", constant_values=1)
    windows = sliding_window_view(pad, (2 * k + 1, 2 * k + 1))
    return windows.min(axis=(-2, -1)).astype(mask.dtype)
