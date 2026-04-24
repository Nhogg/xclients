"""Preset `UsdPreviewSurface` parameter bundles for the robot.

All presets use only ``diffuseColor`` + ``roughness`` + ``metallic``, which the
existing robot material already exposes — no textures, no MDL, no UV layout
required. Good enough for chrome, brushed metals, plastics, rubber, and
painted finishes. Wood / carbon fiber need textures and are out of scope here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialPreset:
    name: str
    diffuse: tuple[float, float, float]
    roughness: tuple[float, float]   # (min, max) — jittered per sample
    metallic: tuple[float, float]
    weight: float = 1.0              # relative sampling weight


PRESETS: list[MaterialPreset] = [
    MaterialPreset("chrome",            (0.95, 0.95, 0.96), (0.03, 0.12), (0.82, 0.88), weight=1.2),
    MaterialPreset("brushed_aluminum",  (0.82, 0.82, 0.84), (0.30, 0.45), (0.80, 0.90), weight=1.0),
    MaterialPreset("copper",            (0.95, 0.62, 0.44), (0.15, 0.35), (0.80, 0.90), weight=0.8),
    MaterialPreset("brass",             (0.90, 0.78, 0.40), (0.18, 0.40), (0.80, 0.90), weight=0.6),
    MaterialPreset("gunmetal",          (0.18, 0.19, 0.21), (0.25, 0.45), (0.85, 0.92), weight=1.0),
    MaterialPreset("matte_black",       (0.03, 0.03, 0.03), (0.65, 0.85), (0.0, 0.05), weight=1.2),
    MaterialPreset("glossy_white",      (0.93, 0.93, 0.93), (0.08, 0.22), (0.0, 0.10), weight=1.0),
    MaterialPreset("rubber_black",      (0.05, 0.05, 0.05), (0.85, 0.95), (0.0, 0.0), weight=0.6),
    MaterialPreset("safety_orange",     (0.95, 0.36, 0.05), (0.30, 0.55), (0.0, 0.10), weight=0.8),
    MaterialPreset("lab_gray",          (0.55, 0.56, 0.58), (0.40, 0.65), (0.0, 0.15), weight=1.0),
]


def sample(rng: random.Random) -> tuple[MaterialPreset, tuple[float, float, float], float, float]:
    """Pick a preset and return (preset, diffuse_rgb, roughness, metallic).

    Roughness + metallic are uniformly jittered within the preset's range
    so two views with the same preset still get some variation.
    """
    preset = rng.choices(PRESETS, weights=[p.weight for p in PRESETS], k=1)[0]
    roughness = rng.uniform(*preset.roughness)
    metallic = rng.uniform(*preset.metallic)
    return preset, preset.diffuse, roughness, metallic


def by_name(name: str) -> MaterialPreset:
    for p in PRESETS:
        if p.name == name:
            return p
    raise KeyError(f"unknown preset: {name!r} (known: {[p.name for p in PRESETS]})")
