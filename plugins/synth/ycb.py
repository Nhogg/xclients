"""
Scan YCB model assets and build a JSON catalog.

Expected layout per model:
    assets/ycb/models/ycb/<model_dir>/
        google_16k/
            kinbody.xml    <- canonical name
            nontextured.ply <- cataloged mesh
            textured.mtl
            texture_map.png

Script reads kinbody.xml for the model name,
parses ply header to get property types / counts,
streams vertices to compute axis-aligned bounding box.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET


def model_name(model_dir: Path) -> str:
    """Read canonical name from kinbody.xml"""
    return ET.parse(os.path.join(model_dir, "kinbody.xml")).getroot().attrib["name"]


def ply_bounds(filepath: Path) -> dict:
    """Compute AABB from a YCB nontextured.ply"""
    with open(filepath) as f:
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[2])
                break
        for line in f:
            if line.strip() == "end_header":
                break

        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3

        for _ in range(n):
            x, y, z = (float(v) for v in f.readline().split()[:3])
            for i, v in enumerate((x, y, z)):
                if v < mins[i]:
                    mins[i] = v
                if v > maxs[i]:
                    maxs[i] = v

    return {
        "min": [round(v, 6) for v in mins],
        "max": [round(v, 6) for v in maxs],
        "size": [round(maxs[i] - mins[i], 6) for i in range(3)],
    }


def scan(base_dir="assets/ycb/models/ycb") -> list:
    ply_paths = sorted(glob.glob(os.path.join(base_dir, "*", "google_16k", "nontextured.ply")))
    catalog = []
    for ply in ply_paths:
        model_dir = os.path.dirname(ply)
        name = model_name(model_dir)
        catalog.append({"name": name, "dir": model_dir, "ply": ply, "bounds": ply_bounds(ply)})
        print(f"{name}", file=sys.stderr)
    return catalog


def main():
    p = argparse.ArgumentParser(description="Build a JSON catalog of YCB models.")
    p.add_argument("--base-dir", default="assets/ycb/models/ycb")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    catalog = scan(args.base_dir)
    blob = json.dumps(catalog, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(blob + "\n")
        print(f"Wrote {len(catalog)} models → {args.output}", file=sys.stderr)
    else:
        print(blob)


if __name__ == "__main__":
    main()
