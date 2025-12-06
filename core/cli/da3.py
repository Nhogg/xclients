from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import rerun as rr
import tyro
from common import Config, spec
from rich import print
from webpolicy.client import Client

logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=3, suppress=True)


@dataclass
class DA3Config(Config):
    cams: list[int] = field(default_factory=lambda: [0])
    show: bool = True
    downscale: int = 1


def main(cfg: DA3Config) -> None:
    client = Client(cfg.host, cfg.port)
    rr.init("DA3", spawn=True)

    caps = {cam: cv2.VideoCapture(cam) for cam in cfg.cams}

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    while True:
        frames = {cam: caps[cam].read() for cam in cfg.cams}
        frames = {cam: frame for cam, (ret, frame) in frames.items() if ret}
        if cfg.downscale > 1:
            frames = {
                cam: cv2.resize(
                    frame,
                    (
                        frame.shape[1] // cfg.downscale,
                        frame.shape[0] // cfg.downscale,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                for cam, frame in frames.items()
            }

        print(spec(frames))
        frames = list(frames.values())

        payload = {"image": frames}

        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        print(spec(out))
        d = np.concatenate(out["depth"])
        cmap = 255.0
        d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * cmap).astype(np.uint8)

        print(out["extrinsics"])
        print(out["intrinsics"])
        print()

        points, colors = out["points"], out["colors"][:, ::-1]  # BGR to RGB

        np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],  # -
                [0, -1, 0, 0],  # -
                [0, 0, 0, 1],
            ]
        )
        # rr.log("/world", rr.Transform3D(
        # translation=FLU2RDF[:3, 3],
        # mat3x3=FLU2RDF[:3, :3],
        # ), static=True)

        rr.log(
            "/world/scene/points",
            rr.Points3D(
                points,
                colors=colors,
                radii=0.002,
            ),
            # static=True,
        )

        if cfg.show:
            cv2.imshow("Camera", d)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(DA3Config))
