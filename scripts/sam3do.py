from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

from xclients.core.cfg import Config, spec

@dataclass
class SAMConfig(Config):
    """Config for SAM3 server"""
    host: str = "localhost"
    port: int = 8000
    prompt: str | "object"
    image_path: Path = None

@dataclass
class CameraInput:
    """Config for camera input"""
    device: int = 0
    show: bool = True

@dataclass ImageInput:
    """Config for image input"""
    image_path: Path = None

@dataclass
class SAM3DoConfig(Config):
    """Config for SAM3Do server"""
    host: str = "localhost"
    port: int = 8000
    input_source: Union[CameraInput, ImageInput] = tyro.MISSING
    sam3: SAMConfig = SAMConfig()
    # def __post_init__(self):
    # assert self.prompt is not None, "Prompt must be provided for SAM model."

def main(cfg: SAM3DoConfig) -> None:
    sam3_client = Client(cfg.sam3.host, cfg.sam3.port)
    sam3do_client = Client(cfg.host, cfg.port)


def main(cfg: SAMConfig) -> None:
    client = Client(cfg.host, cfg.port)
    # cap = cv2.VideoCapture(0)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    while True:
        # ret, frame = cap.read()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        payload = {
            "image": frame,
            "mask": frame[..., 0],
            "type": "image",
            # "text": cfg.prompt,
            # "confidence": cfg.confidence,
        }
        print(spec(payload))
        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        print(spec(out))
        continue
        if out.get("masks") is None:
            if cfg.show:
                cv2.imshow("Camera 0", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        d = out["masks"].sum(0)[0].astype(np.uint8) * 255
        print(d.shape)
        # d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * 255.0).astype(np.uint8)
        # d = 255 - d

        # print(out["extrinsics"])
        # print(out["intrinsics"])

        if cfg.show:
            cv2.imshow("Camera 0", d)
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAMConfig))
