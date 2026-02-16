from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from typing import Dict, missing, Union Optional
import time

import cv2
import numpy as np
from rich import print
import tyro
import websockets.sync.client
from webpolicy.client import Client
from webpolicy import msgpack_numpy

from xclients.core.cfg import Config, spec

@dataclass
class CameraInput:
    """Configuration for camera input"""
    device: int = 0

@dataclass
class ImageInput:
    """Configuration for image file input"""
    image_path: Path

@dataclass SAMConfig(Config):
    """Config for SAM3 server"""
    host: str = "localhost"
    port: int = 8000
    prompt: str = "object"

@dataclass
class SAM3DoConfig(Config):
    """Config for SAM3Do server"""
    host: str = "localhost"
    port: int = 8001
    input_source: Union[CameraInput, ImageInput] = Tyro.MISSING
    show: bool = False
    timeout: float = 60.0  # Timeout for server processing
    sam3: SAMConfig = field(default_factory=SAMConfig)

class CustomClient(Client):
    """Extended Client with configurable timeout"""
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, timeout: float = 120.0) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._timeout = timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def step(self, obs: Dict) -> Dict:
        """Override step to add custom timeout handling"""
        data = self._packer.pack(obs)
        self._ws.send(data)
        
        # Set a socket timeout for recv()
        self._ws.socket.settimeout(self._timeout)
        
        try:
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)
        except TimeoutError:
            raise TimeoutError(f"Server did not respond within {self._timeout} seconds")

def load_image(image_path: Path) -> np.ndarray:
    """Load image from file"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    
    logging.info(f"Loaded image with shape: {img.shape}")
    return img

def get_frame(cfg: SAM3DoConfig, cap: Optional[cv2.VideoCapture]) -> np.ndarray:
    """Get a frame from either camera or image file"""
    if cfg.input_type == InputType.CAMERA:
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame from camera {cfg.camera_device}")
        return frame
    else:  # InputType.IMAGE
        if cfg.image_path is None:
            raise ValueError("image_path must be provided when using IMAGE input type")
        return load_image(cfg.image_path)

def main(cfg: SAM3DoConfig) -> None:
    # Create clients with extended timeouts
    sam3do_client = CustomClient(cfg.host, cfg.port, timeout=cfg.timeout + 30)
    sam3_client = CustomClient(cfg.sam3.host, cfg.sam3.port, timeout=30.0)
   
    cap: Optional[cv2.VideoCapture] = None
    if isinstance(cfg.input_source, CameraInput):
        cap = cv2.VideoCapture(cfg.input_source.device)

    logging.info("Starting SAM3Do orchestration")
    
    while True:
        try:
            # Get frame from input source
            frame = get_frame(cfg, cap)
            
            # Send to SAM3 first
            sam3_payload = {
                "image": frame,
                "type": "image",
                "text": cfg.sam3.prompt,
            }
            logging.info(f"Sending to SAM3...")
            sam3_out = sam3_client.step(sam3_payload)
            
            if not sam3_out:
                logging.error("Failed to get response from SAM3")
                continue
            
            logging.info(f"SAM3 output received")
            
            # Send to SAM3Do with timeout parameter
            sam3do_payload = {
                "image": frame,
                "mask": frame[..., 0],
                "type": "image",
                "timeout": cfg.timeout,  # Pass timeout to server
            }
            logging.info(f"Sending to SAM3Do with server timeout={cfg.timeout}s...")
            sam3do_out = sam3do_client.step(sam3do_payload)
            
            if not sam3do_out:
                logging.error("Failed to read from SAM3Do")
                continue
            
            logging.info(f"SAM3Do processing complete")
            logging.info(f"SAM3Do output keys: {sam3do_out.keys()}")
            
            # Log the results
            for key, value in sam3do_out.items():
                if isinstance(value, dict) and 'shape' in value:
                    logging.info(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")
                elif isinstance(value, (int, float, str)):
                    logging.info(f"  {key}: {value}")
                else:
                    logging.info(f"  {key}: {type(value)}")
            
            # Display if needed
            if cfg.show:
                cv2.imshow("Input Frame", frame)
                if cfg.input_type == InputType.IMAGE:
                    cv2.waitKey(0)
                    break
                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
        except Exception as e:
            logging.error(f"Error: {e}")
            if cfg.input_type == InputType.IMAGE:
                break
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAM3DoConfig))
