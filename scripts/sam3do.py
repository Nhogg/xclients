import jax
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from typing import Dict, Union, Optional
import time

import cv2
import numpy as np
from rich import print
import tyro
import websockets.sync.client
from webpolicy.client import Client
from webpolicy import msgpack_numpy

from xclients.renderer import Renderer
from xclients.core.cfg import Config, spec


def spec(tree) -> dict:
    return jax.tree.map(lambda x: (x.shape, x.dtype), tree)


@dataclass
class CameraInput:
    """Configuration for camera input"""
    device: int = 0

@dataclass
class ImageInput:
    """Configuration for image file input"""
    image_path: Path

@dataclass
class SAMConfig(Config):
    """Config for SAM3 server"""
    host: str = "localhost"
    port: int = 8000
    prompt: str = "object"

@dataclass
class SAM3DoConfig(Config):
    """Config for SAM3Do server"""
    input_source: Union[CameraInput, ImageInput] = tyro.MISSING
    host: str = "localhost"
    port: int = 8001
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
     
    if isinstance(cfg.input_source, CameraInput):
        ret, frame = cap.read()
        return frame
    else:  # InputType.IMAGE
        return load_image(cfg.input_source.image_path)

import json
from datetime import datetime
import numpy as np

def save_results(frame: np.ndarray, output: dict, cfg: SAM3DoConfig, frame_idx: int = 0, mask: np.ndarray = None):
    """Save visualization and raw data from server output"""
    
    # Create output directory
    output_dir = Path("sam3do_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_dir / f"result_{timestamp}_{frame_idx}"
    result_dir.mkdir(exist_ok=True)
    
    # Save the original image
    cv2.imwrite(str(result_dir / "original.png"), frame)
    
    # Save SAM3 mask if available
    if mask is not None:
        original_shape = mask.shape
        logging.info(f"Original mask shape: {original_shape}, dtype: {mask.dtype}")
        
        # Check for empty mask array first
        if mask.size == 0 or (mask.ndim == 4 and mask.shape[0] == 0):
            logging.warning(f"Empty mask array, skipping mask save")
            mask = None
        else:
            # Flatten mask if needed - keep flattening until we get (H, W)
            while mask.ndim > 2:
                if mask.shape[0] == 1:
                    mask = mask[0]  # Remove first dimension if it's 1
                elif mask.shape[1] == 1:
                    mask = mask[:, 0]  # Remove second dimension if it's 1
                else:
                    mask = mask[0]  # Default: take first element
            
            logging.info(f"Processed mask shape: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")
            
            # Normalize mask to 0-255 if needed
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                mask_vis = (mask * 255).astype(np.uint8)
            elif mask.dtype == bool:
                mask_vis = (mask.astype(np.uint8)) * 255
            else:
                mask_vis = mask.astype(np.uint8)
            
            cv2.imwrite(str(result_dir / "sam3_mask.png"), mask_vis)
            
            # Create masked image (mask multiplied by original)
            if mask_vis.ndim == 2:
                # Grayscale mask - expand to 3 channels
                mask_3channel = np.stack([mask_vis] * 3, axis=-1)
            else:
                mask_3channel = mask_vis
            
            logging.info(f"Mask_3channel shape: {mask_3channel.shape}, frame shape: {frame.shape}")
            
            # Apply mask to original image
            masked_image = (frame.astype(np.float32) * (mask_3channel.astype(np.float32) / 255.0)).astype(np.uint8)
            cv2.imwrite(str(result_dir / "masked_image.png"), masked_image)
            
            logging.info("Saved SAM3 mask and masked image")
    
    # Save pointmap visualization if available
    if "pointmap" in output:
        p = output["pointmap"]
            
        # Handle different pointmap shapes
        mu, std = p.mean(), p.std()
        p = (p - mu) / (std + 1e-8)  # Normalize to zero mean and unit variance
        p = (p+1) / 2  # Scale to [0, 1]
        p = (p * 255).astype(np.uint8)
        cv2.imwrite(str(result_dir / "pointmap.png"), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    else:
        logging.warning("No pointmap in output")

    # Save pointmap colors if available
    if "pointmap_colors" in output:
        pointmap_colors = output["pointmap_colors"]
        logging.info(f"Pointmap colors shape: {pointmap_colors.shape}, dtype: {pointmap_colors.dtype}")
        
        if pointmap_colors.ndim == 3 and pointmap_colors.shape[2] == 3:
            pointmap_colors_vis = (pointmap_colors * 255).astype(np.uint8)
            pointmap_colors_bgr = cv2.cvtColor(pointmap_colors_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(result_dir / "pointmap_colors.png"), pointmap_colors_bgr)
    
    # Save overlay with 6D rotation visualization
    overlay = frame.copy()
    y_offset = 30
    
    if "rotation" in output:
        rotation = output["rotation"]
        logging.info(f"Rotation shape: {rotation.shape}, value: {rotation}")
        cv2.putText(overlay, f"Rotation: {rotation.flatten()[:4]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        print(f"Rotation: {output['rotation']}")

    if "6drotation_normalized" in output:
        rotation_6d = output["6drotation_normalized"]
        logging.info(f"6D Rotation shape: {rotation_6d.shape}, value: {rotation_6d}")
        cv2.putText(overlay, f"6D Rot: {rotation_6d.flatten()[:6]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        print(f"6D Rotation: {output['6drotation_normalized']}")
    
    if "translation" in output:
        translation = output["translation"]
        logging.info(f"Translation shape: {translation.shape}, value: {translation}")
        cv2.putText(overlay, f"Trans: {translation.flatten()}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        print(f"Translation: {output['translation']}")
    
    if "scale" in output:
        scale = output["scale"]
        logging.info(f"Scale shape: {scale.shape}, value: {scale}")
        cv2.putText(overlay, f"Scale: {scale.flatten()}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Scale: {output['scale']}")
    
    cv2.imwrite(str(result_dir / "overlay.png"), overlay)
    
    # Save raw data as JSON
    data_to_save = {}
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": value.tolist() if value.size < 1000 else "too large to save"
            }
        else:
            data_to_save[key] = str(value)
    
    with open(result_dir / "output.json", "w") as f:
        json.dump(data_to_save, f, indent=2)
    
    logging.info(f"Saved results to {result_dir}")
    return result_dir

def render_sam3do_output(renderer, frame, sam3do_out, results_dir, frame_idx, cfg):
    """Render SAM3Do mesh output on frame"""
    try:
        # Extract mesh data
        if "mesh" not in sam3do_out:
            logging.warning("No mesh in SAM3Do output, skipping rendering")
            return
        
        mesh_data = sam3do_out["mesh"]
        logging.info(f"Mesh type: {type(mesh_data)}")
        
        # Handle mesh as dict (from server) or as object (from direct inference)
        if isinstance(mesh_data, dict):
            # Server sends mesh as dict with vertices and faces
            vertices = np.array(mesh_data["vertices"])
            faces = np.array(mesh_data["faces"])
        elif isinstance(mesh_data, list):
            # Direct inference returns list of MeshExtractResult objects
            if len(mesh_data) == 0:
                logging.warning("Mesh list is empty")
                return
            mesh_obj = mesh_data[0]
            if hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
                vertices = mesh_obj.vertices
                faces = mesh_obj.faces
                # Convert to numpy if needed
                if hasattr(vertices, 'cpu'):
                    vertices = vertices.cpu().numpy()
                else:
                    vertices = np.array(vertices)
                if hasattr(faces, 'cpu'):
                    faces = faces.cpu().numpy()
                else:
                    faces = np.array(faces)
            else:
                logging.error(f"Mesh object doesn't have vertices/faces. Available: {dir(mesh_obj)}")
                return
        else:
            logging.error(f"Unexpected mesh format: {type(mesh_data)}")
            return
        
        # Set renderer faces
        renderer.faces = faces
        
        # Extract transformation parameters
        translation = np.array(sam3do_out["translation"][0]).astype(np.float32)
        scale_val = np.array(sam3do_out["scale"][0][0]).astype(np.float32)
        rotation_quat = np.array(sam3do_out["rotation"][0]).astype(np.float32)  # [w, x, y, z] quaternion
        
        # Check if intrinsics are available and use them
        if "intrinsics" in sam3do_out:
            intrinsics = np.array(sam3do_out["intrinsics"])
            if intrinsics.ndim == 2:
                focal_length_x = intrinsics[0, 0]
                focal_length_y = intrinsics[1, 1]
                # Use average of fx and fy
                renderer.focal_length = (focal_length_x + focal_length_y) / 2.0
                logging.info(f"Using intrinsics from server: fx={focal_length_x}, fy={focal_length_y}, avg={renderer.focal_length}")
        
        logging.info(f"Mesh: vertices {vertices.shape}, faces {faces.shape}")
        logging.info(f"Transform: translation {translation}, scale {scale_val}, rotation {rotation_quat}")
        logging.info(f"Renderer focal_length: {renderer.focal_length}")
        
        # Convert quaternion to rotation matrix
        # SAM3D uses [w, x, y, z] format, scipy uses [x, y, z, w]
        from scipy.spatial.transform import Rotation as R
        quat_scipy = np.array([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]])  # Convert to scipy format
        rot_matrix = R.from_quat(quat_scipy).as_matrix()
        
        # SAM3D outputs are in PyTorch3D coordinate system
        # The renderer.py applies a 180° flip around X at line 209, which converts Y-up to Y-down
        # So we should only flip X coordinate, not Y
        
        # Apply transformations: scale then rotate
        vertices_scaled = vertices * scale_val
        vertices_rotated = vertices_scaled @ rot_matrix.T
        
        # Apply coordinate flip: Only flip X (not Y, since renderer handles that)
        vertices_opencv = vertices_rotated.copy()
        vertices_opencv[:, 0] *= -1  # Flip X only
        
        # Same for translation
        translation_opencv = translation.copy()
        translation_opencv[0] *= -1  # Flip X only
        
        logging.info(f"Translation before flip: {translation}, after flip: {translation_opencv}")
        
        # Render mesh on original frame with blue color for better visibility
        rendered_img = renderer(
            vertices=vertices_opencv,
            cam_t=translation_opencv,
            image=frame,
            mesh_base_color=(0.2, 0.5, 1.0),  # Blue color (R, G, B)
            scene_bg_color=(0, 0, 0),
            return_rgba=False,
        )
        
        # Also render standalone RGBA view
        rendered_rgba = renderer.render_rgba(
            vertices=vertices_opencv,
            cam_t=translation_opencv,
            rot_angle=0,
            mesh_base_color=(0.2, 0.5, 1.0),  # Blue color
            render_res=[640, 480],
        )
        
        # Save results
        results_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(results_dir / "rendered_on_frame.png"), (rendered_img * 255).astype(np.uint8))
        cv2.imwrite(str(results_dir / "rendered_rgba.png"), (rendered_rgba * 255).astype(np.uint8))
        
        logging.info(f"Saved rendered results to {results_dir}")
        
        if cfg.show:
            cv2.imshow("SAM3Do Rendered", (rendered_img * 255).astype(np.uint8))
            cv2.imshow("SAM3Do RGBA", (rendered_rgba * 255).astype(np.uint8))
    
    except Exception as e:
        logging.error(f"Rendering failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

def main(cfg: SAM3DoConfig) -> None:
    """Create clients with extended timeout and run inference loop"""
    import cv2
    from datetime import datetime
    
    sam3do_client = CustomClient(cfg.host, cfg.port, timeout=cfg.timeout + 30)
    sam3_client = CustomClient(cfg.sam3.host, cfg.sam3.port, timeout=30.0)

    # Initialize renderer ONCE before the loop
    # Use a more appropriate focal length based on image size
    # Typical webcam has FOV ~60-70 degrees, for 640px width: f ≈ width / (2 * tan(FOV/2))
    # For FOV=60°: f ≈ 640 / (2 * tan(30°)) ≈ 554
    focal_length = 554  # Better default for webcam
    renderer = Renderer(focal_length=focal_length)
    logging.info(f"Initialized renderer with focal_length={focal_length}")

    cap: Optional[cv2.VideoCapture] = None
    if isinstance(cfg.input_source, CameraInput):
        cap = cv2.VideoCapture(cfg.input_source.device)

    frame_idx = 0
    logging.info("Starting SAM3Do orchestration")
    
    while True:
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
            logging.error("No output received from SAM3 server")
            continue
        
        logging.info(f"Received output from SAM3: {sam3_out.keys()}")
        
        mask = sam3_out['masks']
        mask = mask.sum(axis=(0, 1)).astype(np.bool_)  # Fixed: np.bool_ instead of np.bool
        
        # Send to SAM3Do with timeout parameter
        sam3do_payload = {
            "image": frame,
            "mask": mask,
            "type": "image",
            "timeout": cfg.timeout,
        }
        logging.info(f"Sending to SAM3Do with server timeout={cfg.timeout}s...")
        sam3do_out = sam3do_client.step(sam3do_payload)
        
        if not sam3do_out:
            logging.error("No output received from SAM3Do server")
            continue
        
        logging.info(f"Received output from SAM3Do: {sam3do_out.keys()}")
        
        # Check if mesh is in output and what it contains
        if "mesh" in sam3do_out:
            logging.info(f"Mesh type: {type(sam3do_out['mesh'])}")
            if isinstance(sam3do_out['mesh'], list):
                logging.info(f"Mesh list length: {len(sam3do_out['mesh'])}")
                if len(sam3do_out['mesh']) > 0:
                    logging.info(f"First mesh item type: {type(sam3do_out['mesh'][0])}")
        
        # Save results
        results_dir = Path(f"sam3do_results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_idx}")
        save_results(frame, sam3do_out, cfg, frame_idx, mask=mask)
        
        # Log the results
        for key, value in sam3do_out.items():
            if isinstance(value, dict) and 'shape' in value:
                logging.info(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")
            elif isinstance(value, np.ndarray):
                logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (int, float, str)):
                logging.info(f"  {key}: {value}")
            elif isinstance(value, list):
                logging.info(f"  {key}: list of {len(value)} items")
            else:
                logging.info(f"  {key}: {type(value)}")

        # Render the output - FIX: pass renderer instance, not the class
        render_sam3do_output(renderer, frame, sam3do_out, results_dir, frame_idx, cfg)
        
        # Display if needed
        if cfg.show:
            cv2.imshow("Input Frame", frame)
            if isinstance(cfg.input_source, ImageInput):
                cv2.waitKey(0)
                break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        frame_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(SAM3DoConfig))
