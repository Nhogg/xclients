from __future__ import annotations
import sys
import os
import torch
import numpy as np
import tyro
import traceback
import trimesh
import nvdiffrast.torch as dr
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Union, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

PLUGIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_DIR.parent.parent
FP_ROOT = REPO_ROOT / "external" / "foundation_pose"

if str(FP_ROOT) not in sys.path:
    sys.path.append(str(FP_ROOT))

try:
    from estimater import *
    from datareader import YcbineoatReader
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    print(f"Could not import FoundationPose from {FP_ROOT}")
    sys.exit(1)

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8082
    device: str | None = None

class Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TrackRequest(Schema):
    type: Literal["track"]
    rgb_dir: str
    mask_path: str
    mesh_path: str
    confidence: float = 0.25
    debug: bool = False

class HealthRequest(Schema):
    type: Literal["health"]

RequestPayload = Annotated[Union[TrackRequest, HealthRequest], Field(discriminator="type")]

class FoundationPosePolicy(BasePolicy):
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.estimator = None
        self.current_mesh_path = None
        
        # Cache the persistent components so we don't reload weights every request
        self.scorer = None
        self.refiner = None
        self.glctx = None
        
        self.adapter = TypeAdapter(RequestPayload)
        print(f"DEBUG: Policy Initialized on {self.device}")

    def _init_networks(self):
        """Loads the neural networks once (heavy operation)."""
        if self.scorer is not None:
            return

        print("DEBUG: Initializing Neural Networks (Scorer & Refiner)...")
        # These classes are imported from 'estimater'
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        print("DEBUG: Networks Loaded.")

    def _load_model(self, mesh_path: str, debug_dir: str = "debug_fp_output"):
        """Loads the specific 3D mesh object."""
        if self.estimator and self.current_mesh_path == mesh_path:
            return

        self._init_networks()

        print(f"DEBUG: Loading 3D Mesh from {mesh_path}...")
        
        mesh = trimesh.load(mesh_path)
        
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=0,
            debug_dir=debug_dir
        )
        
        self.current_mesh_path = mesh_path
        print("DEBUG: FoundationPose Mesh Registered Successfully.")

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            req = self.adapter.validate_python(obs)
            
            if isinstance(req, HealthRequest):
                return {"status": "ok", "loaded": self.estimator is not None}

            if isinstance(req, TrackRequest):
                print(f"DEBUG: Processing TrackRequest for {req.mesh_path}")
                return self._run_track(req)

        except Exception as e:
            print(f"DEBUG: ERROR in step(): {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

        return {"status": "error", "message": "Unknown request"}

    def _run_track(self, req: TrackRequest):
        try:
            self._load_model(req.mesh_path, debug_dir="debug_fp")
            
            print(f"DEBUG: Reading images from {req.rgb_dir}")
            reader = YcbineoatReader(video_dir=req.rgb_dir)
            
            print(f"DEBUG: Loading mask from {req.mask_path}")
            mask_data = np.load(req.mask_path)
            key = 'masks' if 'masks' in mask_data else 'mask'
            mask = mask_data[key]
            
            if mask.ndim == 3: mask = mask[0]
            mask = mask.astype(bool).astype(np.uint8)

            print("DEBUG: Running estimator.register()...")
            pose = self.estimator.register(
                K=reader.K,
                rgb=reader.get_color(0),
                depth=reader.get_depth(0), # Dummy depth if using RGB only
                mask=mask,
                iteration=5
            )
            print(f"DEBUG: Registration complete. Pose: \n{pose}")

            return {
                "status": "success",
                "initial_pose": pose.tolist(),
                "frames_found": len(reader.color_files)
            }
        except Exception as e:
            print(f"DEBUG: ERROR in _run_track: {e}")
            traceback.print_exc()
            raise e

def main(cfg: Config) -> None:
    print(f"Starting FoundationPose Server on {cfg.host}:{cfg.port}...")
    policy = FoundationPosePolicy(device=cfg.device)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()

if __name__ == "__main__":
    main(tyro.cli(Config))
