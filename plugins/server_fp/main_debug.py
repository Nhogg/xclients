from __future__ import annotations
import sys
import os
import torch
import numpy as np
import tyro
import traceback
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
    from estimater import FoundationPose
    from datareader import YcbineoatReader
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
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
        self.current_mesh = None
        self.adapter = TypeAdapter(RequestPayload)
        print(f"DEBUG: Policy Initialized on {self.device}")

    def _load_model(self, mesh_path: str):
        if self.estimator and self.current_mesh == mesh_path:
            return

        print(f"DEBUG: Loading FoundationPose weights...")
        weights = FP_ROOT / "weights" / "foundationpose_weights.pth"
        refine = FP_ROOT / "weights" / "refine_model_weights.pth"

        if not weights.exists():
            raise FileNotFoundError(f"Missing weights: {weights}")

        self.estimator = FoundationPose(
            model_file=str(weights),
            refine_model_file=str(refine),
            mesh_file=mesh_path,
            debug=0,
            debug_dir="debug_fp_output"
        )
        self.current_mesh = mesh_path
        print("DEBUG: Model Loaded Successfully.")

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        print(f"DEBUG: Received request: {obs.keys() if isinstance(obs, dict) else obs}")
        try:
            req = self.adapter.validate_python(obs)
            
            if isinstance(req, HealthRequest):
                return {"status": "ok", "loaded": self.estimator is not None}

            if isinstance(req, TrackRequest):
                print(f"DEBUG: Processing TrackRequest for {req.mesh_path}")
                return self._run_track(req)

        except Exception as e:
            print(f"DEBUG: ERROR in step(): {e}")
            traceback.print_exc() # Print full crash report to server console
            return {"status": "error", "message": str(e)}

        return {"status": "error", "message": "Unknown request"}

    def _run_track(self, req: TrackRequest):
        try:
            self._load_model(req.mesh_path)
            
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
                depth=reader.get_depth(0),
                mask=mask,
                iteration=5
            )
            print("DEBUG: Registration complete.")

            return {
                "status": "success",
                "initial_pose": pose.tolist(),
                "frames_found": len(reader.color_files)
            }
        except Exception as e:
            print(f"DEBUG: ERROR in _run_track: {e}")
            traceback.print_exc()
            raise e # Re-raise to be caught by step()

def main(cfg: Config) -> None:
    print(f"Starting FoundationPose Server on {cfg.host}:{cfg.port}...")
    policy = FoundationPosePolicy(device=cfg.device)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()

if __name__ == "__main__":
    main(tyro.cli(Config))
