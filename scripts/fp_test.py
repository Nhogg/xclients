import asyncio
import websockets
import os
import glob
import numpy as np
import msgpack
import msgpack_numpy
from PIL import Image

# Patch msgpack to handle numpy arrays
msgpack_numpy.patch()

# --- CONFIGURATION ---
DATA_ROOT = "/home/nhogg/FP/FoundationPose/my_data"
RGB_DIR   = os.path.join(DATA_ROOT, "rgb")
MESH_DIR  = os.path.join(DATA_ROOT, "mesh")
GENERATED_MASK_PATH = os.path.join(DATA_ROOT, "dummy_mask.npz")
URI = "ws://localhost:8082"

def setup_files():
    print("Setting up data...")

    # 1. Find the first image to determine dimensions
    images = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    if not images:
        print(f"Error: No PNG images found in {RGB_DIR}")
        return None
    
    first_img_path = images[0]
    
    # 2. Generate a Dummy Mask (Full Image)
    if not os.path.exists(GENERATED_MASK_PATH):
        print("No mask found. Generating a dummy full-frame mask...")
        img = Image.open(first_img_path)
        w, h = img.size
        
        # Create a mask of all 1s (uint8)
        dummy_mask = np.ones((h, w), dtype=np.uint8)
        
        # Save as .npz with 'masks' key
        np.savez(GENERATED_MASK_PATH, masks=dummy_mask)
        print(f"Created dummy mask at: {GENERATED_MASK_PATH}")
    else:
        print(f"Using existing mask: {GENERATED_MASK_PATH}")

    # 3. Find the Mesh
    if os.path.exists(os.path.join(MESH_DIR, "broom.obj")):
        mesh_path = os.path.join(MESH_DIR, "broom.obj")
    elif os.path.exists(os.path.join(MESH_DIR, "broom_meters.obj")):
        mesh_path = os.path.join(MESH_DIR, "broom_meters.obj")
    else:
        objs = glob.glob(os.path.join(MESH_DIR, "*.obj"))
        if not objs:
            print(f"Error: No .obj file found in {MESH_DIR}")
            return None
        mesh_path = objs[0]
    
    print(f"Using mesh: {os.path.basename(mesh_path)}")

    return {
        "rgb_dir": RGB_DIR,
        "mesh_path": mesh_path,
        "mask_path": GENERATED_MASK_PATH
    }


async def run_test():
    paths = setup_files()
    if not paths: return

    payload = {
        "type": "track",
        "rgb_dir": paths['rgb_dir'],
        "mask_path": paths['mask_path'],
        "mesh_path": paths['mesh_path'],
        "confidence": 0.5,
        "debug": True
    }

    print(f"Connecting to {URI}...")
    try:
        async with websockets.connect(URI) as websocket:
            print("Sending Track Request...")
            
            # Send as Binary (MessagePack)
            data = msgpack.packb(payload, use_bin_type=True)
            await websocket.send(data)
            
            print("Waiting for FoundationPose (loading model)...")
            response_raw = await websocket.recv()
            
            # Decode response
            response = msgpack.unpackb(response_raw, raw=False)
            
            print("RESPONSE:")
            print(f"Status: {response.get('status')}")
            if 'initial_pose' in response:
                print(f"Initial Pose:\n{response['initial_pose']}")
            else:
                print(response)

    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
