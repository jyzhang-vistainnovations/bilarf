#!/usr/bin/env python3
"""
NeRF Render Server with ZIP checkpoint support

This script starts a Flask server that loads NeRF models from ZIP files
and renders views based on provided camera parameters.
"""

import os
import sys
import gin
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import traceback
from typing import Dict, Any
from flask import Flask, request, jsonify, send_file
import logging
import pickle
import io
from absl import flags
import glob
import zipfile
import tempfile
import shutil

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import necessary modules
from internal import configs
from internal import models
from internal import checkpoints
from internal import utils
from internal import camera_utils
from torch.utils._pytree import tree_map

# Create a Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload size

# Global variables for model
nerf_model = None
config = None
accelerator = None
device = "cuda" if torch.cuda.is_available() else "cpu"
current_temp_dir = None  # Track current temporary extraction directory

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("nerf_server.log")],
)
logger = logging.getLogger(__name__)

def load_config(config_path, bindings=None):
    if bindings is None:
        bindings = []
    
    gin.parse_config_files_and_bindings(
        [config_path], bindings, skip_unknown=True)
    config = configs.Config()
    
    return config

def extract_zip_checkpoint(zip_path: str) -> str:
    """Extract ZIP file and return the checkpoint directory path"""
    global current_temp_dir
    
    try:
        logger.info(f"Extracting ZIP checkpoint: {zip_path}")
        
        # Clean up previous temporary directory if it exists
        if current_temp_dir and os.path.exists(current_temp_dir):
            try:
                shutil.rmtree(current_temp_dir)
                logger.info(f"Cleaned up previous temporary directory: {current_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up previous temp dir: {str(e)}")
        
        # Create new temporary directory
        current_temp_dir = tempfile.mkdtemp(prefix="nerf_checkpoint_")
        logger.info(f"Created temporary directory: {current_temp_dir}")
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(current_temp_dir)
        
        # Look for checkpoint directory structure
        # Common patterns: checkpoints/, checkpoint/, or files directly in root
        checkpoint_dir = None
        
        # First, check if there's a 'checkpoints' or 'checkpoint' directory
        for potential_dir in ['checkpoints', 'checkpoint']:
            potential_path = os.path.join(current_temp_dir, potential_dir)
            if os.path.exists(potential_path) and os.path.isdir(potential_path):
                checkpoint_dir = potential_path
                break
        
        # If still no checkpoint directory found, use the first subdirectory or root
        if checkpoint_dir is None:
            subdirs = [d for d in os.listdir(current_temp_dir) if os.path.isdir(os.path.join(current_temp_dir, d))]
            if subdirs:
                checkpoint_dir = os.path.join(current_temp_dir, subdirs[0])
            else:
                checkpoint_dir = current_temp_dir
        
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        
        # List contents for debugging
        logger.debug(f"Checkpoint directory contents: {os.listdir(checkpoint_dir)}")
        
        return checkpoint_dir
        
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        raise Exception("Invalid ZIP file")
    except Exception as e:
        logger.error(f"Error extracting ZIP checkpoint: {str(e)}")
        raise

def load_nerf_model(checkpoint_path: str) -> bool:
    """Load a NeRF model from a checkpoint directory or ZIP file"""
    global nerf_model, config, accelerator

    try:
        logger.info(f"Loading NeRF model from: {checkpoint_path}")

        # Check if path exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return False

        # Determine if it's a ZIP file or directory
        if checkpoint_path.endswith('.zip'):
            # Extract ZIP file and get checkpoint directory
            checkpoint_dir = extract_zip_checkpoint(checkpoint_path)
        else:
            # Use directory directly
            checkpoint_dir = checkpoint_path

        # Restore the checkpoint
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        # Set dataset info based on expected config values
        dataset_info_for_model = {
            'size': 49,  # Just a placeholder, not used for rendering
        }
        
        # Create and prepare model
        nerf_model = models.Model(config=config, dataset_info=dataset_info_for_model)
        nerf_model.eval()
        nerf_model = accelerator.prepare(nerf_model)
        
        # Restore checkpoint
        step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator, logger, strict=config.load_state_dict_strict)
        logger.info(f"Successfully loaded NeRF model from checkpoint at step {step}")

        return True
    except Exception as e:
        logger.error(f"Error loading NeRF model: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def create_rays_from_camera_params(params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Create rays from the provided camera parameters, similar to generate_ray_batch"""
    width = params["width"]
    height = params["height"]
    transform = np.array(params['transform'], dtype=np.float32)
    scaling_factor = params['scaling_factor']
    
    camtoworld_list = params["camtoworld"]                                                       
    camtoworld = np.array(camtoworld_list, dtype=np.float32)

    # Ensure it's 4x4
    if camtoworld.shape == (3, 4):
        camtoworld = camera_utils.pad_poses(camtoworld)
    elif camtoworld.shape == (4, 4):
        raise ValueError(f"camtoworld must be 3x4 or 4x4 matrix, got {camtoworld.shape}")
    
    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    # camtoworld = camtoworld @ np.diag([1, -1, -1, 1])

    camtoworld = transform @ camtoworld
    camtoworld[:3, 3] *= scaling_factor
    
    # Ensure it's 3x4
    camtoworld = camtoworld[:3, :]
    
    focal_x = params["focal_x"]
    focal_y = params["focal_y"]
    cx = params["cx"]
    cy = params["cy"]
    image_downsample_fac = params["image_downsample_fac"]
    # # Create inverse intrinsic matrix (pixel to camera)
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(focal_x, focal_y, cx, cy))
    pixtocam = np.array(pixtocam, dtype=np.float32)
    pixtocam = pixtocam @ np.diag([image_downsample_fac, image_downsample_fac, 1.])
    pixtocam = np.array(pixtocam, dtype=np.float32)    
    
    # Generate pixel coordinates - output shape is (height, width)
    pix_x_int, pix_y_int = camera_utils.pixel_coordinates(width, height)

    cam_idx = 0  # Since we're just using one camera
    
    # Prepare camera parameters
    cameras = (
        pixtocam, 
        camtoworld,
        None,  # distortion_params
        None   # pixtocam_ndc
    )
    
    # Prepare pixel parameters
    near = config.near if config else 0.2
    far = config.far if config else 1000.0
    
    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.),
        'near': broadcast_scalar(near),
        'far': broadcast_scalar(far),
        'cam_idx': broadcast_scalar(cam_idx),
    }
    
    pixels = {
        'pix_x_int': pix_x_int, 
        'pix_y_int': pix_y_int, 
        **ray_kwargs
    }
    
    # Use camera_utils to cast rays
    batch = camera_utils.cast_ray_batch(cameras, pixels, camera_utils.ProjectionType.PERSPECTIVE)
    
    # Add cam_dirs (needed by model) - FIX: broadcast to the right shape
    cam_dirs = -camtoworld[:3, 2]  # Extract camera forward direction
    batch['cam_dirs'] = np.broadcast_to(cam_dirs, pix_x_int.shape + (3,))
    
    # Add normalized pixel coordinates
    pix_x_float = (pix_x_int.astype(np.float32) + 0.5) / width
    pix_y_float = (pix_y_int.astype(np.float32) + 0.5) / height
    batch['pix_xy'] = np.stack([pix_x_float, pix_y_float], axis=-1)
    print(batch['pix_xy'])
    
    # Convert to torch tensors
    batch = {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    for k, v in batch.items():
        if v is not None:
            logger.debug(f"Ray batch tensor '{k}' has shape {v.shape}")
    
    return batch


def render_image_from_rays(ray_batch, height, width, config, nerf_model, accelerator):
    """A simpler wrapper around models.render_image that adds extra debugging"""
    # Move batch to device 
    ray_batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, ray_batch)
    
    # Call render_image with the flattened batch
    with torch.no_grad():
        rendering = models.render_image(
            nerf_model, 
            accelerator, 
            ray_batch,
            rand=False,
            train_frac=1.0,
            config=config,
            verbose=True
        )
    
    return rendering


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if nerf_model is not None else "not loaded"
    return jsonify({"status": "ok", "device": device, "model": model_status})


@app.route("/load_model", methods=["POST"])
def load_model():
    """Load a NeRF model from a checkpoint directory or ZIP file"""
    logger.info("Received load_model request")
    
    try:
        data = request.json
        logger.debug(f"Request data: {data}")
        
        checkpoint_path = data.get("checkpoint_path")
        config_path = data.get("config_path", None)
        
        if not checkpoint_path:
            logger.error("Missing checkpoint_path parameter")
            return jsonify({"error": "checkpoint_path is required"}), 400
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return jsonify({"error": f"Checkpoint path not found: {checkpoint_path}"}), 404
        
        # Load the model (handles both ZIP and directory)
        success = load_nerf_model(checkpoint_path)
        if success:
            logger.info("NeRF model loaded successfully")
            return jsonify({"status": "NeRF model loaded successfully"})
        else:
            logger.error("Failed to load NeRF model")
            return jsonify({"error": "Failed to load NeRF model"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in load_model endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/render", methods=["POST"])
def render_image():
    """Render an image with provided camera parameters"""
    logger.info("Received render request")
    
    if nerf_model is None:
        logger.error("NeRF model not initialized")
        return jsonify({"error": "No NeRF model loaded. Call /load_model first."}), 400
    
    try:
        data = request.json
        logger.debug(f"Request data: {data}")
        
        camera_params = data.get("camera")
        
        if not camera_params:
            logger.error("Missing camera parameters")
            return jsonify({"error": "camera parameters are required"}), 400
        
        # Extract width and height for later use
        width = camera_params["width"]
        height = camera_params["height"]
        
        # Create rays from camera parameters
        logger.info("Creating rays from camera parameters")
        ray_batch = create_rays_from_camera_params(camera_params)
        
        # Render the view
        logger.info("Rendering view")
        try:
            rendering = render_image_from_rays(ray_batch, height, width, config, nerf_model, accelerator)
            
            # Convert to numpy
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x, rendering)
            
            # Process results
            result = {}
            if 'rgb' in rendering:
                result["image"] = rendering['rgb']
            
            if 'distance_mean' in rendering:
                result["depth_mean"] = rendering['distance_mean']
            
            if 'distance_median' in rendering:
                result["depth_median"] = rendering['distance_median']
            
            if 'normals' in rendering:
                result["normals"] = rendering['normals']
            
            if 'acc' in rendering:
                result["acc"] = rendering['acc']
                
            logger.info("Rendering completed successfully")
            
            # Pickle the result dictionary
            logger.info("Pickling render result")
            pickled_data = pickle.dumps(result)
            
            # Return the pickled data as a file
            logger.info(f"Sending pickled data ({len(pickled_data)} bytes)")
            return send_file(
                io.BytesIO(pickled_data),
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name="render_result.pkl",
            )
            
        except Exception as e:
            logger.error(f"Error during rendering: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Rendering error: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error processing render request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def cleanup_temp_directories():
    """Clean up temporary directories on server shutdown"""
    global current_temp_dir
    if current_temp_dir and os.path.exists(current_temp_dir):
        try:
            shutil.rmtree(current_temp_dir)
            logger.info(f"Cleaned up temporary directory: {current_temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")


if __name__ == "__main__":
    import argparse
    import atexit
    
    # Register cleanup function
    atexit.register(cleanup_temp_directories)
    
    parser = argparse.ArgumentParser(description="NeRF Render Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory or ZIP file")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    
    args = parser.parse_args()
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Log current directory and paths
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"sys.path: {sys.path}")
    
    # Load config
    try:
        config_path = args.config if args.config else "configs/train.gin"
        logger.info(f"Loading config: {config_path}")
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Load checkpoint if provided
    if args.checkpoint:
        try:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            if not os.path.exists(args.checkpoint):
                logger.error(f"Checkpoint path does not exist: {args.checkpoint}")
                sys.exit(1)
            
            # Load the model (handles both ZIP and directory)
            success = load_nerf_model(args.checkpoint)
            if not success:
                logger.warning("Failed to load checkpoint, but server will start")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("Server will start, but model is not initialized")
    load_nerf_model("exp/test_apr29_1/checkpoints")
    # Start the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup_temp_directories()