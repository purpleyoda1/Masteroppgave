# src/modules/DepthProEstimator.py
import sys
sys.path.append("external/depth-pro/src/depth_pro")

import cv2
import os
import time
import torch
import numpy as np
import logging
from collections import deque
from statistics import mean
from typing import Optional, Tuple, Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))

# Add the depth_pro source directory to the path
depth_pro_src = os.path.join(root_dir, "external/depth-pro/src/depth_pro")
#sys.path.append(depth_pro_src)


import torch
from depth_pro import create_model_and_transforms, load_rgb

class DepthProEstimator:
    """
    Wrapepr for Apples Depth Pro estimator
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DepthProEstimator")

        self.model = None
        self.transfrom = None
        self.device = None
        self.optimize_inference = getattr(config, "depth_pro_optimize_inference", True)
        self.normalize_output = getattr(config, "depth_pro_normalize_output", False)
        self.inference_times = deque(maxlen=30)

    def initialize(self) -> bool:
        """
        Initialize Depth Pro module
        """
        try:
            self.logger.info("Initializing Depth Pro estimator...")

            # Check installations
            try:
                import torch
                import depth_pro
            except ImportError:
                self.logger.error("Pytorch or Depth Pro not propperly installed")
                return False
            
            # Check GPU availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Change weigths location
            checkpoint_path = os.path.abspath(os.path.join(root_dir, "external/depth-pro/checkpoints/depth_pro.pt"))
            self.logger.debug(f"Depth pro weights should be at: {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Depth pro weights not found at: {checkpoint_path}")
                return False
            
            from depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            config = DEFAULT_MONODEPTH_CONFIG_DICT
            config.checkpoint_uri = checkpoint_path


            # Load Depth Pro model
            try:
                self.logger.debug("Starting create_model_and_transforms")
                self.model, self.transform = depth_pro.create_model_and_transforms(config=config)
                self.logger.debug("Starting .to(self.device)")
                self.model.to(self.device)
                self.logger.debug("Starting .eval()")
                self.model.eval()
                self.logger.info("Depth Pro loaded sucessfully")
                return True
            except Exception as e:
                self.logger.error(f"Error loading Depth Pro: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error initializing Depth Pro: {e}")
            return False
            
    def estimate_depth(self, image: np.ndarray, focal_length_px= None, normalize: bool = None) -> Optional[np.ndarray]:
        """
        Apply depth estimation to input image
        """
        if self.model is None:
            self.logger.error("Depth Pro module not initialized")
            return None
        
        # to normalize or not to normalize
        shakespear = self.normalize_output if normalize is None else normalize

        try:
            start_time = time.time()

            # Transform PIL version of input image
            import PIL.Image
            pil_image = PIL.Image.fromarray(image)
            image_tensor = self.transform(pil_image)
            image_tensor = image_tensor.to(self.device)

            if focal_length_px is not None:
                if isinstance(focal_length_px, (int, float)):
                    f_px = torch.tensor([float(focal_length_px)], device=self.device)
                else:
                    f_px = focal_length_px.to(self.device)
            else:
                f_px = None

            # Inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor, f_px=f_px)
                depth_map = prediction["depth"]

                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()

                # Resize to original image size
                if depth_map.shape[:2] != image.shape[:2]:
                    depth_map = cv2.resize(
                        depth_map,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_CUBIC
                    )

            elapsed_time = time.time() - start_time
            self.inference_times.append(elapsed_time)

            # Normalize if enabled
            if shakespear:
                depth_min = depth_map.min()
                depth_max = depth_map.max()

                if depth_max - depth_min > 0:
                    depth_map = 65535 * (depth_map - depth_min) / (depth_max - depth_min)
                
                depth_map = depth_map.astype(np.uint16)
            
            return depth_map

        except Exception as e:
            self.logger.error(f"Error running inference: {e}")
            return None
        
    def estimate_depth_with_metadata(self, image: np.ndarray, normalize: bool = None) -> Optional[np.ndarray]:
        """
        Apply depth estimation to input image, and return all data from prediction
        """
        if self.model is None:
            self.logger.error("Depth Pro module not initialized")
            return None
        
        # to normalize or not to normalize
        shakespear = self.normalize_output if normalize is None else normalize

        try:
            start_time = time.time()

            import PIL.Image
            pil_image = PIL.Image.fromarray(image)

            # Load image
            import depth_pro
            image_tensor, _, f_px = depth_pro.load_rgb(image)
            image_tensor = self.transform(image_tensor)

            # Inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor, f_px=f_px)
                depth_map = prediction["depth"]

                result = {}
                for key, value in prediction.items():
                    if isinstance(value, torch.Tensor):
                        result[key] = value.cpu().numpy()
                    else:
                        result[key] = value

                # Resize to original image size
                if result["depth"].shape[:2] != image.shape[:2]:
                    result["depth"] = cv2.resize(
                        result["depth"],
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_CUBIC
                    )

            elapsed_time = time.time() - start_time
            self.inference_times.append(elapsed_time)
            result["inference_time"] = elapsed_time

            # Normalize if enabled
            if shakespear:
                depth_min = result["depth"].min()
                depth_max = result["depth"].max()

                if result["depth"] - depth_min > 0:
                    result["depth"] = 65535 * (result["depth"] - depth_min) / (depth_max - depth_min)
                
                result["depth"] = result["depth"].astype(np.uint16)
            
            return result

        except Exception as e:
            self.logger.error(f"Error running inference: {e}")
            return None
        
    def get_average_inference_time(self) -> float:
        """
        Return average inference time
        """
        if not self.inference_times:
            return 0
        else:
            return mean(self.inference_times)