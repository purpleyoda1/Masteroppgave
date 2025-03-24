# src/modules/MIDASDepthEstimator.py

import cv2
import torch
import numpy as np
import time
from collections import deque
from statistics import mean
import os
import logging
from typing import Optional

# add midas to python path
import sys
sys.path.append("external/midas")

class MiDaSDepthEstimator:
    """
    A wrapper fo monocular depth estimation using MiDaS v3.1
    """

    def __init__(self, config):
        """
        Initialize model
        """
        self.config = config
        self.logger = logging.getLogger("MIDASDepthEstimator")

        # Set MiDaS properties
        self.model = None
        self.device = None
        self.transform = None
        self.model_type = getattr(config, "midas_model_type", "DPT_Hybrid")
        self.optimize_inference = getattr(config, "midas_optimize_inference", True)
        self.inference_times = deque(maxlen=30)

    def initialize(self) -> bool:
        """
        Initialize model with given configurations
        """
        try:
            self.logger.info("Initializing MiDaS depth estimator...")

            # Check if torch is available
            try:
                import torch
            except ImportError:
                self.logger.error("PyTorch not installed")

            # Check for GPU availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Load MiDaS from torch hub
            try:
                self.logger.info(f"Loading MiDaS model {self.model_type} from torch hub")
                self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
                self.model.to(self.device)
                self.model.eval()

                # Load transformers, and select based on model
                midas_transformers = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                    self.transform = midas_transformers.dpt_transform
                else:
                    self.transform = midas_transformers.small_transform

                self.logger.info("MiDaS model and transform loaded successfully")
                return True
            
            except Exception as e:
                self.logger.error(f"Error loading from torch hub: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error initializing MiDaS model: {e}")
            return False
        
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from color image
        """
        if self.model is None:
            self.logger.error(f"MiDaS model not initialized")
            return None
        
        try:
            start_time = time.time()

            input_batch = self.transform(image).to(self.device)

            # Perform inference
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original image size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size = image.shape[:2],
                    mode = "bicubic",
                    align_corners = False
                ).squeeze()

                elapsed_time = time.time() - start_time
                self.inference_times.append(elapsed_time)

                # To numpy array
                depth_map = prediction.cpu().numpy()
                
                # Normalize
                depth_min = depth_map.min()
                depth_max = depth_map.max()

                if depth_max - depth_min > 0:
                    depth_map = 65535 * (depth_map - depth_min) / (depth_max - depth_min)
                
                depth_map = depth_map.astype(np.uint16)

                return depth_map
            
        except Exception as e:
            self.logger.error(f"Error running inference: {e}")
            return None
        
    def get_average_inference_time(self) -> float:
        """
        
        """
        if not self.inference_times:
            return 0
        else:
            return mean(self.inference_times)
        
    def get_colormap(self, depth_map: np.ndarray, colormap_type: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
        """
        Convert monochrome depth map to colorized depthmap fpr visualization
        """
        if depth_map is None:
            return None
        
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, colormap_type)

        return depth_colormap
