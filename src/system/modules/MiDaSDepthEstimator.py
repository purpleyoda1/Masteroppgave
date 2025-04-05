# src/system/modules/MIDASDepthEstimator.py
from ..SystemModule import SystemModule
from ..SystemData import SystemData

import cv2
import torch
import numpy as np
import time
from collections import deque
from statistics import mean
import os
import logging
from typing import Optional, Any, Dict, Set

class MiDaSDepthEstimator(SystemModule):
    """
    A wrapper for monocular depth estimation with MiDaS 3.0 depth estimator
    """

    def __init__(self, config: Any, module_name: str = "MiDaS_estimator"):
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        self.model = None
        self.device = None
        self.transform = None
        self.model_type = getattr(config, "midas_model_type", "DPT_Hybrid")

        self.is_initialized = False

    def initialize(self, config: Any) -> bool:
        """
        Initialize MiDaS module from Torch Hub
        """
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        self.logger.info(f"initializing {self.name}")
        try:
            if not torch:
                self.logger.error(f"PyTorch not found")
                return False
            
            # Check for GPU availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Load MiDaS model and transformer from torch hub
            try:
                self.logger.debug(f"Loading MiDaS model {self.model_type} from torch hub")
                self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
                self.model.to(self.device)
                self.model.eval()

                midas_transformers = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                    self.transform = midas_transformers.dpt_transform
                else:
                    self.transform = midas_transformers.small_transform
                
                self.logger.debug("MiDaS model and transform loaded successfully")
                self.is_initialized = True
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading MiDaS model from Torch Hub: {e}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error initializing MiDaS model: {e}")
            return False
        
    def get_required_inputs(self) -> Set[str]:
        return {SystemData.COLOR}
    
    def get_dependency_inputs(self) -> Set[str]:
        return {SystemData.COLOR}
    
    def get_outputs(self) -> Set[str]:
        return {SystemData.MIDAS_ESTIMATED_DEPTH}

    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply MiDaS depth estimation to input image"""
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        image: Optional[np.ndarray] = data.get(SystemData.COLOR)
        if image is None:
            self.logger.debug("Color image not found")
            return None
        
        try: 
            input_batch = self.transform(image).to(self.device)

            # Runf inference
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original image size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size = image.shape[:2],
                    mode = "bicubic",
                    align_corners = False
                ).squeeze()
                
            # Convert back to numpy array
            depth_map = prediction.cpu().numpy()

            return {SystemData.MIDAS_ESTIMATED_DEPTH: depth_map}
        
        except Exception as e:
            self.logger.error(f"Error running MiDaS inference: {e}")
            return None
    
    def stop(self) -> None:
        """Release resources"""
        self.logger.info(f"Stopping {self.name}...")
        if self.model is not None:
            try:
                del self.model
                self.logger.debug(f"{self.name} deleted")
            except Exception as e:
                self.logger.error(f"Error when deleting {self.name}: {e}")
            self.model = None
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("PyTorch CUDA cache cleared")
        self.is_initialized = False
        self.logger.info(f"{self.name} stopped")