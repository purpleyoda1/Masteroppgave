# src/system/modules/DepthProEstimator.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData

import cv2
import os
import torch
from depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
import PIL.Image
import numpy as np
import logging
from typing import Optional, Dict, Any, Set

class DepthProEstimator(SystemModule):
    """Wrapper for Apples Depth Pro depth estimator"""
    def __init__(self, config, module_name: str = "DepthPro_Estimator"):
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)
        self.model = None
        self.transfrom = None
        self.device = None
        self.normalize_output = getattr(config, "depth_pro_normalize_output", False)
        self.depth_pro_rel_weight_path = getattr(config, "depth_pro_rel_weight_path", "external/depth-pro/checkpoints/depth_pro.pt")
        self.is_initialized = False

    def initialize(self, config: Any) -> bool:
        """
        Initialize Depth Pro module
        """
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True

        try:
            self.logger.info(f"Initializing {self.name}...")
            
            # Check GPU availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Change weigths location
            project_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
            weigths_path = os.path.join(project_root_dir, self.depth_pro_rel_weight_path)
            self.logger.debug(f"Looking for Depth pro weights at: {weigths_path}")

            if not os.path.exists(weigths_path):
                self.logger.error(f"Depth pro weights not found at: {weigths_path}")
                return False
            
            depth_pro_config = DEFAULT_MONODEPTH_CONFIG_DICT
            depth_pro_config.checkpoint_uri = weigths_path

            # Load Depth Pro model
            self.logger.debug("Loading Depth Pro model and transforms...")
            self.model, self.transform = create_model_and_transforms(config=depth_pro_config)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Depth Pro initialized sucessfully")
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Depth Pro: {e}")
            return False

    def get_required_inputs(self) -> Set[str]:
        return {SystemData.COLOR, SystemData.COLOR_INTRINSICS}
    
    def get_dependency_inputs(self) -> Set[str]:
        return {SystemData.COLOR, SystemData.COLOR_INTRINSICS}
    
    def get_outputs(self) -> Set[str]:
        return {SystemData.PRO_ESTIMATED_DEPTH}
    
    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core implementation of depth estimation
        """
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        image: Optional[np.ndarray] = data.get(SystemData.COLOR)
        if image is None:
            self.logger.debug("Color image not found")
            return None
        
        try:
            # Get camera focal length to enhance model performance
            focal_length_fx = data.get(SystemData.COLOR_INTRINSICS).fx if data.get(SystemData.COLOR_INTRINSICS) else None

            # Convert numpy image to PIL tensor
            pil_image = PIL.Image.fromarray(image)
            image_tensor = self.transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Prepare focal length tensor
            f_px_tensor = None
            if focal_length_fx is not None:
                try:
                    f_px_tensor = torch.tensor([float(focal_length_fx)], device=self.device)
                except Exception as e:
                    self.logger.error(f"Error processing focal length: {e}")
            
            # Run inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor, f_px=f_px_tensor)
                depth_map = prediction["depth"]

                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            
            if depth_map.shape[:2] != image.shape[:2]:
                depth_map = cv2.resize(
                    depth_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

            # Apply normalization if needed
            if self.normalize_output:
                depth_min = depth_map.min()
                depth_max = depth_map.max()

                if depth_max - depth_min > 1e-6:
                    depth_map = 65535 * (depth_map - depth_min) / (depth_max - depth_min)
                    depth_map = depth_map.astype(np.uint16)
            
            return {SystemData.PRO_ESTIMATED_DEPTH: depth_map}

        except Exception as e:
            self.logger.error(f"Error in {self.name} processing: {e}")
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