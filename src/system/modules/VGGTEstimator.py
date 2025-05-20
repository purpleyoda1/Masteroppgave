# src/system/modules/VGGTEstimator.py

from ..SystemData import SystemData
from ..SystemModule import SystemModule

import logging
import torch
from typing import Any, Set, Dict, Optional
import numpy as np
import torchvision.transforms.functional as TF

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

class VGGTEstimator(SystemModule):
    """Wrapper for Metas VGGT 3D reconstructor"""
    def __init__(self, config, module_name):
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        self._device = None
        self._model = None
        self._dtype = None

        self._is_initialized = False

    def initialize(self, config) -> bool:
        """
        Initialize module
        """
        if self._is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        try: 
            self.logger.debug(f"Initializing {self.name}...")
            
            # Check GPU availability
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self._device}")

            # Set dtype for model
            self._dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

            # Load model
            self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(self._device)

            self.logger.info("VGGT initialized sucessfully")
            self._is_initialized = True
            return True
        
        except Exception as e:
            self.logger.error(f"Error initializing VGGT: {e}")

    def get_required_inputs(self) -> Set[str]:
        return {SystemData.COLOR}
    
    def get_dependency_inputs(self) -> Set[str]:
        return {SystemData.COLOR}
    
    def get_outputs(self) -> Set[str]:
        return {SystemData.VGGT_ESTIMATED_DEPTH}
    
    def _process_internal(self, data: Dict[str, any]) -> Optional[Dict[str, Any]]:
        """Apply VGGT to input"""
        if not self._is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        try:
             #TODO: change load_and_preprocess, maybe convert to PIL here as with depthpro
            image_ndarray: Optional[np.ndarray] = data.get(SystemData.COLOR)
            image, resize_context = load_and_preprocess_images(image_ndarray)

            image_GPU = image.to(self._device)
            image_5d = image_GPU.unsqueeze(1)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    # Prepare input tokes to only use depth head
                    aggregated_tokens_list, ps_idx = self._model.aggregator(image_5d)
                
                depth_map_tensor, _ = self._model.depth_head(aggregated_tokens_list, 
                                                      images=image_5d, 
                                                      patch_start_idx=ps_idx)
                
            depth_map = self._reverse_padding(depth_map_tensor, resize_context)

            return {SystemData.VGGT_ESTIMATED_DEPTH: depth_map} 
        
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
        self._is_initialized = False
        self.logger.info(f"{self.name} stopped")

    def _reverse_padding(self,
            depth_map_tensor: torch.Tensor,
            resize_context
    ):
        """Reverse VGGT modification of image"""
        depth_map_squeezed = depth_map_tensor.squeeze()
        
        h_tensor, w_tensor = depth_map_squeezed.shape
        hw_array = resize_context['original_size']
        mode = resize_context['mode']
        depth_map_content: torch.Tensor

        if mode == "pad":
            pad_top, pad_bottom, pad_left, pad_right = resize_context['padding']
            content_h = h_tensor - pad_top - pad_bottom
            content_w = w_tensor - pad_left - pad_right

            depth_map_content = depth_map_squeezed[pad_top : h_tensor - pad_bottom, pad_left: w_tensor - pad_right]

        elif mode == "crop":
            depth_map_content = depth_map_squeezed

        depth_map_for_resize = depth_map_content.unsqueeze(0)
        depth_map_resized = TF.resize(
            depth_map_for_resize,
            hw_array,
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False
        )

        final_depth_tensor = depth_map_resized.squeeze(0)
        depth_map = final_depth_tensor.cpu().numpy()

        return depth_map