# src/system/modules/VGGTEstimator.py

from ..SystemData import SystemData
from ..SystemModule import SystemModule



import logging
import torch
from typing import Any, Set, Dict, Optional
import numpy as np

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
        if self.is_initialized:
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
            image = load_and_preprocess_images(image_ndarray)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    # Prepare input tokes to only use depth head
                    aggregated_tokens_list, ps_idx = self._model.aggregator(image)
                
                depth_map, _ = self._model.depth_head(aggregated_tokens_list, ps_idx)

            return {SystemData.VGGT_ESTIMATED_DEPTH: depth_map} 
        
        except Exception as e:
            self.logger.error(f"Error in {self.name} processing: {e}")
            return None