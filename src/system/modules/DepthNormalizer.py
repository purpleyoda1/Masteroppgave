# src/system/module/DepthNormalizer.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData

import numpy as np
import cv2
import logging
from typing import Optional, Dict, Any, Set

class DepthNormalizer(SystemModule):
    """
    Normalize depth maps from any source to standardized range and data type
    """
    def __init__(self,
                 config: Any,
                 module_name: str = "Depth_Normalizer",
                 input_stream_type: str = SystemData.DEPTH,
                 output_stream_type: str = SystemData.NORM_DEPTH,
                 invert: bool = False,
                 target_min: float = 0.0,
                 target_max: float = 255.0,
                 output_dtype: np.dtype = np.uint8):
        """Initialize normalizer"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        self._input_stream_type = input_stream_type
        self._output_stream_type = output_stream_type
        self._invert = invert
        self._target_min = getattr(self._config, "norm_target_min", 0.0)
        self._target_max = getattr(self._config, "norm_target_max", 255.0)
        self._output_dtype = getattr(self._config, "norm_output_dtype", np.uint8)
        self._range = target_max - target_min

        self.is_initialized = False

    def initialize(self, config: Any) -> bool:
        """Higher level init for sytem structure"""
        self.is_initialized = True
        return True
    
    def get_required_inputs(self) -> Set[str]:
        return {self._input_stream_type}
    
    def get_dependency_inputs(self) -> Set[str]:
        return {self._input_stream_type}
    
    def get_outputs(self) -> Set[str]:
        return {self._output_stream_type}
    
    def _process_internal(self, data) -> Optional[Dict[str, Any]]:
        """Apply normalization logic"""
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not intialized")
            return None
        
        input_depth = data.get(self._input_stream_type)
        if input_depth is None:
            self.logger.debug(f"Input {self._input_stream_type} not found in current data")
            return None
        
        try:
            input_float = input_depth.astype(np.float32)

            # Find min and max values
            min_val = np.min(input_float)
            max_val = np.max(input_float)
            range_val = max_val - min_val

            # Avoid division by 0
            if range_val < 1e-6:
                normalized_01 = np.full_like(input_float, self._target_min)
            else:
                normalized_01 = (input_float - min_val) / range_val

                # Invert if necesarry
                if self._invert:
                    normalized_01 = 1.0 - normalized_01

                # Scale to desired range
                normalized_depth = normalized_01 * self._range + self._target_min

            # Clip to ensure data is within range, then ensure datatype
            np.clip(normalized_depth, self._target_min, self._target_max, out=normalized_depth)
            output = normalized_depth.astype(self._output_dtype)

            return {self._output_stream_type: output}
        
        except Exception as e:
            self.logger.error(f"Error applying normalization: {e}")
            return None
        
    def stop(self) -> None:
        self.is_initialized = False
        self.logger.info(f"Normalizer stopped")


        

