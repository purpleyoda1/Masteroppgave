# src/system/robot/DetectionToTransformationModule

from ..SystemModule import SystemModule
from ..SystemData import SystemData

import numpy as np
import logging
from typing import Optional, Dict, Any, Set, List


class DetectionToTransformationModule(SystemModule):
    """
    Process list of Detections to calculate their 3D postion in the camera frame
    """
    def __init__(self,
                 config: Any,
                 module_name: str,
                 stream_map: Dict[str, str]):
        """Initialize module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(module_name)

        self._stream_map = stream_map

        self._max_depth_m = getattr(config, "DTT_max_depth_m", 3.0)
        self._calc_area_pixels = getattr(config, "DTT_calc_area_pixels") 