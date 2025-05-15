from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import logging

detection_logger = logging.getLogger(f"DetectionClass")

@dataclass
class Detection:
    """
    Unified detection class across different methods
    """

    # Core properties
    class_id: int
    label: str
    conf: float
    source: str      # Which detector created it
    dimension: str   # 2D or 3D
    track_id: Optional[str]
    timestamp: float = field(default_factory=time.time)

    # Dimension dependent data
    bbox2D: Optional[List[float]] = None    # [x1, y1, x2, y2]
    bbox3D: Optional[List[float]] = None    # [x, y, z, width, height, depth, roll, pitch, yaw]
    center3D: Optional[List[float]] = None  # [x, y, z]
    position_camera_frame: Optional[List[float]] = None

    # If addinional metadata is needed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_center_2D(self) -> Optional[List[float]]:
        """Get the center of a 2D detection"""
        if self.bbox2D:
            x1, y1, x2, y2 = self.bbox2D
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        return None
    
    def get_area_2D(self) -> float:
        """Get area of 2D detection"""
        if self.bbox2D:
            x1, y1, x2, y2 = self.bbox2D
            return (x2 - x1) * (y2 - y1)
        return None
    
    def to_bytetrack_format(self) -> Optional[List[float]]:
        """
        Returns detection on a format compatible with ByteTrack:
        [x1, y1, x2, y2, score, class_id]
        """
        if (self.bbox2D is not None and
            len(self.bbox2D) == 4 and
            self.conf is not None and
            self.class_id is not None):
            try:
                x1, y1, x2, y2 = map(float, self.bbox2D)
                score = float(self.conf)
                cls_id = float(self.class_id)

                bytetrack_input = [x1, y1, x2, y2, score, cls_id]
                return bytetrack_input
            
            except Exception as e:
                detection_logger.error(f"Error converting detection to ByteTrack format: {e}")
                return None
            
        else:
            detection_logger.error(f"Missing data for bytetrack format")
            return None