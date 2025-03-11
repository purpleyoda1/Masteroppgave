from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

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
    timestamp: float = field(default_factory=time.time)

    # Dimension dependent data
    bbox2D: Optional[List[float]] = None    # [x1, y1, x2, y2]
    bbox3D: Optional[List[float]] = None    # [x, y, z, width, height, depth, roll, pitch, yaw]
    center3D: Optional[List[float]] = None  # [x, y, z]
    point_indices: Optional[List[int]] = None

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