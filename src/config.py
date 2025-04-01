# src/config.py

from dataclasses import dataclass
import os

@dataclass
class Config:
    # Realsense camera
    depth_stream_width: int = 424
    depth_stream_height: int = 240
    depth_stream_fps: int = 6
    color_stream_width: int = 424
    color_stream_height: int = 240
    color_stream_fps: int = 15

    # YOLO model
    confidence_threshold: float = 0.8
    iou_treshold: float = 0.5
    @property
    def yolo_model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "src\\model_training\\runs\\detect\\v11small\\best.pt"
        return os.path.join(base_path, relative_path)
    
    @property
    def yolo_midas_model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "src\\model_training\\runs\\detect\\MiDaS\\weights\\best.pt"
        return os.path.join(base_path, relative_path)
    
    @property
    def yolo_pro_model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "src\\model_training\\runs\\detect\\raw_2003\\weights\\best.pt"
        return os.path.join(base_path, relative_path)
    
    # Directories
    @property
    def save_dir(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "saved_frames/test"
        return os.path.join(base_path, relative_path)
    
    # MiDaS model
    midas_model_type: str = "DPT_Hybrid"

    # Depth Pro
    depth_pro_rel_weight_path: str = "external/depth-pro/checkpoints/depth_pro.pt"


