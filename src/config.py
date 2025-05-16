# src/config.py

from dataclasses import dataclass, field
import os
from typing import Dict, Tuple, List, Any
import numpy as np
from system.SystemData import SystemData

@dataclass
class Config:
    #------------------------------------------#
    #            Camera settings               #
    #------------------------------------------#
    depth_res = 640
    color_res = 640
    depth_res = 640
    color_res = 640

    if depth_res == 424:
        depth_stream_width: int = 424
        depth_stream_height: int = 240
    if depth_res == 640:
        depth_stream_width: int = 640   
        depth_stream_height: int = 480
    elif depth_res == 640:
        depth_stream_width: int = 640
        depth_stream_height: int = 480
    depth_stream_fps: int = 6

    if color_res == 424:
        color_stream_width: int = 424
        color_stream_height: int = 240
    elif color_res == 848:
        color_stream_width: int = 848
        color_stream_height: int = 480
    elif color_res == 640:
        color_stream_width: int = 640
        color_stream_height: int = 480
    color_stream_fps: int = 15

    realsense_enable_imu = False
    imu_accel_fps = 250
    imu_gyro_fps = 200
    imu_print_interval = 1.0

    #------------------------------------------#
    #        System Controller settings        #
    #------------------------------------------#
    module_active_states: Dict[str, bool] = field(default_factory=lambda: {
        # Camera
        SystemData.CAMERA_NAME: True,
        # Estimator
        SystemData.MIDAS_NAME: True,
        SystemData.PRO_NAME: False,
        # Normalizers
        SystemData.NORM_NAME: True,
        # YOLO Detectors
        SystemData.YOLO_DEPTH_NAME: False,
        SystemData.YOLO_NORM_NAME: True,
        # Detection tracker
        SystemData.TRACKER_NAME: True,
        # Visualization
        SystemData.VIS_NAME: True,
        # Frame Saver
        SystemData.SAVE_NAME: True,
    })

    # List of configurations fpor system modules, which will be initialized
    yolo_configurations: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            'name': SystemData.YOLO_NORM_NAME,
            'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src\\model_training\\runs\\detect\\640_norm_real_L\\weights\\best.pt"),
            'stream_map': {
                SystemData.NORM_MIDAS: SystemData.MIDAS_DETECTIONS,
                SystemData.NORM_PRO: SystemData.PRO_DETECTIONS
            }
        }
    ])

    normalizer_stream_map: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        SystemData.DEPTH: {SystemData.NORM_DEPTH: False},
        SystemData.MIDAS_ESTIMATED_DEPTH: {SystemData.NORM_MIDAS: True},
        SystemData.PRO_ESTIMATED_DEPTH: {SystemData.NORM_PRO: False},
    })

    tracker_stream_map: Dict[str, str] = field(default_factory=lambda: {
        SystemData.DEPTH_DETECTIONS: SystemData.TRACKED_DEPTH_DETECTIONS,
        SystemData.MIDAS_DETECTIONS: SystemData.TRACKED_MIDAS_DETECTIONS,
        SystemData.PRO_DETECTIONS: SystemData.TRACKED_PRO_DETECTIONS,
    })

    visualizer_initial_montage: List[str] = field(default_factory=lambda: [
        SystemData.COLOR,
        SystemData.DEPTH
    ])

    # YOLO model
    confidence_threshold: float = 0.6
    iou_treshold: float = 0.5
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'Capacitor',
        1: 'Bracket',
        2: 'Screw'
    })
    @property
    def yolo_model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "src\\model_training\\runs\\detect\\v11small\\best.pt"
        return os.path.join(base_path, relative_path)
    
    @property
    def yolo_normalized_model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "src\\model_training\\runs\\detect\\640_norm_real_L\\weights\\best.pt"
        return os.path.join(base_path, relative_path)
    
    # MiDaS model
    midas_model_type: str = "DPT_Hybrid"  # DPT_Large or DPT_Hybrid

    # Depth Pro
    depth_pro_rel_weight_path: str = "external/depth-pro/checkpoints/depth_pro.pt"

    # BYTETracker
    byte_track_tresh = 0.45
    byte_track_buffer = 25
    byte_match_tresh = 0.8
    byte_frame_rate = 5

    # Detection to transformation
    DTT_max_depth_m = 3.0           # Max registered depth
    DTT_calc_area_pixels = 10       # Side length of area around centroid for depth calcualtion

    # Normalization
    norm_target_min: float = 0.0
    norm_target_max: float = 255.0
    norm_dtype: np.dtype = np.uint8

    # Visualizer
    vis_view_target_height: int = 480
    vis_canvas_height: int = 960
    vis_canvas_width: int = 1280
    vis_apply_colormap: bool = False
    vis_class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        'Capacitor': (255, 0, 0),
        'Bracket': (0, 255, 0),
        'Screw': (0, 0, 255)
    })
    vis_default_detection_color: Tuple[int, int, int] = (255, 0, 200)
    vis_initial_active_views = {SystemData.COLOR}

    # Frame saver
    @property
    def save_base_dir(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "training_data/real_world"
        return os.path.join(base_path, relative_path)
    streams_to_save = [
        SystemData.COLOR,
        SystemData.DEPTH,
        SystemData.MIDAS_ESTIMATED_DEPTH,
        SystemData.NORM_MIDAS,
        SystemData.PRO_ESTIMATED_DEPTH,
        SystemData.NORM_PRO
    ]