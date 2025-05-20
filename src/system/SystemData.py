# src/system/SystemData.py

class SystemData:
    """Types of data available in system"""
    # Module names
    CAMERA_NAME = "RealSense_Camera"
    MIDAS_NAME = "MiDaS_Estimator"
    PRO_NAME = "DepthPro_Estimator"
    VGGT_NAME = "VGGT_Estimator"
    NORM_NAME = "NormalizerModule"
    YOLO_DEPTH_NAME = "YOLO_Depth"
    YOLO_NORM_NAME = "YOLO_Normalized"
    TRACKER_NAME = "TrackingModule"
    VIS_NAME = "VisualizationModule"
    SAVE_NAME = "FrameSaver"

    # Camera
    COLOR = "color"
    DEPTH = "depth"
    COLOR_INTRINSICS = "color_intrincsics"
    DEPTH_INTRINSICS = "depth_intrinsics"
    CAMERA_DEPTH_SCALE = "camera_depth_scale"

    # MiDaS
    MIDAS_ESTIMATED_DEPTH = "midas_estimated_depth"

    # DepthPro
    PRO_ESTIMATED_DEPTH = "pro_estimated_depth"

    # VGGT
    VGGT_ESTIMATED_DEPTH = "vggt_estimated_depth"

    # YOLO
    DEPTH_DETECTIONS = "depth_detections"
    MIDAS_DETECTIONS = "midas_detections"
    PRO_DETECTIONS = "pro_detections"
    VGGT_DETECTIONS = "vggt_detections"
    
    # Normalizer
    NORM_DEPTH = "norm_depth"
    NORM_MIDAS = "norm_midas"
    NORM_PRO = "norm_pro"
    NORM_VGGT = "norm_vggt"

    # Tracking
    TRACKED_DEPTH_DETECTIONS = "tracked_depth_detections"
    TRACKED_MIDAS_DETECTIONS = "tracked_midas_detections"
    TRACKED_PRO_DETECTIONS = "tracked_pro_detections"
    TRACKED_VGGT_DETECTIONS = "tracked_vggt_detections"


    # Visualizer
    VIS_DEPTH_DETECTIONS = "vis_depth_detections"
    VIS_MIDAS_DETECTIONS = "vis_midas_detections"
    VIS_PRO_DETECTIONS = "vis_pro_detections"
    VIS_VGGT_DETECTIONS = "vis_vggt_detections"
    VIS_DEPTH_TRACKED_DETECTIONS = "vis_depth_tracked_detections"
    VIS_MIDAS_TRACKED_DETECTIONS = "vis_midas_tracked_detections"
    VIS_PRO_TRACKED_DETECTIONS = "vis_pro_tracked_detections"
    VIS_VGGT_TRACKED_DETECTIONS = "vis_vggt_tracked_detections"
    VIS_MONTAGE = "vis_montage"
    VIS_ACTIVE_STREAMS = "vis_active_streams"

    # System
    SYS_FRAME_ID = "sys_frame_id"
    SYS_STATUS_INFO = "sys_status_info"

