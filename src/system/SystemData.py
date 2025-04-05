# src/system/SystemData.py

class SystemData:
    """Types of data available in system"""
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

    # YOLO
    DEPTH_DETECTIONS = "depth_detections"
    MIDAS_DETECTIONS = "midas_detections"
    PRO_DETECTIONS = "pro_detections"
    
    # Normalizer
    NORM_DEPTH = "norm_depth"
    NORM_MIDAS = "norm_midas"
    NORM_PRO = "norm_pro"

    # Visualizer
    VIS_DEPTH_COLORMAP = "vis_depth_colormap"
    VIS_MIDAS_COLORMAP = "vis_midas_colormap"
    VIS_PRO_COLORMAP = "vis_pro_colormap"
    VIS_DEPTH_COLORMAP_DETECTIONS = "vis_depth_colormap_detections"
    VIS_MIDAS_COLORMAP_DETECTIONS = "vis_midas_colormap_detections"
    VIS_PRO_COLORMAP_DETECTIONS = "vis_pro_colormap_detections"
    VIS_MONTAGE = "vis_montage"