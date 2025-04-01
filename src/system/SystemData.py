# src/system/SystemData.py

class SystemData:
    """Types of data available in system"""
    COLOR = "color"
    DEPTH = "depth"
    COLOR_INTRINSICS = "color_intrincsics"
    DEPTH_INTRINSICS = "depth_intrinsics"
    CAMERA_DEPTH_SCALE = "camera_depth_scale"
    DEPTH_COLORMAP = "depth_colormap"
    DEPTH_DETECTIONS = "depth_detections"
    DEPTH_COLORMAP_DETECTIONS = "depth_colormap_detections"
    MIDAS_ESTIMATED_DEPTH = "midas_estimated_depth"
    MIDAS_ESTIMATED_DEPTH_COLORMAP = "midas_estimated_depth_colormap"
    MIDAS_ESTIMATED_DEPTH_DETECTIONS = "midas_estimated_depth_detections"
    PRO_ESTIMATED_DEPTH = "pro_estimated_depth"
    PRO_ESTIMATED_DEPTH_COLORMAP = "pro_estimated_depth_colormap"
    PRO_ESTIMATED_DEPTH_DETECTIONS = "pro_estimated_depth_detections"
    POINT_CLOUD = "point_cloud"