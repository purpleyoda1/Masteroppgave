# src/modules/SystemController.py

import logging
import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Set, Callable
import open3d as o3d

class StreamType:
    """Types of datastream available"""
    COLOR = "color"
    DEPTH = "depth"
    DEPTH_COLORMAP = "depth_colormap"
    DEPTH_DETECTIONS = "depth_detections"
    DEPTH_COLORMAP_DETECTIONS = "depth_colormap_detections"
    ESTIMATED_DEPTH = "estimated_depth"
    ESTIMATED_DEPTH_COLORMAP = "estimated_depth_colormap"
    ESTIMATED_DEPTH_DETECTIONS = "estimated_depth_detections"
    POINT_CLOUD = "point_cloud"

class SystemController:
    """Central controller for entire arcitecture"""

    def __init__(self, config):
        """Initialize controller"""
        self.config = config
        self.logger = logging.getLogger("SystemController")

        # Components
        self.camera = None
        self.point_cloud = None
        self.yolo_detector = None
        self.yolo_DE_detector = None
        self.pointnet_detector = None
        self.depth_estimator = None

        # State
        self.is_running = False
        self.enabled_streams = set([StreamType.COLOR, StreamType.DEPTH])
        self.enabled_detectors = set()
        self.current_data = {}

        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.data_lock = threading.Lock()

        # Callbacks
        self.frame_callbacks = []
        self.detection_callbacks = []

    def set_camera(self, camera) -> None:
        self.camera = camera
        self.logger.info("Camera component set")

    def set_point_cloud(self, point_cloud) -> None:
        self.point_cloud = point_cloud
        self.logger.info("Point cloud component set")

    def set_yolo_detector(self, yolo_detector) -> None:
        self.yolo_detector = yolo_detector
        self.logger.info("YOLO detector component set")
    
    def set_yolo_DE_detector(self, yolo_DE_detector) -> None:
        self.yolo_DE_detector = yolo_DE_detector
        self.logger.info("YOLO dept estimation detector component set")

    def set_pointnet_detector(self, pointnet_detector) -> None:
        self.pointnet_detector = pointnet_detector
        self.logger.info("PointNet detector component set")

    def set_depth_estimator(self, depth_estimator) -> None:
        self.depth_estimator = depth_estimator
        self.logger.info("Depth estimator component set")

    def initialize(self) -> bool:
        """Initalize all added components"""
        try:
            self.logger.info("Initializing system...")

            # Initialize camera
            if self.camera:
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize camera")
                    return False
                self.logger.info("Camera intialized")

                # Get intrinsics for point cloud
                if self.point_cloud and hasattr(self.camera, 'get_o3d_intrinsics'):
                    intrinsics = self.camera.get_o3d_intrinsics()
                    if intrinsics:
                        self.point_cloud.set_intrinsics(intrinsics)
            
            # Initialize YOLO detector
            if self.yolo_detector:
                if not self.yolo_detector.initialize():
                    self.logger.error("Failed to initialize YOLO detector")
                    return False
                self.logger.info("YOLO detector intialized")
            
            # Initialize YOLO estimation detector
            if self.yolo_DE_detector:
                if not self.yolo_DE_detector.initialize():
                    self.logger.error("Failed to initialize YOLO dept estimation detector")
                    return False
                self.logger.info("YOLO depth estimation detector initialized")

            # Initialize PointNet detector
            if self.pointnet_detector:
                if not self.pointnet_detector.initialize():
                    self.logger.error("Failed to initialize PointNet detector")
                    return False
                self.logger.info("PointNet detector intialized")

            # Initialize depth estimator
            if self.depth_estimator:
                if not self.depth_estimator.initialize():
                    self.logger.error("Failed to initialize depth estimator")
                    return False
                self.logger.info("Depth estimator intialized")
            
            self.logger.info("System intialization complete")
            return True
        
        except Exception as e:
            self.logger.error(f"Error intializing system: {e}")
            return False
        
    def start(self) -> bool:
        """Start processing pipeline"""
        if self.is_running:
            self.logger.warning("System already running")
            return False
        
        try:
            # Reset stop event
            self.stop_event.clear()

            # Start frame processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.is_running = True
            self.logger.info("System started")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            return False
        
    def stop(self) -> bool:
        """Stop processing and release thread"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping system...")
        self.stop_event.set()

        # Wait for thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
        
        # Stop camera
        if self.camera:
            self.camera.stop()


        self.is_running = False
        self.logger.info("System stopped")
    
    def enable_stream(self, stream_type: str) -> None:
        """Enable specific datastreams"""
        self.enabled_streams.add(stream_type)
        self.logger.info(f"Stream '{stream_type}' enabled")

    def disable_stream(self, stream_type: str) -> None:
        """Enable specific datastreams"""
        if stream_type in self.enabled_streams:
            self.enabled_streams.remove(stream_type)
            self.logger.info(f"Stream '{stream_type}' disabled")

    def enable_detector(self, detector_type: str) -> None:
        """Enable specific detector"""
        if detector_type == "yolo" and self.yolo_detector:
            self.enabled_detectors.add(detector_type)
            self.logger.info("YOLO detector enabled")
        elif detector_type == "pointnet" and self.pointnet_detector:
            self.enabled_detectors.add(detector_type)
            self.logger.info("PointNet detector enabled")
        elif detector_type == "yoloDE" and self.yolo_DE_detector:
            self.enabled_detectors.add(detector_type)
            self.logger.info("YOLO depth estimation detector enabled")
    
    def disable_detector(self, detector_type: str) -> None:
        """Disable specific detector"""
        if detector_type in self.enabled_detectors:
            self.enabled_detectors.remove(detector_type)
            self.logger.info(f"Detector '{detector_type}' disabled")

    def get_current_data(self) -> Dict[str, any]:
        """Get latest data from prossesing thread"""
        with self.data_lock:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.current_data.items()}
        
    def add_frame_callback(self, callback: Callable[[List[Any]], None]) -> None:
        """Add callback function for new frames"""
        self.frame_callbacks.append(callback)

    def add_detection_callback(self, callback: Callable[[List[Any]], None]) -> None:
        """Add callback function for new detections"""
        self.detection_callbacks.append(callback)
    
    def _processing_loop(self) -> None:
        """Main processing loop that runs in the background thread"""
        self.logger.info("Main processing loop starting...")

        while not self.stop_event.is_set():
            try:
                # Check that we have a camera and recieve frames
                if not self.camera:
                    time.sleep(0.1)
                    continue
                frames = self.camera.get_frames()
                if not frames:
                    time.sleep(0.1)
                    continue
                
                self.logger.debug(f"Frames received: {frames is not None}")
                if frames:
                    self.logger.debug(f"Frame types: {list(frames.keys())}")

                # Update current data 
                with self.data_lock:
                    for frame_type, frame in frames.items():
                        if frame_type in self.enabled_streams:
                            self.current_data[frame_type] = frame
                
                self.logger.debug(f"Current data types: {list(self.current_data.keys())}")

                # Prossed data based on predefined internal function
                self._process_current_data()

                # Call on callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(self.get_current_data())
                    except Exception as e:
                        self.logger.error(f"Error in frame callback: {e}")
                
                # Slight delay to improve stability
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

        self.logger.info("Processing loop ended")

    def _process_current_data(self) -> None:
        """Process current data frame based on enabled features"""
        with self.data_lock:
            all_detections = []

            # Depth image processing
            if StreamType.DEPTH in self.current_data:
                depth_image = self.current_data[StreamType.DEPTH]
                color_image = self.current_data[StreamType.COLOR]

                if StreamType.DEPTH_COLORMAP in self.enabled_streams:
                    depth_colormap_image = self.current_data[StreamType.DEPTH_COLORMAP]

                # Run YOLO detector if enabled
                if "yolo" in self.enabled_detectors and self.yolo_detector:
                    detections = self.yolo_detector.detect(depth_image)
                    all_detections.extend(detections)

                    # Generate visualizations if needed
                    if StreamType.DEPTH_DETECTIONS in self.enabled_streams:
                        self.logger.debug(f"Received {len(detections)} detections from YOLO")
                        depth_detections = self.yolo_detector.draw_detections(depth_image.copy(), detections)
                        self.current_data[StreamType.DEPTH_DETECTIONS] = depth_detections
                    
                    if StreamType.DEPTH_COLORMAP_DETECTIONS in self.enabled_streams:
                        self.logger.debug(f"Received {len(detections)} detections from YOLO")
                        depth_colormap_detections = self.yolo_detector.draw_detections(depth_colormap_image.copy(), detections)
                        self.current_data[StreamType.DEPTH_COLORMAP_DETECTIONS] = depth_colormap_detections
                
                # Run point cloud manager if enabled
                if self.point_cloud and StreamType.POINT_CLOUD in self.enabled_streams:
                    self.point_cloud.add_frame(color_image, depth_image)
                    self.current_data[StreamType.POINT_CLOUD] = self.point_cloud.get_global_point_cloud()

                    # Run pointnet on point cloud if enabeld
                    if "pointnet" in self.enabled_detectors and self.pointnet_detector:
                        point_cloud = self.current_data[StreamType.POINT_CLOUD]
                        detections = self.pointnet_detector.detect(point_cloud)
                        all_detections.extend(detections)
                
            # Color image processing
            if StreamType.COLOR in self.current_data:
                color_image = self.current_data[StreamType.COLOR]
                
                # Run depth estimation if enabled
                if StreamType.ESTIMATED_DEPTH in self.enabled_streams and self.depth_estimator:
                    estimated_depth = self.depth_estimator.estimate_depth(color_image)
                    self.current_data[StreamType.ESTIMATED_DEPTH] = estimated_depth

                    # Generate estimated colormap if enabled
                    if StreamType.ESTIMATED_DEPTH_COLORMAP in self.enabled_streams:
                        estimated_depth_colormap = self.depth_estimator.get_colormap(estimated_depth)
                        self.current_data[StreamType.ESTIMATED_DEPTH_COLORMAP] = estimated_depth_colormap

                    # Run detection on estimation if enabled
                    if "yoloDE" in self.enabled_detectors and self.yolo_DE_detector:
                        detections = self.yolo_DE_detector.detect(estimated_depth)
                        all_detections.extend(detections)

                    # Draw detections on estimation if enabled
                    if StreamType.ESTIMATED_DEPTH_DETECTIONS in self.enabled_streams:
                        self.logger.debug(f"Received {len(detections)} detections from YOLO DE")
                        estimated_depth_detections = self.yolo_DE_detector.draw_detections(estimated_depth.copy(), detections)
                        self.current_data[StreamType.ESTIMATED_DEPTH_DETECTIONS] = estimated_depth_detections

                
            # Call callbacks
            if all_detections and self.detection_callbacks:
                for callback in self.detection_callbacks:
                    try:
                        callback(all_detections)
                    except Exception as e:
                        self.logger.error(f"Error in detection callback: {e}")
