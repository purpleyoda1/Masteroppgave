# src/system/modules/RealSenseCamera.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData

import numpy as np
import pyrealsense2 as rs
from typing import  Optional, Any, Dict, Set
import logging

class RealSenseCamera(SystemModule):
    """
    Wrapper for RealSense camera functionality
    Aquire depth and RGB frames along with camera intrinsics
    """
    def __init__(self, config: Any, module_name: str = "RealSense_Camera"):
        """Initialize RealSense module"""
        if rs is None:
            logging.error("pyrealsense2 library not found")
            raise ImportError("pyrealsense2 library not found")
        
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        self.pipeline = None
        self.align = None
        self.profile = None
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.is_initialized = False

        self.depth_w = getattr(self._config, "depth_stream_width", 424)
        self.depth_h = getattr(self._config, "depth_stream_height", 240)
        self.depth_fps = getattr(self._config, "depth_stream_fps", 6)
        self.color_w = getattr(self._config, "color_stream_width", 424)
        self.color_h = getattr(self._config, "color_stream_height", 240)
        self.color_fps = getattr(self._config, "color_stream_fps", 15)


    def initialize(self, config: Any) -> bool:
        """Initialize module"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already intialized")
            return True

        try:
            self.pipeline = rs.pipeline()
            self.align = rs.align(rs.stream.color)
            rs_config = rs.config()

            self.logger.debug(f"Enabling depth stream")
            rs_config.enable_stream(
                rs.stream.depth,
                self.depth_w,
                self.depth_h,
                rs.format.z16,
                self.depth_fps
            )
            self.logger.debug(f"Enabling color stream")
            rs_config.enable_stream(
                rs.stream.color,
                self.color_w,
                self.color_h,
                rs.format.bgr8,
                self.color_fps
            )

            self.logger.debug("Starting RealSense pipeline")
            self.profile = self.pipeline.start(rs_config)

            # get camera parameters
            depth_sensor = self.profile.get_device().first_depth_sensor()
            if depth_sensor:
                self.depth_scale = depth_sensor.get_depth_scale()
                self.logger.info(f"Depth scale: {self.depth_scale:.5f} m/unit")
            else:
                self.logger.warning("Couldnt access depth sensor")

            self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            # Wait for camera to start up properly
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.logger.info(f"{self.name} initialized succesfully")
            self.is_initialized = True

            return True
        
        except Exception as e:
            self.logger.error(f"Error during initialization of {self.name}: {e}")
            if self.pipeline:
                try: self.pipeline.stop()
                except: pass
            self.is_initialized = False
            return False

    def get_required_inputs(self) -> Set[str]:
        """Empty since it only produce frames"""
        return set()

    def get_outputs(self) -> Set[str]:
        """Returns what StreamData this module produces"""
        return {
            SystemData.COLOR,
            SystemData.DEPTH,
            SystemData.COLOR_INTRINSICS,
            SystemData.DEPTH_INTRINSICS,
            SystemData.CAMERA_DEPTH_SCALE,
        }

    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core implementation of camera functionality
        """
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                logging.warning("Failed to get frames from camera")
                return None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return {
                SystemData.COLOR: color_image,
                SystemData.DEPTH: depth_image,
                SystemData.COLOR_INTRINSICS: self.color_intrinsics,
                SystemData.DEPTH_INTRINSICS: self.depth_intrinsics,
                SystemData.CAMERA_DEPTH_SCALE: self.depth_scale
            }
            
        except Exception as e:
            self.logger.warning(f"Error during frame acquisition: {e}")
            return None
        
    def stop(self) -> None:
        """Release resources"""
        self.logger.info(f"Stopping {self.name}")
        if self.pipeline and self.is_initialized:
            try:
                self.pipeline.stop()
                self.logger.info("RealSense pipeline stopped")
            except Exception as e:
                self.logger.error(f"Error stopping RealSense pipeline: {e}")
        else:
            self.logger.debug("RealSense pipeline not running")

        self.is_initialized = False
        self.pipeline = None
        self.align = None
        self.profile = None
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.logger.info(f"{self.name} stopped")