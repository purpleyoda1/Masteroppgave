# src/modules/RealSenseCamera.py

import os
import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Optional
from utilities import pad_to_square
import logging
import open3d as o3d

class RealSenseCamera:
    """
    Encapsulates RealSense camera functionalities, including frame acquisition,
    YOLO object detection, and display management.
    """

    def __init__(self, config):
        """
        Initializes the RealSenseCamera with the given configuration.

        Args:
            config (Config): Configuration settings.
        """
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.is_running = False
        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _enable_stream(self, rs_config: rs.config) -> None:
        """
        Enables depth and color streams based on the configuration.

        Args:
            rs_config (rs.config): RealSense configuration object.
        """
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.depth_stream_width,
            self.config.depth_stream_height,
            rs.format.z16,
            self.config.depth_stream_fps
        )
        rs_config.enable_stream(
            rs.stream.color,
            self.config.color_stream_width,
            self.config.color_stream_height,
            rs.format.bgr8,
            self.config.color_stream_fps
        )

    def initialize(self) -> None:
        """
        Start camera feed with the settings from the configuration.
        """
        try:
            rs_config = rs.config()

            rs_config.enable_stream(
                rs.stream.depth,
                self.config.depth_stream_width,
                self.config.depth_stream_height,
                rs.format.z16,
                self.config.depth_stream_fps
            )
            rs_config.enable_stream(
                rs.stream.color,
                self.config.color_stream_width,
                self.config.color_stream_height,
                rs.format.bgr8,
                self.config.color_stream_fps
            )

            # Start streaming and retrieve camere intrinsics and depth scale
            self.profile = self.pipeline.start(rs_config)

            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            logging.info(f"Depth scale: {self.depth_scale} meters/unit.")
            logging.info(f"Depth intrinsics: fx={self.depth_intrinsics.fx}, fy={self.depth_intrinsics.fy}, "
                        f"ppx={self.depth_intrinsics.ppx}, ppy={self.depth_intrinsics.ppy}")
            logging.info(f"Color intrinsics: fx={self.color_intrinsics.fx}, fy={self.color_intrinsics.fy}, "
                        f"ppx={self.color_intrinsics.ppx}, ppy={self.color_intrinsics.ppy}")
            self.is_running = True
            return True
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            return False

    def get_frames(self, pad: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves depth and color frames from the camera.

        Args:
            pad (bool, optional): Whether to pad the images to square. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Depth image and color image.
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_data = aligned_frames.get_depth_frame()
            color_data = aligned_frames.get_color_frame()
            if not depth_data or not color_data:
                logging.warning("Failed to get frames from camera")
                return None

            depth_image = np.asanyarray(depth_data.get_data())
            color_image = np.asanyarray(color_data.get_data())

            if pad:
                depth_image, _, _ = pad_to_square(depth_image)
                color_image, _, _ = pad_to_square(color_image)

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            return {
                'color': color_image,
                'depth': depth_image,
                'depth_colormap':depth_colormap
            }
        except Exception as e:
            logging.error(f"Error getting frames: {e}")
            return None
        
    def get_o3d_intrinsics(self, color_stream: bool= False) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        """
        Get intrinsics for Open3D pointcloud generation
        """

        if color_stream:
            intrinsics = self.color_intrinsics
        else:
            intrinsics = self.depth_intrinsics
        
        if not intrinsics:
            logging.warning("Camera intrinsics not available for Open3D preparation")
            return None
        
        return o3d.camera.PinholeCameraIntrinsic(
            width = intrinsics.width,
            height = intrinsics.height,
            fx = intrinsics.fx,
            fy = intrinsics.fy,
            cx = intrinsics.ppx,
            cy = intrinsics.ppy
        )

    def stop(self) -> None:
        """
        Stops the camera and releases all resources.
        """
        self.is_running = False
        self.pipeline.stop()
        logging.info("Camera pipeline stopped and windows closed.")