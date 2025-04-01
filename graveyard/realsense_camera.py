import os
import cv2
import numpy as np
import pyrealsense2 as rs
import datetime
import time
from enum import Enum, auto
from typing import List, Dict, Tuple
from system.YOLODetector import YoloDetector
from utilities import pad_to_square
import matplotlib.pyplot as plt
import logging
import open3d as o3d
#from pointcloud.PointCloudFK import PointCloudFK

class StreamView(Enum):
    """
    Enum for controlling what to include when displaying stream.
    """
    COLOR = auto()
    DEPTH = auto()
    DEPTH_COLORMAP = auto()
    COLOR_OVERLAY = auto()
    DEPTH_OVERLAY = auto()

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
        self.intrinsics = None
        self.model = None
        self.point_cloud = None
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

    def _initialize_yolo_model(self) -> None:
        """
        Initializes the YOLO model based on the configuration.
        """
        self.model = YoloDetector(self.config.model_path, self.config.confidence_threshold)
        self.model.load_model()
        logging.info("YOLO model initialized.")
    
    def _initialize_point_cloud(self) -> None:
        """
        Set up point cloud based on camera intrinsics
        """
        open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        open3d_intrinsics.set_intrinsics(
            self.intrinsics.width,
            self.intrinsics.height,
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.ppx,
            self.intrinsics.ppy
        )
        self.point_cloud = PointCloudFK(open3d_intrinsics)
        logging.info("Point cloud intialized")

    def initialize(self) -> None:
        """
        Initializes the camera with the settings from the configuration.
        """
        rs_config = rs.config()
        self._enable_stream(rs_config)
        
        # Start streaming
        self.profile = self.pipeline.start(rs_config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        logging.info(f"Depth scale: {self.depth_scale} meters/unit.")
        logging.info(f"Camera intrinsics: {self.intrinsics}.")

        # Set up point cloud
        self._initialize_point_cloud()

        # Set up YOLO model
        self._initialize_yolo_model()

        self.is_running = True

    def get_frames(self, pad: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves depth and color frames from the camera.

        Args:
            pad (bool, optional): Whether to pad the images to square. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Depth image and color image.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_data = aligned_frames.get_depth_frame()
        color_data = aligned_frames.get_color_frame()
        if not depth_data or not color_data:
            logging.warning("No frames received.")
            return None, None

        depth_image = np.asanyarray(depth_data.get_data())
        color_image = np.asanyarray(color_data.get_data())

        if pad:
            depth_image, _, _ = pad_to_square(depth_image)
            color_image, _, _ = pad_to_square(color_image)

        return depth_image, color_image

    def apply_yolo(self) -> Dict[StreamView, np.ndarray]:
        """
        Applies the YOLO model to the current frames and draws bounding boxes.

        Returns:
            Dict[StreamView, np.ndarray]: Processed frames for different views.
        """
        depth_image, color_image = self.get_frames(pad=False)
        if depth_image is None or color_image is None:
            logging.warning("No images found for YOLO application.")
            return {}

        # Run inference on depth input (converted to 3 channels)
        depth_3channel = np.stack((depth_image) * 3, axis=-1)
        results = self.model.predict(depth_3channel)

        # Process depth image for display
        depth_image_float = depth_image.astype(float)
        depth_mask = depth_image_float > 0
        if depth_mask.any():
            depth_min = depth_image_float[depth_mask].min()
            depth_max = depth_image_float[depth_mask].max()
            depth_image_float[depth_mask] = (depth_image_float[depth_mask] - depth_min) / (depth_max - depth_min)

        cmap = plt.get_cmap('viridis')
        depth_colored = cmap(depth_image_float)
        depth_colormap = (depth_colored[:, :, :3] * 255).astype(np.uint8)

        # Draw bounding boxes
        depth_overlay, color_overlay = self.model.draw_detections(
            depth_colormap.copy(),
            color_image.copy(),
            results,
            self.depth_scale
        )

        return {
            StreamView.COLOR: color_image,
            StreamView.DEPTH: depth_image,
            StreamView.DEPTH_COLORMAP: depth_colormap,
            StreamView.COLOR_OVERLAY: color_overlay,
            StreamView.DEPTH_OVERLAY: depth_overlay
        }

    def prepare_display(self, processed_frames: Dict[StreamView, np.ndarray], views: List[StreamView], scale_factor: float = 1.5) -> np.ndarray:
        """
        Stacks and resizes requested frames for display.

        Args:
            processed_frames (Dict[StreamView, np.ndarray]): Processed frames.
            views (List[StreamView]): List of views to display.
            scale_factor (float, optional): Scaling factor for the display image. Defaults to 1.5.

        Returns:
            np.ndarray: The combined image ready for display.
        """
        if not views:
            raise ValueError("At least one view must be requested.")

        # Collect frames to display
        frames_to_display = [processed_frames[view] for view in views if view in processed_frames]
        if not frames_to_display:
            logging.warning("No frames available for display.")
            return np.array([])

        # Stack images horizontally
        combined_image = np.hstack(frames_to_display)

        # Resize
        width = int(combined_image.shape[1] * scale_factor)
        height = int(combined_image.shape[0] * scale_factor)

        display_image = cv2.resize(combined_image, (width, height), interpolation=cv2.INTER_LINEAR)
        return display_image

    def save_frame(self, processed_frames: Dict[StreamView, np.ndarray], views: List[StreamView]) -> None:
        """
        Saves specified frames to disk.

        Args:
            processed_frames (Dict[StreamView, np.ndarray]): Processed frames.
            views (List[StreamView]): List of views to save.
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for view in views:
                frame = processed_frames.get(view)
                if frame is not None:
                    filename = f"{view.name.lower()}_{timestamp}.png"
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, frame)
                    logging.info(f"Saved {view.name} frame to {filepath}.")
                else:
                    logging.warning(f"{view.name} frame is not available.")
        except Exception as e:
            logging.error(f"Error saving frame: {e}")

    def run(self, views: List[StreamView] = [StreamView.COLOR, StreamView.DEPTH_COLORMAP], window_name: str = "Realsense Camera", scale_factor: float = 1.5) -> None:
        """
        Runs the camera loop to display and process frames.

        Args:
            views (List[StreamView], optional): List of views to display. Defaults to [StreamView.COLOR, StreamView.DEPTH_COLORMAP].
            window_name (str, optional): Name of the display window. Defaults to "Realsense Camera".
            scale_factor (float, optional): Scaling factor for the display image. Defaults to 1.5.
        """

        last_save = 0
        save_cooldown = 0.5

        try:
            while self.is_running:
                processed_frames = self.apply_yolo()
                if not processed_frames:
                    continue

                display_image = self.prepare_display(processed_frames, views, scale_factor)
                if display_image.size == 0:
                    continue

                cv2.imshow(window_name, display_image)

                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                if key == ord('q'):
                    logging.info("Exit key pressed.")
                    break
                elif key == ord('s'):
                    if (current_time - last_save) > save_cooldown:
                        self.save_frame(processed_frames, views)
                        last_save = current_time
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Stops the camera and releases all resources.
        """
        self.is_running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()
        logging.info("Camera pipeline stopped and windows closed.")