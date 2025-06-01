# src/system/modules/CameraImpostor.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection
from config import Config

import logging
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np
import os
import glob
import cv2

class CameraImpostor(SystemModule):
    """
    Feeds pre-captured images to the system, mimicking the expected camera functionality
    Also feeds corresponding ground truth
    """
    def __init__(self, config: Config, module_name: str = "CameraImpostor"):
        """Initialize impostor"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        self.target_folder_path = getattr(self._config, "eval_target_folder_path", None)
        self.start_index = getattr(self._config, "eval_start_index", 0)
        self.end_index = getattr(self._config, "eval_end_index", -1)
        self.image_width = getattr(self._config, "eval_image_width", 640)
        self.image_height = getattr(self._config, "eval_image_height", 480)
        self.class_names = getattr(self._config, "class_names", None)
        self.load_depth_estimations = getattr(self._config, "eval_load_estimated_depths", True)



        self.color_images: List[Tuple[str, np.ndarray]] = []
        self.depth_images: List[Tuple[str, np.ndarray]] = []
        self.labels: Dict[Tuple[str, List[Detection]]] = {}
        self.midas_images: List[Tuple[str, np.ndarray]] = []
        self.pro_images: List[Tuple[str, np.ndarray]] = []
        self.vggt_images: List[Tuple[str, np.ndarray]] = []


        self.current_index = 0
        self.total_images = 0

        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = 0.001    

        self.is_initialized = False
        self.evaluation_complete = False

    def initialize(self, config: Any) -> bool:
        """Initialize and load images"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already intialized")
            return True
        
        try:
            self.logger.info(f"Initializing {self.name}")

            if not os.path.exists(self.target_folder_path):
                self.logger.error(f"Target path not valid: {self.target_folder_path}")
                return False
            
 
            self._load_images()
            self._load_labels()
            if self.load_depth_estimations:
                self._load_depth_estimations()

            self._setup_intrinsics()

            self.logger.info(f"Loaded {self.total_images} images and {len(self.labels)} labels for evaluation")
            self.is_initialized = True
            return True
        
        except Exception as e:
            self.logger.error(f"Error initializing {self.name}: {e}")
            return False
        
    def _load_images(self):
        """Load color and depth images from folder"""
        color_pattern = os.path.join(self.target_folder_path, "color", "*.png")
        color_files = sorted(glob.glob(color_pattern))

        depth_pattern = os.path.join(self.target_folder_path, "depth", "*.png")
        depth_files = sorted(glob.glob(depth_pattern))

        for color_file in color_files:
            img = cv2.imread(color_file, cv2.IMREAD_COLOR)
            if img is not None:
                basename = os.path.basename(color_file)
                self.color_images.append((basename, img))

        for depth_file in depth_files:
            img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                basename = os.path.basename(depth_file)
                self.depth_images.append((basename, img))

        self.total_images = len(self.color_images)

    def _load_labels(self):
        """Load YOLO labels"""
        label_pattern = os.path.join(self.target_folder_path, "labels", "*.txt")
        label_files = sorted(glob.glob(label_pattern))

        for label_file in label_files:
            basename = os.path.splitext(os.path.basename(label_file))[0]
            detections = self._parse_labels(label_file)
            if detections:
                self.labels[basename] = detections

    def _load_depth_estimations(self):
        """Load preproccesed depth estimations"""
        # MiDaS
        midas_pattern = os.path.join(self.target_folder_path, "midas_norm", "*.png")
        midas_files = sorted(glob.glob(midas_pattern))

        for midas_file in midas_files:
            img = cv2.imread(midas_file, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                basename = os.path.splitext(os.path.basename(midas_file))[0]
                self.midas_images.append((basename, img))
        self.logger.info(f"Loaded {len(self.midas_images)} midas images")

        # Depth Pro
        pro_pattern = os.path.join(self.target_folder_path, "depthpro_norm", "*.png")
        pro_files = sorted(glob.glob(pro_pattern))

        for pro_file in pro_files:
            img = cv2.imread(pro_file, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                basename = os.path.splitext(os.path.basename(pro_file))[0]
                self.pro_images.append((basename, img))
        self.logger.info(f"Loaded {len(self.pro_images)} pro images")

        # VGGT
        vggt_pattern = os.path.join(self.target_folder_path, "vggt_norm", "*.png")
        vggt_files = sorted(glob.glob(vggt_pattern))

        for vggt_file in vggt_files:
            img = cv2.imread(vggt_file, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                basename = os.path.splitext(os.path.basename(vggt_file))[0]
                self.vggt_images.append((basename, img))
        self.logger.info(f"Loaded {len(self.vggt_images)} vggt images")
    
    def _parse_labels(self, label_file: str) -> List[Detection]:
        """Interpret label file and convert to Detections"""
        detections = []

        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    abs_x_center = x_center * self.image_width
                    abs_y_center = y_center * self.image_height
                    abs_width = width * self.image_width
                    abs_height = height * self.image_height

                    x1 = int(abs_x_center - (abs_width / 2))
                    y1 = int(abs_y_center - (abs_height / 2))
                    x2 = int(abs_x_center + (abs_width / 2))
                    y2 = int(abs_y_center + (abs_height / 2))

                    label = self.class_names.get(class_id)

                    detection = Detection(
                        class_id=class_id,
                        label=label,
                        conf=1.0,
                        source="ground_truth",
                        dimension="2D",
                        bbox2D=[x1, y1, x2, y2]
                    )
                    detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"Error parsing YOLO label {label_file}: {e}")
        
        return detections
    
    def _setup_intrinsics(self):
        """Set up intrinsics that mimick the camera"""
        class MockIntrinsics:
            def __init__(self, width, height, fx, fy, cx, cy):
                self.width = width
                self.height = height
                self.fx = fx
                self.fy = fy
                self.cx = cx
                self.cy = cy

        self.color_intrinsics = MockIntrinsics(
            width=self._config.color_stream_width,
            height=self._config.color_stream_height,
            fx=self._config.color_stream_fx, 
            fy=self._config.color_stream_fy,
            cx=self._config.color_stream_cx,
            cy=self._config.color_stream_cy
        )

        self.depth_intrinsics = MockIntrinsics(
            width=self._config.depth_stream_width,
            height=self._config.depth_stream_height,
            fx=self._config.depth_stream_fx, 
            fy=self._config.depth_stream_fy,
            cx=self._config.depth_stream_cx,
            cy=self._config.depth_stream_cy
        )

    def get_required_inputs(self) -> Set[str]:
        return set()
                
    def get_dependency_inputs(self) -> Set[str]:
        return set()
    
    def get_outputs(self) -> Set[str]:
        """Returns what StreamData this module produces"""
        outputs = {
            SystemData.COLOR,
            SystemData.DEPTH,
            SystemData.COLOR_INTRINSICS,
            SystemData.DEPTH_INTRINSICS,
            SystemData.CAMERA_DEPTH_SCALE,
            SystemData.GROUND_TRUTH_DETECTIONS,
            SystemData.GROUND_TRUTH_IMAGE_PATH,
        }

        if self.load_depth_estimations:
            outputs.update({
                SystemData.NORM_MIDAS,
                SystemData.NORM_PRO,
                SystemData.NORM_VGGT
            })
        return outputs
    
    def _process_internal(self, data) -> Optional[Dict[str, Any]]:
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        if self.current_index >= self.total_images:
            self.evaluation_complete = True
            self.logger.info("All images processed")
            return {SystemData.EVAL_COMPLETE: True}
        
        output_data = {}
        self.logger.info(f"Current index: {self.current_index}")
        if self.current_index < len(self.color_images):
            color_filename, color_img = self.color_images[self.current_index]
            output_data[SystemData.COLOR] = color_img
            output_data[SystemData.GROUND_TRUTH_IMAGE_PATH] = color_filename
            filename, depth_img = self.depth_images[self.current_index]
            output_data[SystemData.DEPTH] = depth_img
            basename = os.path.splitext(color_filename)[0] 
            detections = self.labels[basename]
            output_data[SystemData.GROUND_TRUTH_DETECTIONS] = detections

            if self.load_depth_estimations:
                _, midas_depth = self.midas_images[self.current_index]
                output_data[SystemData.NORM_MIDAS] = midas_depth
                _, pro_depth = self.pro_images[self.current_index]
                output_data[SystemData.NORM_PRO] = pro_depth
                _, vggt_depth = self.vggt_images[self.current_index]
                output_data[SystemData.NORM_VGGT] = vggt_depth


        output_data[SystemData.COLOR_INTRINSICS] = self.color_intrinsics
        output_data[SystemData.DEPTH_INTRINSICS] = self.depth_intrinsics
        output_data[SystemData.CAMERA_DEPTH_SCALE] = self.depth_scale

        self.current_index += 1
        self.logger.info(f"Processing image {self.current_index}/{self.total_images}")

        return output_data
    
    def stop(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Stopping {self.name}")
        self.color_images.clear()
        self.depth_images.clear()
        self.labels.clear()
        self.is_initialized = False