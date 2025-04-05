# src/system/modules/YOLODetector.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection

import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional, Set
from ultralytics.engine.results import Results
import logging


class YOLODetector(SystemModule):
    """
    Wrapper for YOLO detection capability
    """
    def __init__(self, 
                 config: Any, 
                 module_name: str, 
                 input_stream_type: str = SystemData.DEPTH,
                 output_stream_type: str = SystemData.DEPTH_DETECTIONS):
        """
        Initializes chosen YOLO depth detector
        """
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        if YOLO is None:
            raise ImportError("Ultralytics library not found")
        
        self._input_stream_type = input_stream_type
        self._output_stream_type = output_stream_type
        self.model_path: Optional[str] = None
        self.confidence_threshold = self._config.confidence_threshold
        self.iou_treshold = self._config.iou_treshold
        self.model = None
        self.is_initialized = False
    
    def initialize(self, config: Any) -> bool:
        """Initialize specified YOLO model"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        self.logger.info(f"Initializing {self.name}")
        try:
            self._set_path_from_config()
            
            if not os.path.exists(self.model_path):
                logging.error(f"YOLO model not found at {self.model_path}")
                return False
            
            # Load model
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO model loaded successfully from {self.model_path}.")
            self.is_initialized = True
            return True

        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            return False


    def _set_path_from_config(self) -> None:
        """
        Set model path
        """
        if self.name == "YOLO_MiDaS":
            self.model_path = self._config.yolo_midas_model_path
        elif self.name == "YOLO_DepthPro":
            self.model_path = self._config.yolo_pro_model_path
        else:
            self.model_path = self._config.yolo_model_path

    def get_required_inputs(self) -> Set[str]:
        return {self._input_stream_type}
    
    def get_dependency_inputs(self) -> Set[str]:
        return {
            SystemData.DEPTH,
            SystemData.MIDAS_ESTIMATED_DEPTH,
            SystemData.PRO_ESTIMATED_DEPTH,
            SystemData.NORM_DEPTH,
            SystemData.NORM_MIDAS,
            SystemData.NORM_PRO
            }
    
    def get_outputs(self) -> Set[str]:
        return {self._output_stream_type}
    
    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core implementation for applying YOLO detection
        """
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        input_image = data.get(self._input_stream_type)
        if input_image is None:
            self.logger.debug(f"Required input {self._input_stream_type} missing")
            return {self._output_stream_type: []}
        
        try:
            detections_list = self._detect(input_image)
        except Exception as e:
            self.logger.error(f"Error applying {self.name} to input image")
            detections_list = []

        return {self._output_stream_type: detections_list}
    
    def _detect(self, image: np.ndarray) -> List[Detection]:
        """
        Internal function for applying detection to image
        """
        if len(image.shape) == 2 or (len(image.shape == 3) and image.shape[2] == 1):
                controlled_image = np.stack((image,) * 3, axis=-1) if len(image.shape) == 2 else np.repeat(image, 3, axis=2)
        else:
            controlled_image = image

        try:
             results = self.model.predict(source=controlled_image, 
                                          verbose=False, 
                                          conf=self.confidence_threshold, 
                                          iou=self.iou_treshold)
        except Exception as e:
            self.logger.error(f"YOLO .predict model failed: {e}")
            return []
        
        try:
            detections = []
            for result in results:
                if result.boxes == None: continue  
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf_score = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = result.names[class_id] if result.names and class_id in result.names else "unkown"

                    detection = Detection(
                        class_id=class_id,
                        label=label,
                        conf=conf_score,
                        source=self.name,
                        dimension = "2D",
                        timestamp = time.time(),
                        bbox2D=[x1, y1, x2, y2]
                    )
                    detections.append(detection)
        except Exception as e:
            self.logger.error(f"Error proccessing detection data")

        self.logger.debug(f"Detected {len(detections)} objects in {self._input_stream_type}")
        return detections
    
    def stop(self) -> None:
        """Release resources"""
        self.logger.info(f"Stopping {self.name}...")
        if self.model is not None:
            try:
                del self.model
                self.logger.debug(f"{self.name} deleted")
            except Exception as e:
                self.logger.error(f"Error when deleting {self.name}: {e}")
            self.model = None
        self.is_initialized = False
        self.logger.info(f"{self.name} stopped")