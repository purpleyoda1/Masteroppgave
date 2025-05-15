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
                 module_path: str,
                 stream_map: Dict[str, str]):
        """
        Initializes chosen YOLO depth detector
        """
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)

        if YOLO is None:
            raise ImportError("Ultralytics library not found")
        
        self._stream_map = stream_map # Input -> Output
        self.model_path = module_path
        self.confidence_threshold = self._config.confidence_threshold
        self.iou_treshold = self._config.iou_treshold
        self.class_names = self._config.class_names
        self.model = None
        self.is_initialized = False
    
    def initialize(self, config: Any) -> bool:
        """Initialize specified YOLO model"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        self.logger.info(f"Initializing {self.name}")
        try:
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

    def get_required_inputs(self) -> Set[str]:
        return set()
    
    def get_dependency_inputs(self) -> Set[str]:
        return set(self._stream_map.keys())
    
    def get_outputs(self) -> Set[str]:
        return set(self._stream_map.values())
    
    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core implementation for applying YOLO detection
        """
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        output_data: Dict[str, Any] = {}

        for input_key, output_key in self._stream_map.items():
            input_image = data.get(input_key)

            if input_image is None:
                self.logger.debug(f"Required input {input_key} missing")
                output_data[output_key] = []
                continue
            
            try:
                detections_list = self._detect(input_image)
                self.logger.debug(f"Detected {len(detections_list)} objects in {input_key}")
                output_data[output_key] = detections_list
            except Exception as e:
                self.logger.error(f"Error applying {self.name} to input image: {e}")
                output_data[output_key] = []

        return output_data
     
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
                model_names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf_score = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = self.class_names.get(class_id)
                    if label == None and model_names:
                        label = model_names.get(class_id)
                    elif label is None:
                        label = f"{class_id}"

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