# src/modules/YOLODetector

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from ultralytics.engine.results import Results
import logging
import time
from structs.Detection import Detection

class YOLODetector:
    """
    Incorporates YOLO model functionalities including loading, prediction, and drawing detections.
    """

    def __init__(self, config):
        """
        Initializes the YoloModel with the path to the model and confidence threshold.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float, optional): Minimum confidence for detections. Defaults to 0.8.
        """
        self.model_path = config.model_path
        self.confidence_threshold = config.confidence_threshold
        self.iou_treshold = config.iou_treshold
        self.model = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Loads the YOLO model from the specified path.
        """
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

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Runs prediction on the input image using the YOLO model.

        Args:
            image (np.ndarray): The input image for prediction.

        Returns:
            List[Detection]: List of detection objects.
        """
        if not self.is_initialized:
            logging.error("YOLO model not initialized when trying to detect")
            return None
        
        try:
            results = self.model.predict(source=image, verbose=False, conf=self.confidence_threshold, iou=self.iou_treshold)

            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf_score = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = result.names[class_id] if result.names and class_id in result.names else "unkown"

                    detection = Detection(
                        class_id=class_id,
                        label=label,
                        conf=conf_score,
                        source="yolo",
                        dimension = "2D",
                        bbox2D=[x1, y1, x2, y2]
                    )
                    detections.append(detection)

            logging.debug(f"Detections: {detections}")
            return detections
        except Exception as e:
            logging.error(f"Error during YOLO prediction: {e}")
            return []

    def draw_detections(self,
                       image: np.ndarray,
                       detections: List[Detection]) -> np.ndarray:
        """
        Draws bounding boxes and labels input image based on YOLO predictions.

        Args:
            image (np.ndarray): Input image
            detections (List[Detections]): YOLO prediction results.

        Returns:
            np.ndarray: Image with detections overlaid
        """
        image_copy = image.copy()

        for det in detections:
            if det.dimension == "2D" and det.bbox2D:
                # Draw box
                x1, y1, x2, y2 = det.bbox2D
                bbox_color = (0, 255, 0) 
                text_color = (0, 0, 0)
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), bbox_color, 2)

                # Construct and draw label
                label_text = f"{det.label} {det.conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image_copy, 
                            (x1, y1 - baseline - text_height),
                            (x1 + text_width, y1), 
                            bbox_color, 
                            cv2.FILLED
                        )
                cv2.putText(image_copy, 
                            label_text, 
                            (x1, y1 - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, 
                            text_color, 
                            1
                        )
        return image_copy


    def evaluate(self, yaml_path: str, test_path: str,
                 conf_thres: float = 0.5, iou_thresh: float = 0.5) -> dict:
        """
        Evaluates the YOLO model on a test dataset and prints metrics.

        Args:
            yaml_path (str): Path to the YAML file defining the dataset.
            test_path (str): Path to the test dataset.
            conf_thres (float, optional): Confidence threshold for detections. Defaults to 0.5.
            iou_thresh (float, optional): Intersection over Union threshold for non-max suppression. Defaults to 0.5.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            start_time = time.perf_counter()
            predictions = self.model.predict(source=test_path, conf=conf_thres, iou=iou_thresh)
            total_time = time.perf_counter() - start_time
            avg_inference_time = (total_time / len(predictions)) * 1000

            metrics = self.model.val(data=yaml_path, conf=conf_thres, iou=iou_thresh, verbose=True)
            results = {
                'Precision': metrics.results_dict.get('metrics/precision(B)', 0),
                'Recall': metrics.results_dict.get('metrics/recall(B)', 0),
                'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                'F1 Score': 2 * (metrics.results_dict.get('metrics/precision(B)', 0) * 
                                  metrics.results_dict.get('metrics/recall(B)', 0)) / 
                                 (metrics.results_dict.get('metrics/precision(B)', 0) + 
                                  metrics.results_dict.get('metrics/recall(B)', 0) + 1e-6), 
                'Inference Time (ms)': avg_inference_time
            }

            logging.info("\n" + "="*50)
            logging.info(f"Evaluation Report on Test Data at {test_path}")
            logging.info("="*50)
            logging.info("\nDetection Metrics:")
            for metric, value in results.items():
                logging.info(f"{metric}: {value:.4f}")
            logging.info("="*50)

            return results
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            return {}