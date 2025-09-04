# src/system/module/BYTETracker.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection

from typing import Optional, Dict, Any, Set, List
import numpy as np
import logging
from bytetracker import BYTETracker
import threading
from collections import deque, defaultdict


class TrackingModule(SystemModule):
    """
    Wrapper for ByteTrack module
    """
    def __init__(self, 
                 config: Any,
                 module_name: str,
                 stream_map: Dict[str, str]):
        """Initialize tracking module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(module_name)

        self._stream_map = stream_map
        self._class_names = getattr(self._config, "class_names", {})

        self.trackers: Dict[str, Optional[BYTETracker]] = {input_key: None for input_key in stream_map.keys()}
        self._active_trackers: Set[str] = set()
        self._internal_lock = threading.Lock()

        self.track_tresh = getattr(config, "byte_track_tresh", 0.45)
        self.track_buffer = getattr(config, "byte_track_buffer", 25)
        self.match_tresh = getattr(config, "byte_match_tresh", 0.4)
        self.frame_rate = getattr(config, "byte_frame_rate", 6)
        self.keep_frames = getattr(config, "byte_keep_frames", 9)
        self.nms_thresh = 0.2

        self.is_initialized = False

    def initialize(self, config: Any) -> bool:
        """Initialize in SystemController"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        self.logger.info(f"Initializing {self.name} for {list(self._stream_map.keys())}")
        success = True

        with self._internal_lock:
            self._active_trackers = set(self._stream_map.keys())

        for input_key in self.trackers.keys():
            self.logger.debug(f"Initializing tracker for {input_key}")
            try:
                self.trackers[input_key] = BYTETracker(
                    self.track_tresh,
                    self.track_buffer,
                    self.match_tresh,
                    self.frame_rate,
                    keep_frames=self.keep_frames
                )
                self.logger.debug(f"Tracker for {input_key} succesfully initialized")
            except Exception as e:
                self.logger.error(f"Error initializing tracker for {input_key}: {e}")
                success = False
                self.trackers[input_key] = False

        self.is_initialized = success
        if self.is_initialized:
            self.logger.info(f"{self.name} initialized successfully")
        else:
            self.logger.error(f"Error intializing {self.name}")
        
        return self.is_initialized
    
    def get_required_inputs(self) -> Set[str]:
        return set()
    
    def get_dependency_inputs(self) -> Set[str]:
        dep = set(self._stream_map.keys())
        dep.add(SystemData.SYS_FRAME_ID)
        return dep
    
    def get_outputs(self) -> Set[str]:
        return set(self._stream_map.values())
    
    def set_tracking(self, input_key: str, active: bool, reset_tracker: bool = False):
        """Updates active status of specific tracker"""
        with self._internal_lock:
            if input_key not in self._stream_map:
                self.logger.debug(f"Cannot activate, {input_key} not in tracker")

            if active: 
                self._active_trackers.add(input_key)
                status = "activated"

                # Optional reset of tracker
                if reset_tracker and self.is_initialized:
                    try:
                        self.trackers[input_key] = BYTETracker(
                            self.track_tresh,
                            self.track_buffer,
                            self.match_tresh,
                            self.frame_rate
                        )
                    except Exception as e:
                        self.logger.error(f"Error reseting tracker for {input_key}: {e}")
                        self.trackers[input_key] = None
            
                else:
                    self._active_trackers.discard(input_key)
                    status = "deactivated"
            
            self.logger.info(f"{input_key} tracker {status}")

    def is_tracking(self, input_key: str) -> bool:
        """Check status of tracker"""
        with self._internal_lock:
            return input_key in self._active_trackers
        
    def get_active(self) -> Set[str]:
        """Return current active tracking streams"""
        with self._internal_lock:
            return self._active_trackers
        
    def _perform_nms_on_detections(self, detections: List[Detection], nms_threshold: float) -> List[Detection]:
        """
        Performs Non-Maximum Suppression on the final list of tracked detections,
        applied on a per-class basis.
        """
        if not detections:
            return []

        # --- MODIFIED: Group detections by class_id ---
        detections_by_class = defaultdict(list)
        for det in detections:
            detections_by_class[det.class_id].append(det)

        final_detections = []
        # --- MODIFIED: Apply NMS for each class independently ---
        for class_id, class_detections in detections_by_class.items():
            if not class_detections:
                continue

            # Extract bounding boxes and scores for the current class
            boxes = np.array([d.bbox2D for d in class_detections if d.bbox2D is not None])
            scores = np.array([d.conf for d in class_detections if d.bbox2D is not None])
            
            if len(boxes) == 0:
                continue

            # Get indices of boxes to keep after NMS
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]
            
            keep_indices = []
            while order.size > 0:
                i = order[0]
                keep_indices.append(i)
                
                # Calculate IoU with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                intersection = w * h
                
                iou = intersection / (areas[i] + areas[order[1:]] - intersection)
                
                # Keep boxes with IoU below the threshold
                inds_to_keep = np.where(iou <= nms_threshold)[0]
                order = order[inds_to_keep + 1]
                
            # Filter the original detections list for this class and add to final list
            original_detections_with_boxes = [d for d in class_detections if d.bbox2D is not None]
            for i in keep_indices:
                final_detections.append(original_detections_with_boxes[i])
        
        return final_detections
        
    def _process_internal(self, data) -> Optional[Dict[str, Any]]:
        if not self.is_initialized:
            return None
        
        output_data: Dict[str, Any] = {}

        for input_key, output_key in self._stream_map.items():
            with self._internal_lock:
                if input_key not in self._active_trackers:
                    output_data[output_key] = []
                    continue
            
            detections: Optional[List[Detection]] = data.get(input_key)
            tracker = self.trackers.get(input_key)
            if tracker is None or detections is None:
                output_data[output_key] = []
                continue
            
            bytetrack_input_list = [det.to_bytetrack_format() for det in detections if det.to_bytetrack_format()]
            if bytetrack_input_list:
                bytetrack_input_np = np.array(bytetrack_input_list, dtype=np.float32)
            else:
                bytetrack_input_np = np.empty((0, 6), dtype=np.float32)

            online_targets = []
            try:
                online_targets = tracker.update(bytetrack_input_np)
            except Exception as e:
                self.logger.error(f"Error during tracker.update for {input_key}: {e}")
                online_targets = []

            tracked_detections: List[Detection] = []
            if len(online_targets) > 0:
                for target in online_targets:
                    try:
                        x1, y1, x2, y2 = map(int, target[0:4])
                        track_id = int(target[4])
                        class_id = int(target[5])
                        score = float(target[6])
                        
                        label = self._class_names.get(class_id, "Unknown")
                        
                        tracked_det = Detection(
                            class_id=class_id, label=label, conf=score,
                            source=output_key, dimension="2D",
                            bbox2D=[x1, y1, x2, y2], track_id=track_id 
                        )
                        tracked_detections.append(tracked_det)
                    except Exception as e:
                        self.logger.error(f"Error building tracked detection from target: {e}")

            final_tracked_detections = self._perform_nms_on_detections(
                tracked_detections, 
                nms_threshold=self.nms_thresh
            ) 
            output_data[output_key] = final_tracked_detections
        
        return output_data
    

    def stop(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Stopping {self.name}...")
        self.trackers.clear()
        self.is_initialized = False
        self.logger.info(f"{self.name} stopped")
