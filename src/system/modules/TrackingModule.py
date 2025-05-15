# src/system/module/BYTETracker.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection

from typing import Optional, Dict, Any, Set, List
import numpy as np
import logging
from bytetracker import BYTETracker
import threading


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
        self.match_tresh = getattr(config, "byte_match_tresh", 0.8)
        self.frame_rate = getattr(config, "byte_frame_rate", 6)

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
                    self.frame_rate
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
    
    def _process_internal(self, data) -> Optional[Dict[str, Any]]:
        """Process and update trackers"""
        if not self.is_initialized:
            self.logger.debug(f"{self.name} not initialized, skipping process")
            return None
        
        output_data: Dict[str, Any] = {}
        current_frame_id = data.get(SystemData.SYS_FRAME_ID, 0)

        for input_key, output_key in self._stream_map.items():
            tracked_detections: List[Detection] = []

            with self._internal_lock:
                if input_key not in self._active_trackers:
                    output_data[output_key] = []
                    continue
            
            # Check for valid tracker and detections
            detections: Optional[List[Detection]] = data.get(input_key)
            if detections is None:
                #self.logger.debug(f"{input_key} not found, processing as empty")
                output_data[output_key] = []
                continue

            tracker = self.trackers.get(input_key)
            if tracker is None:
                self.logger.debug(f"Tracker for {input_key} missing")
                output_data[input_key] = []
                continue
            
            # Format data
            bytetrack_input_list = []
            for det in detections:
                formatted_data: Optional[List[float]] = det.to_bytetrack_format()
                if formatted_data is not None:
                    bytetrack_input_list.append(formatted_data)
            
            # Validate and format bytetrack input
            if bytetrack_input_list:
                bytetrack_input_np = np.array(bytetrack_input_list, dtype= np.float32)

                if bytetrack_input_np.ndim != 2 or bytetrack_input_np.shape[1] != 6:
                    bytetrack_input_np = np.empty((0, 6), dtype=np.float32)
            
            else:
                bytetrack_input_np = np.empty((0, 6), dtype=np.float32)

            # Apply tracking
            online_targets = []
            try:
                online_targets = tracker.update(bytetrack_input_np, current_frame_id)
            except Exception as e:
                self.logger.error(f"Error applying {input_key} tracking: {e}")
                online_targets = []

            # Create new tracked detection objects
            if online_targets is not None:
                if isinstance(online_targets, np.ndarray) and online_targets.ndim == 2 and online_targets.shape[1] == 7:
                    for target in online_targets:
                        try:
                            x1, y1, x2, y2 = map(int, target[0:4])
                            track_id = int(target[4])
                            class_id = int(target[6])
                            score = int(target[5])
                            label = self._class_names.get(class_id, f"Track_{track_id}")

                            tracked_det = Detection(
                                class_id=class_id,
                                label=label,
                                conf=score,
                                source=output_key,
                                dimension="2D",
                                bbox2D=[x1, y1, x2, y2],
                                track_id=track_id 
                            )
                            tracked_detections.append(tracked_det)
                        except Exception as e:
                            self.logger.error(f"Error building tracked detection: {e}")
                else:
                    #self.logger.warning(f"Tracker output for {input_key} on unexpected format")
                    pass

            output_data[output_key] = tracked_detections
        
        return output_data 
    

    def stop(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Stopping {self.name}...")
        self.trackers.clear()
        self.is_initialized = False
        self.logger.info(f"{self.name} stopped")
