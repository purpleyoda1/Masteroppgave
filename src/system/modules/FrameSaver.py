      
from ..SystemModule import SystemModule
from ..SystemData import SystemData

import os
import cv2
import numpy as np
import logging
from typing import Any, Set, Dict, Optional, List
import re
import threading

class FrameSaver(SystemModule):
    """
    Enables saving of frames in the Systems data as it runs
    """
    def __init__(self, config: Any, module_name: str = "FrameSaver"):
        """Initialize FrameSaver module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(module_name)

        self._save_base_dir = getattr(self._config, "save_base_dir")
        
        self._save_flag = False
        self._continous_save = False
        self._current_run_dir = None
        self._current_sequence_dir = None
        self.is_initialized = False
        self._sequence_frame_count = 0
        self._sequence_num = 0
        self._total_saved = 0
        self._save_lock = threading.Lock()
        self._activated = False

    def _determine_next_run_number(self) -> int:
        """Find next number based on exisiting directories"""
        max_run_num = 0
        if not os.path.isdir(self._save_base_dir):
            return 1
        
        # Use re library to search for pattern run_###
        pattern = re.compile(rf"^{re.escape('run')}_(\d+)$")
        try:
            for item in os.listdir(self._save_base_dir):
                item_path = os.path.join(self._save_base_dir, item)
                match = pattern.match(item)
                if match:
                    run_num = int(match.group(1))
                    if run_num > max_run_num:
                        max_run_num = run_num
            self.logger.info(f"Run number determined: {max_run_num+1}")
            return (max_run_num + 1)
        except OSError as e:
            self.logger.error(f"Error scanning save directory: {e}")
            return 1

    def _create_run_dir(self) -> None:
        """Setup save dir"""
        try:
            # Ensure save dir exists
            os.makedirs(self._save_base_dir, exist_ok=True)

            # Determine run number and create run dir
            run_num = self._determine_next_run_number()
            run_dir = os.path.join(self._save_base_dir, f"run_{run_num:03d}")
            self._current_run_dir = run_dir
            os.makedirs(self._current_run_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating save dir: {e}")

    def initialize(self, config: Any) -> bool:
        """Set up and initialize module"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        try:
            self.is_initialized = True
            self.logger.info(f"{self.name} initialized succesfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing {self.name}: {e}")

    def get_required_inputs(self) -> Set[str]:
        return set()
    
    def get_dependency_inputs(self) -> Set[str]:
        config_save_streams = self._config.streams_to_save
        dependencies = {
            SystemData.VIS_MONTAGE,
            SystemData.VIS_ACTIVE_STREAMS
        }
        dependencies.update(config_save_streams)
        return dependencies
    
    def get_outputs(self) -> Set[str]:
        return set()
    
    def request_save(self) -> None:
        "Trigger to request save of current frames"
        if not self.is_initialized:
            self.logger.error(f"Initialize before requesting save")
            return
        
        with self._save_lock:
            if not self._continous_save:
                self.logger.info("Request save button pressed: Forcing continuous save start.")
                self.set_continous_save(True)
        
        self._save_flag = True

    def set_continous_save(self, enable: bool) -> None:
        """Enable or disable continous save"""
        if enable and not self._continous_save:
            # If this is the first save action ever, create the run directory
            if not self._activated:
                self._create_run_dir()
                self._activated = True

            # Start new sequence
            self._sequence_num += 1
            self._sequence_frame_count = 0
            sequence_dir = os.path.join(self._current_run_dir, f"sequence_{self._sequence_num:03d}")
            os.makedirs(sequence_dir, exist_ok=True)
            self._current_sequence_dir = sequence_dir
            self.logger.info(f"Started continous save to: {self._current_sequence_dir}")
        elif not enable and self._continous_save:
            self.logger.info(f"Stopped continous save. Saved {self._sequence_frame_count} frames to {self._current_sequence_dir}")
            self._current_sequence_dir = None
        
        self._continous_save = enable
    
    def is_continous_saving(self) -> bool:
        """Check if continous saving"""
        with self._save_lock:
            return self._continous_save
        
    def _save_frames(self, data: Dict[str, Any], save_dir: str, frame_number: int) -> None:
        """Internal method to save frames"""
        # Get frames to save from the active streams list provided by the visualization module
        streams_to_save = data.get(SystemData.VIS_ACTIVE_STREAMS, set()).copy()
        
        # Also save the main montage if it exists
        if SystemData.VIS_MONTAGE in data:
            streams_to_save.add(SystemData.VIS_MONTAGE)

        # Iteratively save frames
        for key in streams_to_save:
            if key not in data:
                self.logger.warning(f"Requested save of '{key}', but it's not in the current data dictionary. Skipping.")
                continue

            image = data[key]
            if image is None:
                self.logger.warning(f"Data in '{key}' is None, cannot save.")
                continue

            save_folder = os.path.join(save_dir, key)
            os.makedirs(save_folder, exist_ok=True)

            filename = f"{frame_number:04d}.png"
            filepath = os.path.join(save_folder, filename)

            # Attempt save
            try:
                cv2.imwrite(filepath, image)
            except Exception as e:
                self.logger.error(f"Failed to save image to {filepath}: {e}")

    def _process_internal(self, data: Dict[str, Any]) -> None:
        "Saves frames based on the current state (single save request or continuous save)."
        with self._save_lock:
            # Determine what actions are needed in this cycle
            should_save_single = self._save_flag
            should_save_continuous = self._continous_save and self._current_sequence_dir

            # If no save operation is needed, exit early
            if not should_save_single and not should_save_continuous:
                return None

            # If this is the first save operation of any kind, create the main run directory
            if not self._activated:
                self._create_run_dir()
                self._activated = True
            
            # Handle single frame save request
            if should_save_single:
                self._total_saved += 1
                self._save_frames(data, self._current_run_dir, self._total_saved)
                self.logger.info(f"Saved frame #{self._total_saved} to run directory: {self._current_run_dir}")
                self._save_flag = False  

            # Handle continuous save
            if should_save_continuous:
                self._sequence_frame_count += 1
                self._save_frames(data, self._current_sequence_dir, self._sequence_frame_count)
            
        return None
    
    def stop(self) -> None:
        return None

    