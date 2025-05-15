# src/system/modules/FrameSaver.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData

import os
import cv2
import numpy as np
import logging
from typing import Any, Set, Dict
import re

class FrameSaver(SystemModule):
    """
    Enables saving of frames in the Systems data as it runs
    """
    def __init__(self, config: Any, module_name: str = "FrameSaver"):
        """Initialize FrameSaver module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(module_name)

        self._save_base_dir = getattr(self._config, "save_base_dir")
        self._streams_to_save = getattr(self._config, "streams_to_save")
        
        self._save_flag = False
        self._current_run_dir = None
        self.is_initialized = False
        self._total_saved = 0

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
                match = pattern.match(item_path)
                if match:
                    run_num = int(match.group(1))
                    if run_num > max_run_num:
                        max_run_num = run_num
            return (max_run_num + 1)
        except OSError as e:
            self.logger.error(f"Error scanning save directory: {e}")
            return 1

    def initialize(self, config: Any) -> bool:
        """Set up and initialize module"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        try:
            # Ensure save dir exists
            os.makedirs(self._save_base_dir, exist_ok=True)

            # Determine run number and create run dir
            run_num = self._determine_next_run_number()
            run_dir = os.path.join(self._save_base_dir, f"run_{run_num:03d}")
            self._current_run_dir = run_dir
            os.makedirs(self._current_run_dir, exist_ok=True)

            self.is_initialized = True
            self.logger.info(f"{self.name} initialized succesfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing {self.name}: {e}")

    def get_required_inputs(self) -> Set[str]:
        return set()
    
    def get_dependency_inputs(self) -> Set[str]:
        return set(self._streams_to_save).union(SystemData.SYS_FRAME_ID)
    
    def get_outputs(self) -> Set[str]:
        return set()
    
    def request_save(self) -> None:
        "Trigger to request save of current frames"
        if not self.is_initialized:
            self.logger.error(f"Initialize before requesting save")
            return
        
        self._save_flag = True

    def _process_internal(self, data: Dict[str, Any]) -> None:
        "Saves frames to dictionary defined in config file"
        if not self._save_flag:
            return None
        
        # Reset save flag immediatly
        self._save_flag = False
        self._total_saved += 1

        # Iteratively save frames
        for key in self._streams_to_save:
            if key not in data:
                self.logger.warning(f"Requested save data not in current data dictionary")
                continue

            image = data[key]
            if image is None:
                self.logger.warning(f"Data in {key} is None, couldnt save")

            # Make subdirectory
            save_folder = os.path.join(self._current_run_dir, key)
            os.makedirs(save_folder, exist_ok=True)

            filename = f"{self._total_saved:04d}.png"
            filepath = os.path.join(save_folder, filename)

            # Attempt save
            cv2.imwrite(filepath, image)
            
        return None
    
    def stop(self) -> None:
        return None

