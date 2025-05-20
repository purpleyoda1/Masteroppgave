# src/system/SystemController.py

from .SystemData import SystemData
from .SystemModule import SystemModule
from system.modules.RealSenseCamera import RealSenseCamera
from system.modules.YOLODetector import YOLODetector
from system.modules.MiDaSDepthEstimator import MiDaSDepthEstimator
from system.modules.VisualizationModule import VisualizationModule
from system.modules.DepthNormalizer import DepthNormalizer
from system.modules.TrackingModule import TrackingModule
from system.modules.FrameSaver import FrameSaver
from config import Config

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
import sys

class SystemController:
    """Central controller managing information flow and interaction between modules"""
    def __init__(self, config: Config, USING_LAPTOP: bool= False):
        """Initialize controller"""
        self.config = config
        self.logger = logging.getLogger("SystemController")
        self.USING_LAPTOP = USING_LAPTOP

        # Components
        self.modules: Dict[str, SystemModule] = {}
        self.module_states: Dict[str, bool] = {}
        self.processing_order: Optional[List[str]] = None
        self._primary_source_name: str = "RealSense_Camera"

        # State
        self.is_running = False
        self.desired_streams: Set[str] = set()
        self._required_streams: Set[str] = set()
        self.current_data: Dict[str, Any] = {}

        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.data_lock = threading.Lock()

        # Frame counter
        self.sys_frame_counter: int = 0

        self._is_initialized = False

    def add_module(self, module: SystemModule) -> None:
        """Register module in controller"""
        if module.name in self.modules:
            self.logger.warning(f"Module {module.name} already in controller")

        self.modules[module.name] = module
        self.module_states[module.name] = False
        self.logger.info(f"Module {module.name} added to SystemController")

        # Reset processing order
        self.processing_order = None

    def enable_module(self, module_name: str, enable: bool = True) -> None:
        """Enable or disable specific modules"""
        if module_name not in self.modules:
            self.logger.warning(f"Cannont enable/disable {module_name}")
            return
        
        self.module_states[module_name] = enable
        status = "enabled" if enable else "disabled"
        self.logger.info(f"Module {module_name} {status}")

        # Reset processing order
        self.processing_order = None
        self.processing_order = self._determine_processing_order()
    
    def disable_module(self, module_name: str) -> None:
        """To keep the functionality clear and seperate"""
        self.enable_module(module_name, False)
    
    def initialize(self) -> bool:
        """Initialize all registered modules"""
        if self._is_initialized:
            self.logger.warning(f"SystemController already initialized")
            return True
        
        self.logger.info("Initializing SystemController")
        success = True
        init_order = list(self.modules.keys())

        # Add all modules
        # Camera
        self.add_module(RealSenseCamera(self.config, SystemData.CAMERA_NAME))
        
        # Estimators
        self.add_module(MiDaSDepthEstimator(self.config, SystemData.MIDAS_NAME))
        if not self.USING_LAPTOP:
            from system.modules.DepthProEstimator import DepthProEstimator
            from system.modules.VGGTEstimator import VGGTEstimator
            self.add_module(DepthProEstimator(self.config, SystemData.PRO_NAME))
            self.add_module(VGGTEstimator(self.config, SystemData.VGGT_NAME))

        # Normalizer
        self.add_module(DepthNormalizer(self.config,
                                        SystemData.NORM_NAME,
                                        self.config.normalizer_stream_map))
        
        # Object detectors
        for yolo_config in self.config.yolo_configurations:
            self.add_module(YOLODetector(self.config, yolo_config.get("name"), yolo_config.get("model_path"), yolo_config.get("stream_map")))
        
        # Tracker
        self.add_module(TrackingModule(self.config, SystemData.TRACKER_NAME, self.config.tracker_stream_map))

        # Saver
        self.add_module(FrameSaver(self.config, SystemData.SAVE_NAME))

        # Visualization
        self.add_module(VisualizationModule(self.config, SystemData.VIS_NAME))


        # Initialize modules
        for name in self.modules:
            module = self.modules[name]
            self.logger.info(f"Initializing {module}...")
            try:
                if not module.initialize(self.config):
                    self.logger.error(f"Error initializing module: {module.name}")
                    success = False
                else:
                    self.logger.debug(f"{module.name} succesfully initialized")
            except Exception as e:
                self.logger.exception(f"Exception during intialization of {module.name}: {e}")
                success = False
        
        # Set active modules
        self.module_states = self.config.module_active_states.copy()

        # Find processing order
        if success:
            try:
                self.processing_order = self._determine_processing_order()
                self.logger.debug(f"Determined processing order: {self.processing_order}")
            except Exception as e:
                self.logger.exception(f"Failed to determine processing order: {e}")
                success = False
            self.logger.info("SystemController intializatin complete")
        else:
            self.logger.error("SystemController initialization failed")
        
        return success
    
    def _determine_processing_order(self) -> List[str]:
        """Determines valid processing order with Kahn's topological sort"""
        self.logger.debug(f"Determining processing order...")

        # Dictionaries for storing dependices as the graph builds
        module_inputs: Dict[str, Set[str]] = {}
        module_outputs: Dict[str, Set[str]] = {}
        output_map: Dict[str, str] = {} # output -> module
        enabled_modules = {name for name, enabled in self.module_states.items() if enabled}

        added_and_enabled_modules = {
            name for name, enabled in self.module_states.items() if enabled and name in self.modules
        }

        self.logger.debug(f"Currently active modules: {added_and_enabled_modules}")

        # Populate dictionaries
        for name in added_and_enabled_modules:
            if name not in self.modules:
                raise ValueError(f"Module {name} not found in SystemController")
            module = self.modules[name]
            outputs = module.get_outputs()
            module_outputs[name] = outputs
            module_inputs[name] = module.get_dependency_inputs()
            for data_type in outputs:
                if data_type in output_map:
                    self.logger.warning(f"System has multiple sources for {data_type}. Might cause issues.")
                output_map[data_type] = name
        
        # Dependency dictionaries
        adj: Dict[str, List[str]] = {name: [] for name in enabled_modules} # Module -> Modules that depends upon it
        in_degree: Dict[str, int] = {name: 0 for name in enabled_modules} # How far out in the order

        # Populate them
        for name in enabled_modules:
            req_in = module_inputs[name]
            for req in req_in:
                producer = output_map.get(req)
                if producer and producer != name and producer in enabled_modules:
                    if name not in adj[producer]:
                        adj[producer].append(name)
                        in_degree[name] += 1
        
        # Apply Kahn's algorithm
        queue: List[str] = [name for name in enabled_modules if in_degree[name] == 0]
        order: List[str] = []

        while queue:
            u = queue.pop(0)
            order.append(u)

            for v in sorted(adj[u]):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Check for cycles
        if len(order) != len(enabled_modules):
            cycle_modules = {name for name in enabled_modules if in_degree[name] > 0}
            raise RuntimeError(f"SystemController contains cyclical dependecies: {cycle_modules}")
        
        # Shuffle primary source to front of sorting
        if self._primary_source_name in enabled_modules:
            if self._primary_source_name in order:
                order.remove(self._primary_source_name)
            order.insert(0, self._primary_source_name)

        self.logger.info(f"")

        return order

    def get_primary_source_outputs(self) -> Set[str]:
        """Returns data types produced by primary source"""
        primary = self.modules.get(self._primary_source_name)
        return primary.get_outputs()
    
    def _update_required_streams(self) -> None:
        """Update which data streams are required by the system"""
        self._required_streams = set()
        for name, enabled in self.module_states.items():
            if enabled and name in self.modules:
                self._required_streams.update(self.modules[name].get_required_inputs())

    def start(self) -> bool:
        """Attempts to start the system and processing thread"""
        if self.is_running:
            self.logger.warning(f"SystemController already running")
            return True
        
        if self.processing_order is None:
            try:
                self.processing_order = self._determine_processing_order()
                self.logger.debug(f"Processing order set: {self.processing_order}")
            except Exception as e:
                self.logger.error(f"Error determining processing order: {e}")
                return False
            
        self.logger.info("Starting SystemController...")
        try:
            self.stop_event.clear()
            target_func = self._processing_loop
            self.logger.info(f"Attempting to start thread with target: {target_func}")
            self.processing_thread = threading.Thread(target=target_func, name="Processing Thread")
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info(f"Call to processing_thread.start() returned. Is thread alive? {self.processing_thread.is_alive()}")
            self.is_running = True
            self.logger.info("SystemController started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting SystemController: {e}")
            return False
        
    def stop(self) -> None:
        """Stops processing thread"""
        if not self.is_running and not self.stop_event.is_set():
            self.logger.info("SystemController not running")
            return

        self.logger.info("Stopping SystemController")
        self.stop_event.set()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        self.processing_thread = None

        for name, module in self.modules.items():
            try:
                module.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")

        self.is_running = False
        self.logger.info(f"SystemController stopped")

    def set_view(self, view_key: str, active: bool, vis_name: str= SystemData.VIS_NAME) -> None:
        """Set status of data streams in visualizer"""
        if vis_name not in self.modules:
            self.logger.warning(f"Visualizer not found, cant set view {vis_name}")
            return
        
        visualizer = self.modules.get(vis_name)
        visualizer.set_view(view_key, active)

    def toggle_view(self, view_key: str, vis_name: str= "Visualization_Module") -> None:
        """Toggle status of data streams in visualizer"""
        if vis_name not in self.modules:
            self.logger.warning(f"Visualizer not found, cant set view {vis_name}")
            return
        
        visualizer = self.modules.get(vis_name)
        visualizer.toggle_view(view_key)

    def set_tracker(self, input_key: str, active: bool, reset_tracker: bool = False, tracker_module_name: str = "TrackingModule") -> None:
        """Set status of tracking stream""" 
        if tracker_module_name not in self.modules:
            self.logger.warning(f"Tracker not found, cant set tracking of {input_key}")
            return
        
        tracker = self.modules.get(tracker_module_name)
        tracker.set_tracking(input_key, active, reset_tracker)
    
    def get_active_trackers(self, tracker_module_name: str = "TrackingModule") -> Set[str]:
        """Retrieve currently active trackers"""
        if tracker_module_name not in self.modules:
            self.logger.warning(f"Tracker not found, cant retrieve active views")
            return
        
        tracker = self.modules.get(tracker_module_name)
        return tracker.get_active()

    def request_save(self, saver_module_name: str = SystemData.SAVE_NAME) -> None:
        """Send request to saver module"""
        if saver_module_name not in self.modules:
            self.logger.warning(f"Saver not found, cant send request")
            return
        
        saver = self.modules.get(saver_module_name)
        saver.request_save()


    def get_current_data(self) -> Dict[str, Any]:
        """Retrieve a copy of the current SystemData"""
        with self.data_lock:
            return self.current_data.copy()
        
    def _processing_loop(self) -> None:
        """Main processing loop"""
        #self.logger.debug("Starting SystemController proccesing loop")
        try:    
            if not self.processing_order:
                self.logger.error("SystemController processing loop cannot run without valid processing order")
                return
            
            while not self.stop_event.is_set():
                cycle_start_time = time.perf_counter()
                cycle_data: Dict[str, Any] = {}
                cycle_data[SystemData.SYS_FRAME_ID] = self.sys_frame_counter

                try:
                    # Run modules processing loops in correct order
                    for name in self.processing_order:
                        if self.stop_event.is_set(): break

                        module = self.modules.get(name)
                        if not module or not self.module_states.get(name):
                            continue

                        #self.logger.debug(f"Getting module required input in processing loop")
                        req = module.get_required_inputs()
                        if not req.issubset(cycle_data.keys()):
                            missing_data = req - cycle_data.keys()
                            self.logger.debug(f"Skipping {name}, missing data: {missing_data}") 
                            continue

                        #self.logger.debug(f"Creating input subsets")
                        dep = module.get_dependency_inputs()
                        input_subset = {key: cycle_data[key] for key in dep if key in cycle_data}

                        try:
                            #self.logger.debug(f"Processing data with {name}")
                            new_data = module.process(input_subset)

                            if new_data:
                                cycle_data.update(new_data)
                        except Exception as e:
                            self.logger.error(f"Error processing {name}: {e}")
                        
                    if self.stop_event.is_set(): break
                    
                    # Update current data directory
                    with self.data_lock:
                        self.current_data = cycle_data
                
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    time.sleep(0.5)
                
                self.sys_frame_counter += 1
                active_modules = [module for module, state in self.module_states.items() if state]
                self.logger.debug(f"Active modules: {active_modules}")
                self.logger.debug(f"Current available data: {cycle_data.keys()}")
                cycle_end_time = time.perf_counter()
                # TODO: complete cycle timing logic
        except Exception as e:
            self.logger.error(f"Error in internal processing loop: {e}")
            return 
        