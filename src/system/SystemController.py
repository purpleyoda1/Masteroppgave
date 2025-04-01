# src/system/SystemController.py

from .SystemData import SystemData
from .SystemModule import SystemModule

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
import sys

class SystemController:
    """Central controller managing information flow and interaction between modules"""
    def __init__(self, config: Any):
        """Initialize controller"""
        self.config = config
        self.logger = logging.getLogger("SystemController")

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

    def add_module(self, module: SystemModule) -> None:
        """Register module in controller"""
        if module.name in self.modules:
            self.logger.warning(f"Module {module.name} already in controller")

        self.modules[module.name] = module
        self.module_states[module.name] = False
        self.logger.info(f"Module {module.name} added to SystemController")

        # Reset processing order
        self.processing_order = None
        self._determine_processing_order()

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
        self._determine_processing_order()
    
    def disable_module(self, module_name: str) -> None:
        """To keep the functionality clear and seperate"""
        self.enable_module(module_name, False)
    
    def initialize(self) -> bool:
        """Initialize all registered modules"""
        self.logger.info("Initializing SystemController")
        success = True
        init_order = list(self.modules.keys())

        # Initialize modules
        for name in init_order:
            module = self.modules[name]
            self.logger.info(f"Initializing {module.name}...")
            try:
                if not module.initialize(self.config):
                    self.logger.error(f"Error initializing module: {module.name}")
                    success = False
                else:
                    self.logger.debug(f"{module.name} succesfully initialized")
            except Exception as e:
                self.logger.exception(f"Exception during intialization of {module.name}: {e}")
                success = False
        
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

        # Populate dictionaries
        for name in enabled_modules:
            if name not in self.modules:
                raise ValueError(f"Module {name} not found in SystemController")
            module = self.modules[name]
            outputs = module.get_outputs()
            module_outputs[name] = outputs
            module_inputs[name] = module.get_required_inputs()
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
                elif not producer and req not in self.get_primary_source_outputs():
                    self.logger.warning(f"Module {name} requires {req} which is currently not provided")
        
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

    def enable_stream(self, stream_type: str) -> None:
        """Adds a SystemData type to the desired streams"""
        self.desired_streams.add(stream_type)
        self.logger.info(f"External stream of {stream_type} enabled")
    
    def disable_stream(self, stream_type: str) -> None:
        """Removes SystemData streamtype from desired streams"""
        if stream_type in self.desired_streams:
            self.desired_streams.remove(stream_type)
            self.logger.info(f"External stream of {stream_type} disabled")

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
                        input_subset = {key: cycle_data[key] for key in req}

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
                
                cycle_end_time = time.perf_counter()
                # TODO: complete cycle timing logic
        except Exception as e:
            self.logger.error(f"Error in internal processing loop: {e}")
            return 
        










    def _simple_test_loop(self):
        my_name = threading.current_thread().name
        print(f"!!! THREAD {my_name}: Entered _simple_test_loop !!! [Direct Print]", file=sys.stderr, flush=True)
        self.logger.critical(f"!!! THREAD {my_name}: Entered _simple_test_loop !!! [Logging]")
        counter = 0
        try:
            while not self.stop_event.is_set() and counter < 50: # Limit loops for testing
                print(f"THREAD {my_name}: Simple loop iteration {counter}", file=sys.stderr, flush=True)
                self.logger.info(f"THREAD {my_name}: Simple loop iteration {counter}")
                counter += 1
                time.sleep(0.1) # Sleep briefly
            print(f"THREAD {my_name}: Simple loop finished.", file=sys.stderr, flush=True)
            self.logger.info(f"THREAD {my_name}: Simple loop finished.")
        except Exception as e:
             print(f"THREAD {my_name}: *** SIMPLE LOOP EXCEPTION ***: {e}", file=sys.stderr, flush=True)
             self.logger.exception(f"THREAD {my_name}: Exception in simple loop: {e}")
        finally:
            print(f"!!! THREAD {my_name}: Exiting _simple_test_loop (finally block) !!! [Direct Print]", file=sys.stderr, flush=True)
            self.logger.critical(f"!!! THREAD {my_name}: Exiting _simple_test_loop (finally block) !!!")