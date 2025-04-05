# src/modules/SystemModule.py

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Set, Deque
import time
from collections import deque
from statistics import mean

class SystemModule(ABC):
    """
    Standard interface for system modules
    """
    def __init__(self, config: Any, module_name: str, timing_buffer_size: int = 30):
        """
        Base initializer, for the most standard interface possible
        """
        self._config = config
        self._name = module_name
        self._processing_times: Deque[float] = deque(maxlen= timing_buffer_size)

    @property
    def name(self) -> str:
        """A unique name for this component instance"""
        return self._name

    @abstractmethod
    def initialize(self, config: Any) -> bool:
        """Initialize module with config file"""
        pass

    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Public process wrapper for timing and analyzing purposes
        Refer to internal process function for specific processing logic
        """
        start_time = time.perf_counter()
        try:
            result = self._process_internal(data)
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self._processing_times.append(elapsed_time)
        
        return result
    
    @abstractmethod
    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Module processing logic
        This is the one that should be implemented by subclasses
        """
        pass

    @abstractmethod
    def get_required_inputs(self) -> Set[str]:
        """Set of StreamType strings that the module needs to function"""
        pass

    def get_dependency_inputs(self) -> Set[str]:
        """Set of SystemData the module might or can use"""
        pass

    @abstractmethod
    def get_outputs(self) -> Set[str]:
        """Set of StreamType strings the module produces"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the module cleanly"""
        pass

    def get_last_processing_time(self) -> Optional[float]:
        """Returns duration of last processing run"""
        if not self._processing_times:
            return None
        return self._processing_times[-1]
    
    def get_average_processing_time(self) -> Optional[float]:
        """Returns average processing time"""
        if not self._processing_times:
            return None
        return mean(self._processing_times)