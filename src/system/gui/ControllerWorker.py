# src/system/gui/ControllerWorker.py

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
import numpy as np
import time
from typing import Any

from ..SystemController import SystemController
from ..SystemData import SystemData

class ControllerWorker(QObject):
    # Signals to GUI
    new_montage_ready = pyqtSignal(np.ndarray)
    status_message = pyqtSignal(str)
    system_started_signal = pyqtSignal()
    system_stopped_signal = pyqtSignal()

    def __init__(self, controller: SystemController, config: Any):
        super().__init__()
        self.controller = controller
        self.config = config
        self.is_running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.processing_step)

    @pyqtSlot()
    def start_system(self):
        if self.is_running:
            self.status_message.emit("System already running")
            return

        self.status_message.emit("Starting SystemController...")
        if not self.controller.start():
            self.status_message.emit("CRITICAL: SystemController start failed")
            return
        
        self.is_running = True
        self.system_started_signal.emit()
        self._timer.start(100)

    @pyqtSlot()
    def stop_system(self):
        if not self.is_running:
            self.status_message.emit("System not running")
            return
        
        self.is_running = False
        self._timer.stop()
        self.status_message.emit("Stopping SystemController...")
        self.controller.stop()
        self.status_message.emit("SystemController stopped")
        self.system_stopped_signal.emit()

    def processing_step(self):
        """Standard interaction between GUI and SystemController"""
        if not self.is_running or not self.controller.is_running:
            self.status_message.emit("System not running")
            return
        
        current_data = self.controller.get_current_data()
        if not current_data:
            return
        
        # Prepare info about current controller status
        active_modules = [name for name, state in self.controller.module_states.items() if state and name in self.controller.modules]
        vis_module = self.controller.modules.get(SystemData.VIS_NAME)
        active_views_str = []
        if vis_module:
            active_views_str = vis_module.get_active_views()
        
        status_info_package = {
            'Active Modules': active_modules if active_modules else "None",
            'Active Views': active_views_str if active_views_str else "None"
        }
        current_data[SystemData.SYS_STATUS_INFO] = status_info_package

        montage = current_data.get(SystemData.VIS_MONTAGE)
        if montage is not None:
            self.new_montage_ready.emit(montage)

    @pyqtSlot(str, bool)
    def gui_toggle_module(self, module_name: str, enable: bool):
        """Sets status of module in controller"""
        if module_name in self.controller.modules:
            self.controller.enable_module(module_name, enable)
        else:
            self.status_message.emit(f"Module not found, couldnt toggle: {module_name}")

    @pyqtSlot(str, bool)
    def gui_toggle_view(self, view_key: str, enable: bool):
        """Sets status of visualiser view"""
        if SystemData.VIS_NAME in self.controller.modules:
            self.controller.set_view(view_key, enable)
        else:
            self.status_message.emit(f"Visalizer not found, couldnt toggle: {view_key}")
    
    @pyqtSlot()
    def request_save(self):
        """Rewuest save from Frame Saver"""
        if SystemData.SAVE_NAME in self.controller.modules:
            self.controller.request_save()
            self.status_message.emit("Frame save requested")
        else:
            self.status_message.emit("Frame Saver not found")