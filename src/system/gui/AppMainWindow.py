# src/system/gui/main_window.py

from PyQt6 import uic
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QCheckBox, QLabel, QScrollArea, QGroupBox, QStatusBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import os
import inspect
from typing import Any
import logging

from ..SystemData import SystemData
from config import Config
from .ControllerWorker import ControllerWorker

class AppMainWindow(QMainWindow):
    """Main application window"""
    # Signals to Controller worker
    start_system_signal = pyqtSignal()
    stop_system_signal = pyqtSignal()
    toggle_module_signal = pyqtSignal(str, bool)
    toggle_view_signal = pyqtSignal(str, bool)
    request_save_signal = pyqtSignal()

    def __init__(self, config: Config, worker: ControllerWorker, worker_thread: QThread):
        super().__init__()
        self.config = config
        self.worker = worker
        self.worker_thread = worker_thread
        self.logger = logging.getLogger("AppWindow")

        ui_file_path = os.path.join(os.path.dirname(__file__), "window_layout.ui")
        if not os.path.exists(ui_file_path):
            self.logger.error(f"UI file not found at: {ui_file_path}")
            return
        
        uic.loadUi(ui_file_path, self)
        self.setWindowTitle("System Control Panel")

        # Define module and views to connect to checkboxes
        self.module_checkbox_object_names = [
            SystemData.MIDAS_NAME,
            SystemData.PRO_NAME,
            SystemData.YOLO_DEPTH_NAME,
            SystemData.YOLO_NORM_NAME,
            SystemData.NORM_NAME,
            SystemData.TRACKER_NAME,
            SystemData.SAVE_NAME,
            SystemData.VIS_NAME
        ]

        self.view_checkbox_object_names = [
            SystemData.COLOR,
            SystemData.DEPTH,
            SystemData.NORM_DEPTH,
            SystemData.VIS_DEPTH_DETECTIONS,
            SystemData.VIS_DEPTH_TRACKED_DETECTIONS,
            SystemData.MIDAS_ESTIMATED_DEPTH,
            SystemData.NORM_MIDAS,
            SystemData.VIS_MIDAS_DETECTIONS,
            SystemData.VIS_MIDAS_TRACKED_DETECTIONS,
            SystemData.PRO_ESTIMATED_DEPTH,
            SystemData.NORM_PRO,
            SystemData.VIS_PRO_DETECTIONS,
            SystemData.VIS_PRO_TRACKED_DETECTIONS
        ]

        # Connect signals from UI to slots
        if hasattr(self, "start_button"): self.start_button.clicked.connect(self.start_system_signal)
        if hasattr(self, "stop_button"): self.stop_button.clicked.connect(self.stop_system_signal)
        if hasattr(self, "save_button"): self.save_button.clicked.connect(self.request_save_signal)

        # Connect module checkboxes
        for module_name in self.module_checkbox_object_names:
            checkbox = getattr(self, module_name, None)
            if checkbox:
                initial_state =  self.config.module_active_states.get(module_name, False)
                checkbox.setChecked(initial_state)
                checkbox.stateChanged.connect(
                    lambda state_val, name=module_name: self.handle_single_module_toggle(name, state_val)
                    )
            else:
                self.logger.warning(f"Checkbox not found in UI: {module_name}")
        
        # Connect view checkboxes
        default_active_views = self.config.vis_initial_active_views
        for view_name in self.view_checkbox_object_names:
            checkbox = getattr(self, view_name, None)
            if checkbox:
                checkbox.setChecked(view_name in default_active_views)
                checkbox.stateChanged.connect(
                    lambda state_val, name=view_name: self.handle_view_toggle(name, state_val)
                    )
                
        # Connect worker signals to GUI slots
        self.worker.new_montage_ready.connect(self.display_montage)
        self.worker.status_message.connect(self.show_status_message)
        self.worker.system_started_signal.connect(self.on_system_start)
        self.worker.system_stopped_signal.connect(self.on_system_stop)

        # Connect GUI signals to worker slot
        self.start_system_signal.connect(self.worker.start_system)
        self.stop_system_signal.connect(self.worker.stop_system)
        self.toggle_module_signal.connect(self.worker.gui_toggle_module)
        self.toggle_view_signal.connect(self.worker.gui_toggle_view)
        self.request_save_signal.connect(self.worker.request_save)

        # Set up initial config
        self.show_status_message("Ready")
        self.sync_initial_gui_and_worker()
        

    def sync_initial_gui_and_worker(self):
        """Sync initial enabling between GUI and Controller"""
        # modules
        for module_name in self.module_checkbox_object_names:
            checkbox = getattr(self, module_name, None)
            if checkbox:
                self.handle_single_module_toggle(module_name, checkbox.checkState().value)

        # Views
        for view_name in self.view_checkbox_object_names:
            checkbox = getattr(self, view_name, None)
            if checkbox:
                self.handle_view_toggle(view_name, checkbox.checkState().value)


    def handle_single_module_toggle(self, name, enable):
        self.toggle_module_signal.emit(name, enable)
        self.logger.debug(f"Module {name} status set: {enable}")

    def handle_view_toggle(self, name, enable):
        self.toggle_view_signal.emit(name, enable)
        self.logger.debug(f"View {name} status set: {enable}")

    @pyqtSlot(np.ndarray)
    def display_montage(self, image: np.ndarray):
        """ Prepares and displays montage """
        display_widget = getattr(self, "montage_display", None)
        if not display_widget:
            self.logger.error(f"montage_display not found in UI")
            return
        
        if image is None or image.size == 0:
            display_widget.setText("No image found")
            display_widget.setPixmap(QPixmap())
            return
        
        try:
            # Attempt conversion from cv2 array to pyqt
            h, w = 0, 0
            ch = 0
            qt_image_format = None
            if image.ndim == 3 and image.shape[2] == 3:
                h, w, ch = image.shape
                bytes_per_line = ch * w
                contiguous_image = np.ascontiguousarray(image)
                qt_image = QImage(contiguous_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            else:
                display_widget.setText(f"Unsopported image format: {h}, {w}, {ch}")
                return

            pixmap = QPixmap.fromImage(qt_image)

            # Scale appropriately
            pixmap_scaled = pixmap.scaled(
                display_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            #display_widget.setScaledContents(True)
            display_widget.setPixmap(pixmap_scaled)

            #display_widget.setPixmap(pixmap.scaled(
            #    display_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            #))

        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            display_widget.setText(f"Error displaying image")

    @pyqtSlot(str)
    def show_status_message(self, message: str):
        """Shows status message from worker"""
        status_widget = getattr(self, "status_message_display", None)
        if status_widget:
            status_widget.setText(message)
    
    @pyqtSlot()
    def on_system_start(self):
        if hasattr(self, "start_button"): self.start_button.setEnabled(False)
        if hasattr(self, "stop_button"): self.stop_button.setEnabled(True)
        self.show_status_message("System started")

    
    @pyqtSlot()
    def on_system_stop(self):
        if hasattr(self, "start_button"): self.start_button.setEnabled(True)
        if hasattr(self, "stop_button"): self.stop_button.setEnabled(False)
        self.show_status_message("System stopped")
        if hasattr(self, "montage_display"):
            getattr(self, "montage_display").setText("System Stopped.")
            getattr(self, "montage_display").setPixmap(QPixmap())

    def closeEvent(self, event):
        """For closing pyqt thread"""
        self.logger.info(f"Close event detected, stopping system...")
        self.show_status_message(f"Stopping system")
        self.stop_system_signal.emit()
        if self.worker_thread.isRunning():
            if not self.worker_thread.wait(100):
                self.logger.warning(f"Worker thread did not close properly")
                self.worker_thread.terminate()
                self.worker_thread.wait()
        super().closeEvent(event)
