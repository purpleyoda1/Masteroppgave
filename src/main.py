# src/main.py

import sys
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

from system.SystemController import SystemController
from system.gui.AppMainWindow import AppMainWindow
from system.gui.ControllerWorker import ControllerWorker
from config import Config

USING_LAPTOP = True

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not USING_LAPTOP:
        import pythoncom
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)

    # Set up needed instances
    app = QApplication(sys.argv)

    config = Config()
    controller = SystemController(config, USING_LAPTOP)
    controller.initialize()

    controller_thread = QThread()
    controller_worker = ControllerWorker(controller, config)
    controller_worker.moveToThread(controller_thread)

    main_gui = AppMainWindow(config, controller_worker, controller_thread)

    # Start systems
    controller_thread.start()
    main_gui.show()
    exit_code = app.exec()

    # Cleanup
    if controller_thread.isRunning():
        logging.info("Requesting worker to stop from main exit...")
        controller_worker.stop_system()
        if not controller_thread.wait(100):
            logging.warning("Worker thread did not finish cleanly")
            controller_thread.terminate()
            controller_thread.wait()

    try:
        pythoncom.CoUninitialize()
    except Exception:
        pass

    sys.exit(exit_code)

if __name__ == "__main__":
    main()