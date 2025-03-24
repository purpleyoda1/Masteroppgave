# src/main.py

import os
import cv2
from config import Config
from modules.RealSenseCamera import RealSenseCamera
from modules.SystemController import SystemController, StreamType
from modules.YOLODetector import YOLODetector
from modules.MiDaSDepthEstimator import MiDaSDepthEstimator
import logging
import time

def main():
    # Configure logging level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("TESTING: Debug logging is enabled")
    logging.info("TESTING: Info logging is enabled")

    try:
        import pythoncom
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        logging.info("COM initialized with apartment threading model")
    except ImportError:
        logging.info("pythoncom not available")

    # Load config
    config = Config()

    # Create system controller
    controller = SystemController(config)

    # Initialize components and add to controller
    camera = RealSenseCamera(config)
    controller.set_camera(camera)

    yolo_detector = YOLODetector(config)
    controller.set_yolo_detector(yolo_detector)

    yolo_DE_detector = YOLODetector(config, DE= True)
    controller.set_yolo_DE_detector(yolo_DE_detector)

    depth_estimator = MiDaSDepthEstimator(config)
    controller.set_depth_estimator(depth_estimator)

    # Initialize system
    if not controller.initialize():
        logging.error("Failed to initialize controller")
        return
    


    ########################################
    #             DATA STREAMS             #
    ########################################
    controller.enable_stream(StreamType.COLOR)
    controller.enable_stream(StreamType.DEPTH)
    #controller.enable_stream(StreamType.DEPTH_COLORMAP)
    #controller.enable_stream(StreamType.DEPTH_DETECTIONS)
    #controller.enable_stream(StreamType.DEPTH_COLORMAP_DETECTIONS)
    controller.enable_stream(StreamType.ESTIMATED_DEPTH)
    controller.enable_stream(StreamType.ESTIMATED_DEPTH_DETECTIONS)



    ########################################
    #               DETECTORS              #
    ########################################
    #controller.enable_detector("yolo")
    controller.enable_detector("yoloDE")



    # Start system
    controller.start()
    logging.info("Waiting for camera to initialize...")
    time.sleep(1.0) 

    # Create visualization window
    cv2.namedWindow("Detection system", cv2.WINDOW_NORMAL)

    try:
        while True:
            data = controller.get_current_data()
            logging.debug(f"Current data types: {list(data.keys())}")

            display_image = data[StreamType.ESTIMATED_DEPTH_DETECTIONS]

            cv2.imshow("Detection system", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        controller.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()