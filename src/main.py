# src/main.py

from system.SystemController import SystemController
from system.SystemData import SystemData
from system.modules.RealSenseCamera import RealSenseCamera
from system.modules.YOLODetector import YOLODetector
from system.modules.MiDaSDepthEstimator import MiDaSDepthEstimator
from system.modules.DepthProEstimator import DepthProEstimator

import logging
import cv2
import time
import numpy as np

from config import Config
from structs.Detection import Detection

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting main()")
    controller = None

    import pythoncom
    pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)

    try:
        config = Config()
        controller = SystemController(config)

        #------------------------------------------#
        #              Module names                #
        #------------------------------------------#

        CAM_NAME = "RealSense_Camera"
        MIDAS_NAME = "MiDaS_estimator"
        DEPTHPRO_NAME = "DepthPro_estimator"
        YOLO_DEPTH_NAME = "YOLO"
        YOLO_MIDAS_NAME = "YOLO_MiDaS"
        YOLO_DEPTHPRO_NAME = "YOLO_DepthPro"

        #------------------------------------------#
        #       Add modules to controller          #
        #------------------------------------------#
        controller._primary_source_name = CAM_NAME
        controller.add_module(RealSenseCamera(config, CAM_NAME))
        controller.add_module(MiDaSDepthEstimator(config, MIDAS_NAME))
        controller.add_module(DepthProEstimator(config, DEPTHPRO_NAME))
        controller.add_module(YOLODetector(config, YOLO_DEPTH_NAME))
        controller.add_module(YOLODetector(config,
                                           YOLO_MIDAS_NAME,
                                           SystemData.COLOR,
                                           SystemData.MIDAS_ESTIMATED_DEPTH))
        controller.add_module(YOLODetector(config,
                                           YOLO_DEPTHPRO_NAME,
                                           SystemData.COLOR,
                                           SystemData.PRO_ESTIMATED_DEPTH))
        
        # Initialize controller
        if not controller.initialize():
            logging.critical("Failed to intialize SystemController")
            return


        #------------------------------------------#
        #             Enable modules               #
        #------------------------------------------#
        controller.enable_module(CAM_NAME)
        #controller.enable_module(YOLO_DEPTH_NAME)
        controller.enable_module(MIDAS_NAME)
        #controller.enable_module(YOLO_MIDAS_NAME)
        #controller.enable_module(DEPTHPRO_NAME)
        #controller.enable_module(YOLO_DEPTHPRO_NAME)


        # Start controller
        if not controller.start():
            logging.critical("Failed to start system")
            return
        
        time.sleep(1.0)

        #------------------------------------------#
        #             Display window               #
        #------------------------------------------#
        window_name = "Synthetic depth detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            current_data = controller.get_current_data() 

            #if not current_data: 
                 #time.sleep(0.05) 
                 #continue

            #logging.debug(f"Current data: {list(current_data.keys())}")
            display_image = current_data[SystemData.DEPTH]
            
            # Display
            if display_image is not None:
                cv2.imshow(window_name, display_image)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for data...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, placeholder)

            # Handle User Input
            key = cv2.waitKey(10) & 0xFF 
            if key == ord('q'):
                logging.info("Quit key pressed. Exiting loop.")
                break
    
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected, exiting program")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if controller and controller.is_running:
            controller.stop()
            cv2.destroyAllWindows()

if __name__=="__main__":
    main()
    