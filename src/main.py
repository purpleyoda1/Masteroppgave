# src/main.py

from system.SystemController import SystemController
from system.SystemData import SystemData
from system.modules.RealSenseCamera import RealSenseCamera
from system.modules.YOLODetector import YOLODetector
from system.modules.MiDaSDepthEstimator import MiDaSDepthEstimator
from system.modules.DepthProEstimator import DepthProEstimator
from system.modules.VisualizationModule import VisualizationModule
from system.modules.DepthNormalizer import DepthNormalizer


import logging
import cv2
import time
import numpy as np

from config import Config
from system.structs.Detection import Detection

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

        # Camera
        CAM_NAME = "RealSense_Camera"

        # Estimators
        MIDAS_NAME = "MiDaS_estimator"
        DEPTHPRO_NAME = "DepthPro_estimator"

        # Normalizers
        NORM_DEPTH = "Normalizer_Depth"
        NORM_MIDAS = "Normalizer_MiDaS"
        NORM_PRO = "Normalizer_DepthPro"

        # YOLO
        YOLO_DEPTH_NAME = "YOLO"
        YOLO_MIDAS_NAME = "YOLO_MiDaS"
        YOLO_DEPTHPRO_NAME = "YOLO_DepthPro"

        # VISUALIZER
        VIS_NAME = "Visualization_Module"

        #------------------------------------------#
        #       Add modules to controller          #
        #------------------------------------------#

        # Camera
        controller._primary_source_name = CAM_NAME
        controller.add_module(RealSenseCamera(config, CAM_NAME))

        # Estimators
        controller.add_module(MiDaSDepthEstimator(config, MIDAS_NAME))
        #controller.add_module(DepthProEstimator(config, DEPTHPRO_NAME))

        # Normalizers
        controller.add_module(DepthNormalizer(config, NORM_DEPTH,  SystemData.DEPTH, SystemData.NORM_DEPTH))
        controller.add_module(DepthNormalizer(config, NORM_MIDAS, SystemData.MIDAS_ESTIMATED_DEPTH, SystemData.NORM_MIDAS, invert=True))
        #controller.add_module(DepthNormalizer(config, NORM_PRO, SystemData.PRO_ESTIMATED_DEPTH, SystemData.NORM_PRO))

        # YOLO
        #controller.add_module(YOLODetector(config, YOLO_DEPTH_NAME))
        controller.add_module(YOLODetector(config, YOLO_MIDAS_NAME, SystemData.MIDAS_ESTIMATED_DEPTH, SystemData.MIDAS_DETECTIONS))
        #controller.add_module(YOLODetector(config, YOLO_DEPTHPRO_NAME, SystemData.PRO_ESTIMATED_DEPTH, SystemData.PRO_DETECTIONS))
        
        # Visualization
        controller.add_module(VisualizationModule(config, VIS_NAME, view_target_height=480))
        
        # Initialize controller
        if not controller.initialize():
            logging.critical("Failed to intialize SystemController")
            return


        #------------------------------------------#
        #             Enable views                 #
        #------------------------------------------#
        controller.set_view(SystemData.VIS_DEPTH_COLORMAP, True)
        controller.set_view(SystemData.VIS_MIDAS_COLORMAP, True)
        controller.set_view(SystemData.VIS_MIDAS_COLORMAP_DETECTIONS, True)
        


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

            if not current_data: 
                 time.sleep(0.05) 
                 continue

            logging.debug(f"Current data: {list(current_data.keys())}")
            display_image = current_data.get(SystemData.VIS_MONTAGE)
            
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
    