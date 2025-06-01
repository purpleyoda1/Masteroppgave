# src/eval.py

import sys
import logging
import os
import time

from system.SystemController import SystemController
from system.modules.CameraImpostor import CameraImpostor
from system.modules.EvaluationModule import EvaluationModule
from system.SystemData import SystemData
from config import Config
from typing import Dict
from dataclasses import field

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(SystemData.EVAL_NAME)
    config = Config()

    config.evaluation_mode = True
    config.module_active_states = {
        # Camera
        SystemData.CAMERA_NAME: True,
        # Estimator
        SystemData.MIDAS_NAME: False,
        SystemData.PRO_NAME: False,
        SystemData.VGGT_NAME: False,
        # Normalizers
        SystemData.NORM_NAME: True,
        # YOLO Detectors
        SystemData.YOLO_DEPTH_NAME: True,
        SystemData.YOLO_NORM_NAME: True,
        SystemData.YOLO_COLOR_NAME: False,
        # Detection tracker
        SystemData.TRACKER_NAME: False,
        # Visualization
        SystemData.VIS_NAME: True,
        # Frame Saver
        SystemData.SAVE_NAME: False,
        # Evaluation
        SystemData.CAMERA_IMPOSTOR_NAME: True,
        SystemData.EVAL_NAME: True,
    }

    config.normalizer_stream_map  =  {
        SystemData.DEPTH: {SystemData.NORM_DEPTH: False},
    }

    controller = SystemController(config)
    camera_impostor = CameraImpostor(config, module_name=SystemData.CAMERA_IMPOSTOR_NAME)
    evaluation_module = EvaluationModule(config, module_name=SystemData.EVAL_NAME)

    controller.add_module(camera_impostor)
    controller.add_module(evaluation_module)
    controller.initialize()

    if not controller.start():
        logger.error(f"Error starting controller")
        return 1
    
    start_time = time.time()

    try:
        # Run until all images are processed
        frame_count = 0
        last_log_time = time.time()
        
        while controller.is_running:
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
            # Get current data to check progress
            current_data = controller.get_current_data()
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                if SystemData.GROUND_TRUTH_IMAGE_PATH in current_data:
                    current_image = current_data[SystemData.GROUND_TRUTH_IMAGE_PATH]
                    logger.info(f"Processing: {current_image} (frame {frame_count})")
                last_log_time = current_time
            
            # Check if evaluation is complete
            if current_data.get(SystemData.EVAL_COMPLETE):
                logger.info("Evaluation complete signal received")
                break
                
            frame_count += 1
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
    
    finally:
        # Clean shutdown
        logger.info("Shutting down evaluation system...")
        controller.stop()
        
        # Report timing
        elapsed_time = time.time() - start_time
        logger.info(f"Total evaluation time: {elapsed_time:.2f} seconds")
        
        # Report results location
        if hasattr(evaluation_module, 'output_dir'):
            logger.info(f"Results saved to: {evaluation_module.output_dir}")
    
    logger.info("Evaluation finished successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

