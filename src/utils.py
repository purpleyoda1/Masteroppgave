# training_data/apply_estimation.py

from system.modules.MiDaSDepthEstimator import MiDaSDepthEstimator
from system.modules.DepthProEstimator import DepthProEstimator
from system.modules.DepthNormalizer import DepthNormalizer
from config import Config
from system.SystemData import SystemData

from typing import Any
import os
import cv2
import logging

def apply_estimation(input_dir: str, run_name: str, estimator: Any, normalizer: Any = None):
    """Applies estimator to all files in a folder and saves in adjacent folder"""
    output_dir = os.path.join(os.path.dirname(input_dir), run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Save dir: {output_dir}")

    # Iteratively populate images dict
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if os.path.exists(image_path):
            # load image
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Prepare input on expected form
            estimation_input = {
                SystemData.COLOR: image_rgb
            }

            # Apply estimation and normalization
            estimation_output = estimator._process_internal(estimation_input)
            if normalizer:
                output = normalizer._process_internal(estimation_output)
            else:
                output = estimation_output

            # Prepare and write output
            output_image_data = list(output.values())[0]
            output_path = os.path.join(output_dir, filename)
            output_bgr = cv2.cvtColor(output_image_data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_image_data)
            print(f"Saved to: {output_path}")
        else:
            print(f"Image not found")

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting main()")

    # Set up modules
    config = Config()
    midas = MiDaSDepthEstimator(config, "MiDaS")
    depth_pro = DepthProEstimator(config, "DepthPro")
    midas_normalizer = DepthNormalizer(config, 
                                       input_stream_type=SystemData.MIDAS_ESTIMATED_DEPTH, 
                                       output_stream_type=SystemData.NORM_MIDAS,
                                       invert=True)
    depthpro_normalizer = DepthNormalizer(config, 
                                       input_stream_type=SystemData.PRO_ESTIMATED_DEPTH, 
                                       output_stream_type=SystemData.NORM_PRO)
    
    midas.initialize(config)
    depth_pro.initialize(config)
    midas_normalizer.initialize(config)
    depthpro_normalizer.initialize(config)

    run_num = "run_000"
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_path, "training_data", "real_world", run_num, "color")

    #apply_estimation(input_dir, "midas_norm", midas, midas_normalizer)
    apply_estimation(input_dir, "depthpro_norm", depth_pro, depthpro_normalizer)

if __name__ == "__main__":
    main()


    
