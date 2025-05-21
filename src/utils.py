# training_data/apply_estimation.py

from system.modules.MiDaSDepthEstimator import MiDaSDepthEstimator
from system.modules.DepthProEstimator import DepthProEstimator
from system.modules.DepthNormalizer import DepthNormalizer
from system.modules.VGGTEstimator import VGGTEstimator
from config import Config
from system.SystemData import SystemData

from typing import Any
import os
import cv2
import logging

def apply_estimation(input_dir: str, run_name: str, estimator: Any, normalizer: Any = None, normalizer_output_key: str = None):
    """Applies estimator to all files in a folder and saves in adjacent folder"""
    output_dir = os.path.join(os.path.dirname(input_dir), run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Save dir: {output_dir}")

    # Iteratively populate images dict
    i = 0
    for filename in os.listdir(input_dir):
        i += 1
        if i < 87:
            continue
        else:
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
                    output_map = normalizer._process_internal(estimation_output)
                else:
                    output = estimation_output

                # Prepare and write output
                output_image_data = output_map[normalizer_output_key]
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
    vggt = VGGTEstimator(config, "VGGT")
    normalizer = DepthNormalizer(config, 
                                "normalizer",
                                {SystemData.MIDAS_ESTIMATED_DEPTH: {SystemData.NORM_MIDAS: True},
                                SystemData.PRO_ESTIMATED_DEPTH: {SystemData.NORM_PRO: False},
                                SystemData.VGGT_ESTIMATED_DEPTH: {SystemData.NORM_VGGT: False}})
    #midas.initialize(config)
    #depth_pro.initialize(config)
    vggt.initialize(config)
    normalizer.initialize(config)
    run_num = "run_002"
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_path, "training_data", "real_world", run_num, "color")

    #apply_estimation(input_dir, "midas_norm", midas, normalizer, SystemData.NORM_MIDAS)
    #apply_estimation(input_dir, "depthpro_norm", depth_pro, normalizer, SystemData.NORM_PRO)
    apply_estimation(input_dir, "vggt_norm", vggt, normalizer, SystemData.NORM_VGGT)


if __name__ == "__main__":
    main()


    
