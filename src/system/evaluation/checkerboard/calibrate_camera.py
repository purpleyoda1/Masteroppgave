import numpy as np
import cv2
import glob
import os
from typing import Tuple

def calibrate_camera(image_folder, intrinsic_path:str, checkerboard_size: Tuple[int, int], square_size_mm: float):
    """Runs checkerboard calibration on images in input folder"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape(-1, 2)
    objp = objp * square_size_mm

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_folder, '*.png'))
    if not images:
        print(f"Error: No images found in {image_folder}")
        return
    print(f"Found {len(images)} images for calibration")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (checkerboard_size[1], checkerboard_size[0]), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"Chekcerboard not found in {fname}")

    
    if not objpoints:
        print(f"Checkerboard not found in any images")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\nCalibration Successful!")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)

        output_file = intrinsic_path
        np.savez(output_file, mtx=mtx, dist=dist)
        print(f"\nIntrinsics saved to {output_file}")
    else:
        print("Calibration failed.")

def main():
    # PARAMETERS #
    script_folder = os.path.dirname(os.path.abspath(__file__))
    calibration_image_folder = os.path.join(script_folder, 'calibration_images')
    intrinsic_path = os.path.join(script_folder, 'data', 'camera_intrinsics.npz')

    checkerboard_rows = 6
    checkerboard_cols = 9

    square_size_mm = 25
    # --------- #

    calibrate_camera(
        image_folder=calibration_image_folder,
        intrinsic_path=intrinsic_path,
        checkerboard_size=(checkerboard_rows, checkerboard_cols),
        square_size_mm=square_size_mm
    )

if __name__ == "__main__":
    main()
