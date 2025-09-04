import numpy as np
import cv2
import os
from typing import Tuple

def estimate_pose(
    image_path: str, 
    intrinsic_file: str, 
    checkerboard_size: Tuple[int, int],
    square_size_mm: float, 
    output_folder: str
    ):
    
    os.makedirs(output_folder, exist_ok=True)

    with np.load(intrinsic_file) as data:
        mtx = data['mtx']
        dist = data['dist']
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pattern_size_wh = (checkerboard_size[1], checkerboard_size[0])

    ret, corners = cv2.findChessboardCorners(gray, pattern_size_wh, None)

    if ret:
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size_wh[0], 0:pattern_size_wh[1]].T.reshape(-1, 2)
        objp = objp * square_size_mm
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        if ret:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transformation_matrix = np.eye(4)
            transformation_matrix[0:3, 0:3] = rotation_matrix
            transformation_matrix[0:3, 3] = tvec.squeeze()
            np.set_printoptions(precision=2, suppress=True)
            print("Transformation Matrix:")
            print(transformation_matrix)

            # Save pose data to an .npz file for later use
            pose_path = os.path.join(output_folder, 'camera_pose.npz')
            np.savez(
                pose_path, 
                rvec=rvec, 
                tvec=tvec, 
                transformation_matrix=transformation_matrix
            )
            print(f"Pose data saved to {pose_path}")

            # Draw axes and save the resulting image
            from cv2 import drawFrameAxes
            axis_length = 3 * square_size_mm 
            drawFrameAxes(img, mtx, dist, rvec, tvec, axis_length)

            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"{basename}_pose.png")
            cv2.imwrite(output_path, img)

def main():
    image = "0001" 
    script_folder = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_folder, 'input', image + ".png")
    intrinsic_path = os.path.join(script_folder, 'data', 'camera_intrinsics.npz')
    output_folder = os.path.join(script_folder, "output")

    checkerboard_rows = 6
    checkerboard_cols = 9
    square_size_mm = 25

    estimate_pose(
        image_path=image_path,
        intrinsic_file=intrinsic_path,
        checkerboard_size=(checkerboard_rows, checkerboard_cols),
        square_size_mm=square_size_mm,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()