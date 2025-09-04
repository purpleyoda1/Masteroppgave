import bpy
import numpy as np
import os
import sys
import argparse
import cv2
from mathutils import Matrix


def get_args():
    argv = sys.argv
    try:
        index = argv.indec("--") + 1
        return argv[index:]
    except ValueError:
        return []
    
def opencv_to_blender_matrix(rvec, tvec):
    """Converts OpenCV output to 4x4 transformation matrix"""
    R, _ = cv2.Rodrigues(rvec)
    R_blender_to_opencv = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1)
    ))

    T_opencv = Matrix.Identity(4)
    T_opencv[:3, :3] = Matrix(R.tolist())