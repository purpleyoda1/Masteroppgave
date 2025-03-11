import os
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt

def pad_to_square(image):
    height, width = image.shape[:2]
    side_size = max(height, width)
    
    if len(image.shape) == 3:
        new_image = np.zeros((side_size, side_size, image.shape[2]), dtype=image.dtype)
    else:
        new_image = np.zeros((side_size, side_size), dtype=image.dtype)
    
    y_offset = (side_size - height) // 2
    x_offset = (side_size - width) // 2
    
    new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return new_image, x_offset, y_offset

def print_png_info(path):
    if not os.path.isfile(path):
        print(f"File not found at: {path}")
        return
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Unable to read image at: {path}")
        return

    filesize = os.path.getsize(path)
    shape = img.shape
    img_type = type(img)
    max_val = np.max(img)
    min_val = np.min(img)
    avg_val = np.average(img)

    color_mode = "Unknown"
    if len(img.shape) == 2:
        color_mode = "Grayscale"
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            color_mode = "Grayscale"
        elif channels == 3:
            color_mode = "BGR"
        elif channels == 4:
            color_mode = "BGRA"
    
    dtype = img.dtype

    print(f"--- PNG at {path} ---")
    print(f"Filesize: {filesize} bytes")
    print(f"Shape: {shape}")
    print(f"Data Type: {dtype}")
    print(f"Image Type: {img_type}")
    print(f"Max Value: {max_val}")
    print(f"Min Value: {min_val}")
    print(f"Average Value: {avg_val}")
    print(f"Color Mode: {color_mode}")

def calculate_transformation(depth_map, depth_scale, bbox, intrinsics):
    """
    Calculate the 3D transformation based on depth map and bounding box.
    """
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    x1, y1, x2, y2 = bbox
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    depth_value = depth_map[v, u] * depth_scale

    if depth_value <= 0:
        print(f"Invalid depth value at ({u}, {v})")
        return None, None

    x = (u - cx) * depth_value / fx
    y = (v - cy) * depth_value / fy
    z = depth_value

    transformation = np.eye(4)
    transformation[0:3, 3] = np.array([x, y, z])

    return transformation, (x, y, z)

def visualize_depth_map(depth_map, title="Depth Map"):
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar(label='Depth value (mm)')
    plt.title(title)
    plt.show()
