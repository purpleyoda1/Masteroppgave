import os
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import random
import shutil
from pathlib import Path

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


def split_data_for_yolo(
    source_folder,
    output_folder,
    class_names,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Splits depth maps and their corresponding labels into train, validation, and test sets
    for YOLO model training.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create source paths
    depth_path = Path(source_folder) / "depth"
    labels_path = Path(source_folder) / "labels"
    
    # Verify source paths exist
    if not depth_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Source folders not found: {depth_path} or {labels_path}")
    
    # Get all depth map files
    depth_files = [f for f in os.listdir(depth_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle the files
    random.shuffle(depth_files)
    
    # Calculate split indices
    n_files = len(depth_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Split files
    train_files = depth_files[:n_train]
    val_files = depth_files[n_train:n_train + n_val]
    test_files = depth_files[n_train + n_val:]
    
    # Create output directories
    output_path = Path(output_folder)
    
    # Structure for YOLO
    dirs = {
        'train': {'images': output_path / "images" / "train", 'labels': output_path / "labels" / "train"},
        'val': {'images': output_path / "images" / "val", 'labels': output_path / "labels" / "val"},
        'test': {'images': output_path / "images" / "test", 'labels': output_path / "labels" / "test"},
    }
    
    # Create all directories
    for split in dirs.values():
        for dir_path in split.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Function to copy files for a specific split
    def copy_files(file_list, split_name):
        print(f"Copying {len(file_list)} files to {split_name} set...")
        for img_file in file_list:
            # Get the base name without extension
            base_name = os.path.splitext(img_file)[0]
            
            # Source paths
            img_src = depth_path / img_file
            # Try different label extensions (.txt, .xml, etc.)
            label_extensions = ['.txt', '.xml']
            label_src = None
            
            for ext in label_extensions:
                potential_label = labels_path / f"{base_name}{ext}"
                if potential_label.exists():
                    label_src = potential_label
                    break
            
            if label_src is None:
                print(f"Warning: No label file found for {img_file}, skipping.")
                continue
            
            # Destination paths
            img_dst = dirs[split_name]['images'] / img_file
            label_dst = dirs[split_name]['labels'] / label_src.name
            
            # Copy files
            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)
    
    # Copy files for each split
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # Create a dataset.yaml file for YOLO with dynamic paths and class info
    train_images_path = os.path.abspath(os.path.join(output_folder, 'images', 'train'))
    val_images_path = os.path.abspath(os.path.join(output_folder, 'images', 'val'))
    test_images_path = os.path.abspath(os.path.join(output_folder, 'images', 'test'))
    
    yaml_content = f"""# YOLO dataset configuration
path: {os.path.dirname(train_images_path)} 
train: {os.path.join('images', 'train')} 
val: {os.path.join('images', 'val')} 
test: {os.path.join('images', 'test')}  

# Classes
nc: {len(class_names)}
names: {class_names} 
"""
    
    # Write the YAML file to the output directory
    yaml_path = os.path.join(output_folder, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"Created YAML configuration file at: {yaml_path}")
    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    source_path = os.path.join(parent_dir, "synthetic_data/depth_maps_raw")
    output_path = os.path.join(parent_dir, "synthetic_data/training/raw")
    split_data_for_yolo(source_path, output_path, ["cylinder"])