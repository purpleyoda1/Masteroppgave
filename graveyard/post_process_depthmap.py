import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import random
import numpy as np
import cv2
import math
import shutil

####################################################################################
#                            PATHS AND PARAMETERS
####################################################################################
#SCRIPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, os.pardir)
RAW_DIR = os.path.join(PROJECT_DIR, "synthetic_data", "depth_maps_raw")

# Output folder after applying pre-processing
run_name = "MiDaS"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "synthetic_data", "training", run_name)

# Set ratios for splitting data
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

####################################################################################
#                            UTILITIES
####################################################################################
def read_exr(path):
     # Load the image
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if exr is None:
        return None
        
    # For Blender normal maps, we expect BGR format
    # Convert BGR to RGB if necessary
    if len(exr.shape) == 3 and exr.shape[2] >= 3:
        exr = exr[:, :, :3]  # Take only first 3 channels if there are more
        exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    else:
        print(f"Warning: Unexpected normal map format. Shape: {exr.shape}")
        return None

    # Ensure the values are in float32 format
    return exr.astype(np.float32)

def create_directory_structure(processed_dir, recreate=True):
    """Create the necessary directory structure for processed data."""
    if recreate and os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(processed_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, "labels", split), exist_ok=True)

####################################################################################
#                            NOISE FUNCTIONS
####################################################################################
def add_gaussian(depth_map, mean=0.0, std_factor=0.01):
    """
    Add Gaussian noise scaled by current depth values.
    """
    depth_map = depth_map.astype(np.float32)
    max_val = np.max(depth_map)
    noise = np.random.normal(mean, std_factor * max_val, depth_map.shape)
    depth_map += noise
    depth_map = np.clip(depth_map, 0, max_val)
    return depth_map

def add_speckle(depth_map, mean=1.0, std=0.05):
    """
    Multiply depth map by random factor ~ Normal(mean, std).
    """
    depth_map = depth_map.astype(np.float32)
    max_val = np.max(depth_map)
    noise = np.random.normal(mean, std, depth_map.shape)
    depth_map *= noise
    depth_map = np.clip(depth_map, 0, max_val)
    return depth_map

def add_missing_data(depth_map, dropout_prob= 0.01):
    """
    Randomly single out pixels
    """
    dropout = np.random.rand(*depth_map.shape) < dropout_prob
    depth_map[dropout] = 0

    return depth_map.astype(np.uint16)

def add_edge_noise(depth_map, kernel_size=3):
    """
    Use Canny on the normalized depth to zero out edges. 
    Then morphologically close to make them connected.
    """
    # Normalize to 0..255 for Canny
    d8 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(d8, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate or morph-close
    structure = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure)
    
    # zero-out
    depth_map[closed > 0] = 0
    return depth_map

def add_invalid_band(depth_map, noise_settings):
    """
    Adds the so-called invalid band, a section of zeroed out data in the left of the frame because of the stereo imaging
    """
    # Get depth map shape
    height, width = depth_map.shape
    
    # Get depth map mean to be used as scene depth
    mean = depth_map.mean()
    
    # Calculate band with using formula from datasheet
    avg_band_width = ((50 * width) / (2 * mean * 0.01 * np.tan(1.5555273294448853/2))) * 0.8
    print(f"Band width set to: {avg_band_width}")
    
    # Randomize edge
    fluctuation = noise_settings.get('invalid_fluctuation', 2)
    band_width = np.random.normal(loc= avg_band_width, scale= fluctuation, size= height)
    band_width = np.clip(np.round(band_width), 0, width).astype(int)
    for row in range(height):
        bw = band_width[row]
        #print(f"Band width: {bw}")
        if bw > 0:
            depth_map[row, :bw] = 0
            
    return depth_map

def add_normal_noise(depth_map, normal_map, fx= 215.262, fy= 215.262, cx= 212.262, cy= 120.86, angle_min=70, angle_max=90, exponent=1.0, kernel_size=3):
    """
    Zeros out pixels whose angle between (camera->pixel) vector and `normal_map` 
    is large enough.
    """
    if normal_map is None or depth_map is None:
        return depth_map

    h, w = depth_map.shape

    # Rescale normal map from [0,1] -> [-1,1]
    normals = normal_map * 2.0 - 1.0
    # Flip Z if needed 
    normals[..., 2] *= -1.0

    # Prepare output mask
    dropout_mask = np.zeros((h, w), dtype=np.uint8)

    # For each pixel, compute the 3D camera->pixel vector
    yy, xx = np.indices((h, w)) 
    Z = depth_map.astype(np.float32)
    valid = (Z > 0)
    X_c = (xx - cx) / fx * Z
    Y_c = (yy - cy) / fy * Z
    V = np.stack((X_c, Y_c, Z), axis=-1)

    # unit normal of pixel -> V / |V|
    mag = np.linalg.norm(V, axis=-1, keepdims=True)
    mag[mag == 0] = 1e-6  # avoid div by zero
    Vn = V / mag

    # Dot product with normal_map
    dot = (Vn * normals).sum(axis=-1)

    # Clip dot to [-1..1], compute angle in degrees
    dot = np.clip(dot, -1, 1)
    angles_deg = np.degrees(np.arccos(dot))

    # Now define a linear drop probability based on angle range
    angle_range = angle_max - angle_min
    alpha = (angles_deg - angle_min) / float(angle_range)
    alpha = np.clip(alpha, 0.0, 1.0)
    drop_prob = alpha ** exponent

    # Random test
    rand_map = np.random.rand(h, w)
    mask = (valid) & (rand_map < drop_prob)
    dropout_mask[mask] = 1

    # Morphological close to make holes more connected
    structure = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(dropout_mask, cv2.MORPH_CLOSE, structure)

    # Zero out depth
    depth_map[closed > 0] = 0

    return depth_map



####################################################################################
#                             FRAME PROCESSING
####################################################################################
def convert_to_midas(depth_map):
    """
    Convert to MiDaS inverse depth format
    """
    # Create a copy
    depth = depth_map.copy().astype(np.float32)
    
    # Handle invalid values
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return np.zeros_like(depth)
    
    # Find min/max and normalize to 1-0 range
    min_depth = np.min(depth[valid_mask])
    max_depth = np.max(depth[valid_mask])

    normalized = np.zeros_like(depth)
    if max_depth > min_depth:
        normalized[valid_mask] = (depth[valid_mask] - min_depth) / (max_depth - min_depth)
    
    inverted = np.zeros_like(normalized)
    inverted[valid_mask] = 1.0 - normalized[valid_mask]

    # Convert to 16bit PNG format
    midas_16bit = (inverted * 65535).astype(np.uint16)

    return midas_16bit




####################################################################################
#                             FRAME PROCESSING
####################################################################################
def collect_frame_data(raw_dir):
    """
    Collect all frame data from the raw directory
    """
    depth_dir = os.path.join(raw_dir, "depth")
    normal_dir = os.path.join(raw_dir, "normal")
    label_dir = os.path.join(raw_dir, "labels")

    frames = []
    for fname in os.listdir(depth_dir):
        if fname.endswith(".png"):
            frame_id = os.path.splitext(fname)[0]
            frames.append({
                'frame_id': frame_id,
                'depth_path': os.path.join(depth_dir, fname),
                'normal_path': os.path.join(normal_dir, frame_id + ".exr"),
                'label_path': os.path.join(label_dir, frame_id + ".txt")
            })
    return frames

def split_frames(frames, train_ratio, val_ratio):
    """
    Split frames into training, validation and test sets
    """
    random.shuffle(frames)
    N = len(frames)
    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)

    return {
        'train': frames[:n_train],
        'val': frames[n_train:n_train + n_val],
        'test': frames[n_train + n_val:]
    }

def process_depth_map(depth_map, normal_map, noise_settings):
    """
    Apply noise to depth image
    """
    if depth_map is None:
        return None
    
    depth_map = depth_map.astype(np.float32)

    if noise_settings["add_normal"]:
        depth_map = add_normal_noise(
            depth_map=depth_map,
            normal_map=normal_map,
            fx=noise_settings["fx"],
            fy=noise_settings["fy"],
            cx=noise_settings["cx"],
            cy=noise_settings["cy"],
            angle_min=noise_settings["normal_angle_min"],
            angle_max=noise_settings["normal_angle_max"],
            exponent=noise_settings["normal_exponent"],
            kernel_size=3
        )

    if noise_settings["add_edge"]:
        depth_map = add_edge_noise(depth_map, noise_settings.get('edge_kernel_size', 3))

    if noise_settings["add_invalid_band"]:
        depth_map = add_invalid_band(depth_map, noise_settings)

    if noise_settings["add_gaussian"]:
        depth_map = add_gaussian(depth_map, noise_settings.get('gaussian_mean', 0), noise_settings.get('gaussian_std', 0.01))

    if noise_settings["add_speckle"]:
        depth_map = add_speckle(depth_map, noise_settings.get('speckle_mean', 1.0), noise_settings.get('speckle_std', 0.05))

    if noise_settings["add_missing_data"]:
        depth_map = add_missing_data(depth_map, noise_settings.get('missing_dropout_prob', 0.01))

    depth_map = np.clip(depth_map, 0, np.max(depth_map))
    return depth_map.astype(np.uint16)

def save_frame(frame_data, processed_dir, split_name):
    """
    Save processed frame data and labels
    """
    img_out_dir = os.path.join(processed_dir, "images", split_name)
    label_out_dir = os.path.join(processed_dir, "labels", split_name)
    
    # Save image
    out_fname = f"{frame_data['frame_id']}.png"
    cv2.imwrite(os.path.join(img_out_dir, out_fname), frame_data['processed_depth'])
    
    # Save label if it exists
    if os.path.exists(frame_data['label_path']):
        with open(frame_data['label_path'], "r") as f_in:
            labels = f_in.read()
        with open(os.path.join(label_out_dir, f"{frame_data['frame_id']}.txt"), "w") as f_out:
            f_out.write(labels)


def process_split(frames, split_name, processed_dir, noise_settings):
    """
    Process all frames in a given split
    """
    for frame in frames:
        # Read depth
        depth_map = cv2.imread(frame['depth_path'], cv2.IMREAD_UNCHANGED)
        normal_map = read_exr(frame['normal_path']) if noise_settings["add_normal"] else None
        
        # Process depth
        processed_depth = process_depth_map(depth_map, normal_map, noise_settings)
        if processed_depth is not None:
            frame['processed_depth'] = processed_depth
            save_frame(frame, processed_dir, split_name)


####################################################################################
#                                    MAIN
####################################################################################

def main():
    create_directory_structure(OUTPUT_DIR)

    noise_settings = {
        'add_gaussian': False,              #Gaussian
        'gaussian_mean': 0,
        'gaussian_std': 0.02,
        'add_speckle': False,               #Speckle
        'speckle_mean': 1.0,
        'speckle_std': 0.05,
        'add_missing_data': False,          #Random missing data
        'missing_dropout_prob': 0.01,
        'add_edge': True,                   #Missing data around edges
        'edge_dropout_prob': 1.0,
        'edge_kernel_size': 3,
        'edge_iterations': 1,
        'add_invalid_band': True,           #Invalid band
        'invalid_fluctuation': 1,
        'add_normal': True,                 #Missing data perpendicular surfaces
        "fx": 215.262,
        "fy": 215.262,
        "cx": 212.262,
        "cy": 120.86,
        'normal_angle_min': 82,
        'normal_angle_max': 90,
        'normal_exponent': 1.0,
        'add_padding': True,
    }

    frames = collect_frame_data(RAW_DIR)
    split_frames_dict = split_frames(frames, TRAIN_RATIO, VAL_RATIO)

    for split_name, split_frame in split_frames_dict.items():
        process_split(split_frame, split_name, OUTPUT_DIR, noise_settings)
    
    print(f"Post processing applied! Frames saved to: {OUTPUT_DIR}")

if __name__=="__main__":
    main()