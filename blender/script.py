import sys
sys.path.append(r"c:\users\sondr\appdata\roaming\python\python311\site-packages")

import bpy
import bpy_extras
import math
from mathutils import Vector
import os
import shutil
import numpy as np
import json
import random
import cv2
from bpy_extras.object_utils import world_to_camera_view

###############################################################################
#                            USER PARAMETERS
###############################################################################

# Run settings
CLEAN_OUTPUT_DIR = False
SAVE_AABB_LABELS = True
SAVE_OBB_LABELS = True

# Class ID mapping
CLASS_ID_MAPPING = {
    "Capacitor": 0,
    "Bracket": 1,
    "Screw": 2
}

# D435i Camera Parametrers
WIDTH = 424  
HEIGHT = 240  
FX = 215.262  
FY = 215.262
CX = 212.262 
CY = 120.86
FOCAL_LENGTH_MM = 1.93
baseline = 50 #mm

# Object placement
MAX_PLACEMENT_RETRIES = 5
GROUND_LEVEL = 0.0
IMPORT_COLLECTION_NAME = "ImportedObjects"

# Camera movement grid
BASE_PLACEMENT_RANGE = 0.15
NUM_HEIGHT_STEPS = 7
MIN_HEIGHT = 0.35
MAX_HEIGHT = 1.0
NUM_STEPS_XY = 20

###############################################################################
#                                PATHS
###############################################################################

PARENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)

# CAD Models
MODEL_FILES_DIR = os.path.joint(PARENT_DIR, 'synthetic_data', 'CAD')

# File outputs
OUTPUT_DIR = os.path.join(PARENT_DIR, "synthetic_data", "depth_maps_raw")
log_file = os.path.join(OUTPUT_DIR, 'progress.log')
# Specific subdirectories
OUTPUT_DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
OUTPUT_NORMAL_DIR = os.path.join(OUTPUT_DIR, "normal")
OUTPUT_LABELS_AABB_DIR = os.path.join(OUTPUT_DIR, "labels_aabb")
OUTPUT_LABELS_OBB_DIR = os.path.join(OUTPUT_DIR, "labels_obb")


###############################################################################
#                             UTILITES
############################################################################### 

def check_output_dirs():
    "Create or reset output dirs"
    if CLEAN_OUTPUT_DIR and os.path.exists(OUTPUT_DIR):
        print(f"Cleaning output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NORMAL_DIR, exist_ok=True)
    if SAVE_AABB_LABELS:
        os.makedirs(OUTPUT_LABELS_AABB_DIR, exist_ok=True)
    if SAVE_OBB_LABELS:
        os.makedirs(OUTPUT_LABELS_OBB_DIR, exist_ok=True)
        
    # Set up collection to keep importet CAD models
    import_collection = bpy.data.collections[IMPORT_COLLECTION_NAME]
    if not import_collection:
        raise ValueError(f"Required collection '{IMPORT_COLLECTION_NAME}' not found")
    else:
        import_collection.hide_viewport = True
        import_collection.hide_render = True


def get_class_id(obj_name):
    "Get objects class id"
    base_name = obj_name.split(".")[0]
    return CLASS_ID_MAPPING.get(base_name, -1)

def world_to_camera_view(scene, camera, coordinate):
    "Casts 3D coordinate into 2D camera cooridnate"
    return bpy_extras.object_utils.world_to_camera_view(scene, camera, coordinate)
    

###############################################################################
#                             BLENDER SETUP
###############################################################################    

def setup_camera():
    "Sets up camera to match Intel RealSense D435i"
    camera = bpy.data.objects['Camera']
    cam_data = camera.data
    
    # Set lens to mm and set sensor size from intrinsics
    cam_data.lens_unit = 'MILLIMETERS'
    cam_data.lens = FOCAL_LENGTH_MM
    SENSOR_WIDTH_MM = (WIDTH * FOCAL_LENGTH_MM) / FX
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.sensor_height = SENSOR_WIDTH_MM * (HEIGHT/WIDTH)
    
    # Shift to principal point
    cam_data.shift_x = (CX - WIDTH / 2) / WIDTH
    cam_data.shift_y = (CY - HEIGHT / 2) / HEIGHT
    
    # Set reslution
    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    
    # Set clipping planes
    cam_data.clip_start = 0.1
    cam_data.clip_end = 10
    
    # Set render engine and enable GPU
    bpy.context.scene.render.engine = 'CYCLES'
    try:
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'

        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'CUDA':
                device.use = True
                print("GPU enabled")
    except Exception as e:
        print(f"GPU unavailable, falling back on CPU")
        bpy.context.scene.cycles.device = 'CPU'

    
    # Enable depth and normal passes
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_normal = True

    # Set it to point downard
    camera.rotation_euler = (0.0, 0.0, 0.0)

    return camera

def set_camera_position(camera, x, y, z):
    "Positions camera at [x, y, z] pointing downards"
    camera.location = (x, y, z)
    
def clean_render_results():
    "Removes everything thats render in an ettempt to keep things tidy"
    for image in bpy.data.images:
        if image.name in ("Render Result", "Viewer Node"):
            bpy.data.images.remove(image)
            
def setup_nodes():
    "Sets up nodes to output depth map and normals"
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    # Base render node
    render_layers = nodes.new('CompositorNodeRLayers')
              
    # Depth output
    output_depth = nodes.new('CompositorNodeOutputFile')
    output_depth.label = 'Depth output'
    output_depth.base_path = OUTPUT_DEPTH_DIR
    output_depth.format.file_format = 'PNG'
    output_depth.format.color_depth = '16'
    output_depth.format.color_mode = 'BW'
    depth_slot = output_depth.file_slots.get("Depth") or output_depth.file_slots.new("Depth_")
    depth_slot.path = "####"
    depth_slot.use_node_format = True
    
    # Normal output
    output_normal = nodes.new('CompositorNodeOutputFile')
    output_normal.label = 'Normal output'
    output_normal.base_path = OUTPUT_NORMAL_DIR
    output_normal.format.file_format = 'OPEN_EXR'
    output_normal.format.color_depth = '32'
    output_normal.format.color_mode = 'RGB'
    normal_slot = output_normal.file_slots.get("Normal") or output_normal.file_slots.new("Normal_")
    normal_slot.path = "####"
    normal_slot.use_node_format = True

    # Link nodes
    links.new(render_layers.outputs['Depth'],  output_depth.inputs[0])
    links.new(render_layers.outputs['Normal'], output_normal.inputs[0])

    print(f"Copmositor nodes set up")


###############################################################################
#                            OBJECT PLACEMENT
############################################################################### 
# References for cleanup
placed_object_copies = []

def clean_placed_objects():
    "Removes all placed objects from previous frames"
    global placed_object_copies
    if not placed_object_copies: return

    objects_to_delete = [obj for obj in placed_object_copies if obj and obj.name in bpy.data.objects]

    # Select objects one by one and delete them
    if objects_to_delete:
        bpy.ops.objects.select_all(action='DESELECT')
        for obj in objects_to_delete:
            if obj in bpy.data.objects:
                obj.select_set = True

        if bpy.context.selected_objects:
            bpy.ops.object.delete(use_global=False)
    
    placed_object_copies = []

def get_object_aabb(obj):
    "Calculates world-coordinates AABB"
    if not obj: return None, None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = bpy.evaluated_get(depsgraph)
    try:
        local_bbox_corner = [Vector(corner) for corner in obj_eval.bound_box]
        if not local_bbox_corner: return None, None
    except Exception as e:
        print(f"Could not get bounding box for {obj.name}: {e}")
        return None, None
    
    world_matrix = obj_eval.matrix_world
    world_bbox_corner = [world_matrix @ corner for corner in local_bbox_corner]
    min_coord = Vector(np.min([c.to_tuple() for c in world_bbox_corner], axis=0))
    max_coord = Vector(np.max([c.to_tuple() for c in world_bbox_corner], axis=0))

    return min_coord, max_coord

def check_collision_aabb(obj1_min, obj1_max, obj2_min, obj2_max):
    "Checks for overlap between two objects"
    if not all([obj1_min, obj1_max, obj2_min, obj2_max]): return False
    margin = 0.01
    overlap_x = (obj1_min.x < obj2_max.x - margin) and (obj1_max.x > obj2_min.x + margin)
    overlap_y = (obj1_min.y < obj2_max.y - margin) and (obj1_max.y > obj2_min.y + margin)
    return overlap_x and overlap_y


def place_components(template_objects, placement_range_xy, ground_z=GROUND_LEVEL):
    """Place random set of subcomponents and place them randomly, checking for collisions"""
    global placed_object_copies
    placed_copies_this_frame = []
    placed_aabb = []

    # Select random subset of components
    max_objects = len(template_objects)
    num_to_place = random.randit(0, max(0, max_objects))
    selected_templates = random.sample(template_objects, num_to_place)

    # Get scene collection (all objects in scene)
    scene_collection = bpy.context.scene.collection

    for template_obj in selected_templates:
        # CHeck that object only contains a mesh
        if template_obj.type != 'MESH':
            print(f"Object {template_obj.name} is not a MESH")
            continue

        # Retrieve object and make a copy
        obj_copy = template_obj.copy()
        if obj_copy.data:
            obj_copy.data = obj_copy.data.copy()
        else:
            print(f"Object {template_obj.name} doesn't contain mesh")
            bpy.data.objects.remove(obj_copy, do_unlink= True)
            continue

        scene_collection.objects.link(obj_copy, do_unlink = True)
        print(f"Created copy {obj_copy.name}")

        # Try to place
        placed_successfully = False
        for attempt in range(MAX_PLACEMENT_RETRIES):
            rand_x = random.uniform(-placement_range_xy, placement_range_xy)
            rand_y = random.uniform(-placement_range_xy, placement_range_xy)
            rand_rot_z = random.uniform(0, 2 * np.pi)

            obj_copy.location = (rand_x, rand_y, ground_z)
            obj_copy.rotation = (0, 0, rand_rot_z)
            bpy.context.view_layer.update()

            collides = False
            current_aabb_min, current_aabb_max = get_object_aabb(obj_copy)

            # Compare aabb to already placed objects
            if current_aabb_min and current_aabb_max:
                for placed_aabb_min, placed_aabb_max in placed_aabb:
                    if check_collision_aabb(current_aabb_min, current_aabb_max, placed_aabb_min, placed_aabb_max):
                        print(f"Collision detected, retrying placement")
                        collides = True
                        break
            else:
                print(f"Not able to retrieve AABB for {obj_copy.name}")
                collides = True
            
            if not collides:
                placed_successfully = True
                placed_copies_this_frame.append(obj_copy)
                placed_aabb.append((current_aabb_min, current_aabb_max))
                break

        # If placed, add it to global placed objects for later cleanup
        if not placed_successfully:
            print(f"Unable to place {obj_copy.name}")
            bpy.data.objects.remove(obj_copy, do_unlink=True)

        else:
            placed_object_copies.append(obj_copy)
    
    return placed_copies_this_frame



###############################################################################
#                             BOUNDING BOX
###############################################################################  

def save_bounding_boxes(placed_copies, camera, filepath):
    "Saves bounding box of chosen object in YOLO format (class_id, xc, yc, w, h)"
    # Get scene
    scene = bpy.context.scene
    lines = []

    # Iterate trough objects
    for obj in placed_copies:
        coords = [world_to_camera_view(scene, camera, object.matrix_world @ v.co) for v in obj.data.vertices]
        min_x = min(c[0] for c in coords)
        max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_y = max(c[1] for c in coords)

        # Skip if out of frame
        if max_x < 0 or min_x > 1 or max_y < 0 or min_y > 1:
            return

        # Clip to [0,1]
        min_x = max(min_x, 0.0)
        max_x = min(max_x, 1.0)
        min_y = max(min_y, 0.0)
        max_y = min(max_y, 1.0)

        x_center = (min_x + max_x) / 2
        y_center = 1.0 - (min_y + max_y) / 2 
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            continue

        class_id = get_class_id(obj.name)
        line = f"{class_id} {x_center} {y_center} {width} {height}"
        lines.append(line)

    with open(filepath, "a") as f:
        f.writelines(lines)

def save_oriented_bounding_boxes(placed_copies, camera, filepath):
    "Save oriented bounding boxes on Ultralytics format [x1 y1 x2 y2 x3 y3 x4 y4]"
    # Retrieve scene information
    scene = bpy.context.scene
    render = scene.render
    width_res = int(render.resolution_x * (render.resolution_percentage)/100)
    height_res = int(render.resolution_y * (render.resolution_percentage)/100)

    if width_res == 0 or height_res == 0:
        print(f"Resolution dimension is 0, unable to save OBB")
        return
    
    # Create container for lines to write
    lines = []
    found_obb = False
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Process one object at a time
    for mesh_obj in placed_copies:
        class_id = get_class_id(mesh_obj.name)
        if class_id == -1:
            print(f"Object {mesh_obj.name} not found in ID mapping, skipping in OBB calc")
            continue
        
        # Evaluate (resolve) objects and get mesh
        obj_eval = mesh_obj.evaluated_get(depsgraph)
        try:
            mesh_eval = obj_eval.to_mesh()
            if not mesh_eval or mesh_eval.vertices:
                print(f"Unable to retrieve evaluated mesh {mesh_obj.name} for OBB")
                if mesh_eval: obj_eval.to_mesh_clear()
                continue
        except Exception as e:
            print(f"Exception evaluation mesh object {mesh_obj.name}: {e}")
        
        # Project mesh vertices to camera space
        world_matrix = obj_eval.matrix_world
        world_vertices = [world_matrix @ v.co for v in mesh_eval.vertices]
        obj_eval.to_mesh_clear()

        coords_2d = [world_to_camera_view(scene, camera, v) for v in world_vertices]
        # Filter valid points and convert to pixel value
        pixel_points = []
        for c in coords_2d:
            if 0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z > 0.0:
                pixel_x = width_res * c.x
                pixel_y = height_res * (1 - c.y)
                pixel_points.appen([pixel_x, pixel_y])

        if len(pixel_points) < 3:
            print(f"Not enough valid pixel points for {mesh_obj.name}")
            continue

        # Calculate min area rectangle with OpenCV
        pixel_points_np = np.array(pixel_points, dtype=np.float32)

        try:
            rectangle = cv2.minAreaRect(pixel_points_np) # ((center_x, center_y), (width, height), angle)
        except Exception as e:
            print(f"Error running cv2.minAreaRect for {mesh_obj.name}: {e}")
            continue

        box_pixel_coords = cv2.boxPoints(rectangle)

        # Normalize
        normalized_corners = []
        for px, py in box_pixel_coords:
            norm_x = np.clip(px / width_res, 0.0, 1.0)
            norm_y = np.clip(py / height_res, 0.0, 1.0)
            normalized_corners.append(f"{norm_x:.6f}")
            normalized_corners.append(f"{norm_y:.6f}")
        

        # Format output
        line = f"{class_id} {' '.join(normalized_corners)}"
        lines.append(line)
        found_obb = True

    # Write to file
    if found_obb:
        try:
            with open(filepath, 'a') as f:
                f.writelines(lines)
        except IOError as e:
            print(f"Error writing OBB to file: {e}")
    elif placed_copies:
        print(f"No valid OBBs found")
        open(filepath, 'w').close()

        
###############################################################################
#                             PROGRESS LOG (in case of crash)
###############################################################################  
def save_progress(log_file, params, counts):
    "Saves current looping params and how far it has come"
    log_data = {
        'params': params,
        'counts': counts
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

def load_progress(log_file):
    "Attempts to load progress from json file"
    if not os.path.exists(log_file):
        return None
    with open(log_file, 'r') as f:
        try:
            log_data = json.load(f)
            return log_data['params'], log_data['counts']
        except json.JSONDecodeError:
            return None 
        
###############################################################################
#                             RENDER LOOP
###############################################################################  
def render_depth_maps(camera, template_objects, camera_positions, heights):
    "Runs the main rendering loop, outputting depth, normals, and labels"
    # Set up scene and nodes
    scene = bpy.context.scene
    output_depth_node, output_normal_node = setup_nodes()

    # Read logger
    start_frame = 0
    last_params = None
    progress_data = load_progress(log_file)
    if progress_data:
        last_params, last_count = progress_data
        print(f"Progress found: {last_params}       {last_count}")

    # Initiate counters
    frame_count = start_frame
    total_generated = 0
    skip_until_parameter_match = (last_params is not None)
    
    # Move camera and objects, and save frames
    for z in heights:
        positions = camera_positions[z]

        # Calculate component placement range
        scale_factor = z / heights[0]
        current_placement_range = BASE_PLACEMENT_RANGE * scale_factor


        for y in positions:
            for x in positions:
                # Skip if last saved params were found
                current_params = [x, y, z]
                if skip_until_parameter_match:
                    if current_params == last_params:
                        skip_until_parameter_match = False
                        continue
                    else:
                        continue

                frame_count += 1 
                frame_number_str = f"{frame_count:04d}"

                # Set up and render scene
                # Clean previous frames objects
                clean_placed_objects()

                # Place new objects
                placed_copies = place_components(template_objects, current_placement_range)

                # Move camera
                set_camera_position(camera, x, y, z)
                bpy.context.view_layer.update() 
                
                # Set current frame number in scene
                bpy.context.scene.frame_current = frame_count               

                # Set output paths
                aabb_label_path = os.path.join(OUTPUT_LABELS_AABB_DIR, frame_number_str + '.txt')
                obb_label_path = os.path.join(OUTPUT_LABELS_OBB_DIR, frame_number_str + '.txt')           
                
                # Set output on nodes
                output_depth_node.file_slots["Depth"].path = frame_number_str
                output_normal_node.file_slots["Normal"].path = frame_number_str

                # Render 
                try:
                    bpy.ops.render.render(write_still = True)
                except Exception as e:
                    print(f"Render failed for frame number {frame_count}: {e}")
                    clean_placed_objects()
                    continue
                
                # Save bounding boxes
                if placed_copies:
                    if SAVE_AABB_LABELS:
                        save_bounding_boxes(placed_copies, camera, aabb_label_path)
                    if SAVE_OBB_LABELS:
                        save_oriented_bounding_boxes(placed_copies, camera, obb_label_path)
                # Create empty labels if no objects
                else:
                    if SAVE_AABB_LABELS: open(aabb_label_path, 'w').close()
                    if SAVE_OBB_LABELS: open(obb_label_path, 'w').close()

                # Clan render
                clean_render_results()

                # Save progress
                save_progress(log_file, current_params, frame_count)
                total_generated += 1
    
    clean_placed_objects()
    print("Finished generating raw depth and normal maps!")
    
    
###############################################################################
#                             MAIN
###############################################################################  
def main():
    # Create directories and setup camera
    check_output_dirs()
    camera = setup_camera()
    
    # Get template collection and extract meshes
    template_collection = bpy.data.collections.get(IMPORT_COLLECTION_NAME)
    template_objcts = []
    valid_class_id = True
    for obj in template_collection.objects:
        if obj.type == 'MESH':
            base_name = obj.name.split('.')[0]
            if base_name not in CLASS_ID_MAPPING:
                valid_class_id = False
                print(f"Invalid object name")
            if not obj.data or not obj.data.vertices:
                print(f"Template object has no vertices")
                continue
            template_objcts.append(obj)
    
    # Define camera movements
    camera_positions = {}
    heights = np.linspace(MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHT_STEPS)
    for height in heights:
        # Scale factor increases with height
        scale_factor = height / heights[0] 
        
        # Scale the range of x/y positions based on height
        scaled_range = BASE_PLACEMENT_RANGE * scale_factor
        positions = np.linspace(-scaled_range, scaled_range, 20)
        camera_positions[height] = positions
        
    render_depth_maps(camera, template_objcts, camera_positions, heights)
    
    print(camera.data.angle_x)
        

if __name__=="__main__":
    try: 
        main()
    except Exception as e:
        import traceback
        traceback.print_exc   