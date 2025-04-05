# src/system/modules/VisualizationModule.py

from ..SystemModule import SystemModule
from ..SystemData import SystemData
from ..structs.Detection import Detection

import numpy as np
import pyrealsense2 as rs
from typing import  Optional, Any, Dict, Set, Tuple, List
import logging
import cv2
import threading
import math

class VisualizationModule(SystemModule):
    """
    Standardized visualization for proper comparing of depth maps
    """
    def __init__(self, 
                 config: Any, 
                 module_name: str = "Visualization_Module", 
                 view_target_height: int = 240,
                 canvas_height: int = 1080, 
                 canvas_width: int = 1920):
        """Initialize Visualization module"""
        super().__init__(config, module_name)
        self.logger = logging.getLogger(self.name)
        self._view_target_height = view_target_height
        self._canvas_height = canvas_height
        self._canvas_width = canvas_width
        
        self.is_initialized = False
        self.detection_color = (255, 192, 203)
        self._vis_lock = threading.Lock()
        
        self._active_views: Set[str] = {SystemData.COLOR}
        self._ALL_POSSBLE_VIEWS = {
            SystemData.COLOR,
            SystemData.DEPTH,
            SystemData.VIS_DEPTH_COLORMAP,
            SystemData.VIS_MIDAS_COLORMAP,
            SystemData.VIS_PRO_COLORMAP,
            SystemData.VIS_DEPTH_COLORMAP_DETECTIONS,
            SystemData.VIS_MIDAS_COLORMAP_DETECTIONS,
            SystemData.VIS_PRO_COLORMAP_DETECTIONS
        }

        # Views that simply should be passed on
        self._pass_on_views: Set[str] = {SystemData.COLOR,
                                        SystemData.DEPTH}
        # Views that need colormap
        self._colormap = cv2.COLORMAP_JET
        self._colormap_inputs: Dict[str, str] = {
            SystemData.VIS_DEPTH_COLORMAP: SystemData.NORM_DEPTH,
            SystemData.VIS_MIDAS_COLORMAP: SystemData.NORM_MIDAS,
            SystemData.VIS_PRO_COLORMAP: SystemData.NORM_PRO
        }
        # Views that need detection overlay
        self._detection_views: Dict[str, Tuple[str, str]] = {
            SystemData.VIS_DEPTH_COLORMAP_DETECTIONS: (SystemData.NORM_DEPTH, SystemData.DEPTH_DETECTIONS),
            SystemData.VIS_MIDAS_COLORMAP_DETECTIONS: (SystemData.NORM_MIDAS, SystemData.MIDAS_DETECTIONS),
            SystemData.VIS_PRO_COLORMAP_DETECTIONS: (SystemData.NORM_PRO, SystemData.PRO_DETECTIONS)
        }

    def set_view(self, view_key: str, active: bool) -> None:
        """Sets status of a single view"""
        if view_key not in self._ALL_POSSBLE_VIEWS:
            self.logger.warning(f"Attemptet to set invalid key: {view_key}")
            return
        
        with self._vis_lock:
            if active:
                self._active_views.add(view_key)
            else:
                self._active_views.discard(view_key)
        status = "activated" if active else "deactivated"
        self.logger.debug(f"{view_key} {status}")

    def toggle_view(self, view_key: str) -> None:
        """Toggle status of single view"""
        if view_key not in self._ALL_POSSBLE_VIEWS:
            self.logger.warning(f"Attempted to toggle invalid key: {view_key}")
            return
        
        with self._vis_lock:
            if view_key in self._active_views:
                self._active_views.discard(view_key)
                active = False
            else:
                self._active_views.add(view_key)
                active = True
        status = "activated" if active else "deactivated"
        self.logger.debug(f"{view_key} {status}")

    def get_active_views(self) -> List[str]:
        """Returns all active keys"""
        with self._vis_lock:
            return sorted(list(self._active_views))

    def initialize(self, config: Any) -> bool:
        """Set up and initialize module"""
        if self.is_initialized:
            self.logger.info(f"{self.name} already initialized")
            return True
        
        # No logic actually needed, but has to be implemented for other System functionality

        self.logger.info(f"{self.name} initialized succesfully")
        self.is_initialized = True
        return True
    
    def get_required_inputs(self) -> Set[str]:
        return {SystemData.COLOR}
    
    def get_dependency_inputs(self) -> Set[str]:
        dependency_inputs = set(self._pass_on_views) \
                                .union(set(self._colormap_inputs.values())) \
                                .union({base for base, det in self._detection_views.values()}) \
                                .union({det for base, det in self._detection_views.values()})           
        return dependency_inputs
    
    def get_outputs(self) -> Set[str]:
        return {SystemData.VIS_DEPTH_COLORMAP,
                SystemData.VIS_MIDAS_COLORMAP,
                SystemData.VIS_PRO_COLORMAP,
                SystemData.VIS_DEPTH_COLORMAP_DETECTIONS,
                SystemData.VIS_MIDAS_COLORMAP_DETECTIONS,
                SystemData.VIS_PRO_COLORMAP_DETECTIONS,
                SystemData.VIS_MONTAGE}
    
    def _apply_colormap(self, depth_map: np.ndarray) -> Optional[np.ndarray]:
        """Normalizes depth map then applies colormap"""
        if depth_map is None:
            self.logger.error(f"_apply_colormap received None-type")
            return None
        if not isinstance(depth_map, np.ndarray):
            self.logger.error(f"_apply_colormap received non-array input: {type(depth_map)}")
            return None
        
        try:
            # Apply colormap
            depth_map = depth_map.astype(np.uint8)
            colormapped = cv2.applyColorMap(depth_map, self._colormap)
            
            return colormapped
        
        except Exception as e:
            self.logger.error(f"Error coloring depthmap: {e}")
            return None
        
    def _draw_detections(self, image: np.ndarray, detections: List[Detection]):
        """Draw detection bounding boxes on images"""
        if image is None:
            return None
        output_image = image.copy()
        if detections == None or len(detections) == 0:
            return output_image
        
        if len(output_image.shape) == 2: 
            output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        elif output_image.shape[2] == 1:
             output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

        for det in detections:
            try:
                if det.dimension == "2D" and det.bbox2D:
                    # Draw box
                    x1, y1, x2, y2 = det.bbox2D
                    label = det.label if det.label else "unknown"
                    conf = det.conf if det.conf is not None else 0.0
                    source = det.source if det.source else "unknown"

                    bbox_color = self.detection_color
                    text_color = (0, 0, 0)

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), bbox_color, 2)

                    # Construct and draw label
                    label_text = f"{label}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + baseline

                    # text background
                    bg_y1 = max(0, text_y - text_height - baseline)
                    bg_y2 = min(output_image.shape[0], text_y + baseline)
                    bg_x1 = max(0, x1)
                    bg_x2 = min(output_image.shape[1], x1 + text_width)
                    cv2.rectangle(output_image, 
                                (bg_x1, bg_y1),
                                (bg_x2, bg_y2), 
                                bbox_color, 
                                cv2.FILLED
                            )
                    cv2.putText(output_image, 
                                label_text, 
                                ((min(output_image.shape[0] - baseline), text_y - baseline // 2), bg_x1), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, 
                                text_color, 
                                1,
                                cv2.LINE_AA
                            )
            except Exception as e:
                self.logger.error(f"Error drawing detections, skipping: {e}")
                continue

        return output_image
    
    def _resize_for_montage(self, image: Optional[np.ndarray], target_height: int) -> Optional[np.ndarray]:
        """Internall helper for mergin frames"""
        if image is None:
            self.logger.debug(f"Input image to _resize is None")
            return None
        if not isinstance(image, np.ndarray):
            self.logger.warning(f"Invalid input to _resize. Type: {image.dtype} Shape: {image.shape}")
            return None
        
        processed_image = None
        original_dtype = image.dtype
        original_shape = image.shape

        try:
            temp_image = None
            if len(original_shape) == 2:
                if original_dtype == np.uint8:
                    img_8u = image
                elif original_dtype in [np.uint16, np.float32, np.float64]:
                    img_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    try:
                        img_8u = image.astype(np.uint8)
                        self.logger.warning(f"Directly casted 2D input from {original_dtype} to uint8.")
                    except Exception:
                        self.logger.error(f"Cannot handle 2D input dtype {original_dtype}.")
                        return None
                temp_image = np.stack((img_8u,) * 3, axis=-1)

            elif len(original_shape) == 3:
                if original_shape[2] == 1:
                    channel = image[:, :, 0]
                    if channel.dtype == np.uint8:
                        img_8u = channel
                    elif channel.dtype in [np.uint16, np.float32, np.float64]:
                        img_8u = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    else: 
                        try:
                            img_8u = channel.astype(np.uint8)
                            self.logger.warning(f"Directly casted 3D/1ch input from {channel.dtype} to uint8.")
                        except Exception:
                            self.logger.error(f"Cannot handle 3D/1ch input dtype {channel.dtype}.")
                            return None
                    temp_image = np.stack((img_8u,) * 3, axis=-1)

                elif original_shape[2] == 3:
                    temp_image = image
                else:
                    self.logger.warning(f"Unsupported channel count {original_shape[2]}")
                    return None
            else:
                self.logger.warning(f"Unsupported shape {original_shape}.")
                return None
        except Exception as e:
            self.logger.warning(f"Error converting input to 3-channel: {e}")
       

        h, w = temp_image.shape[:2]
        if h == target_height:
            return temp_image
        if h == 0:
            return None
        
        scale = target_height / h
        target_width = int(w*scale)
        if target_width < 0:
            return None
        
        try:
            return cv2.resize(temp_image, (target_width, target_height), interpolation = cv2.INTER_LINEAR).astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return None
        
    def _create_montage(self, views: List[np.ndarray], montage_width: int, montage_height: int) -> Optional[np.ndarray]:
        """Grid montage of active views"""
        if not views:
            self.logger.warning("Couldnæt create montage, views not available")
            return None
        
        num_views = len(views)
        if num_views == 0: return None

        # Get cell dimensions and grid size
        cell_h, cell_w = views[0].shape[:2]
        grid_cols = int(math.ceil(np.sqrt(num_views)))
        grid_rows = int(math.ceil(num_views/grid_cols))

        # Black cells if needed
        black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        padding_needed = (grid_cols * grid_rows) - num_views
        padded_views = views + [black_cell] * padding_needed

        # Assemble rows
        montage_rows = []
        for r in range(grid_rows):
            start_index = r * grid_cols
            end_index = start_index + grid_cols
            row_views = padded_views[start_index:end_index]
            try:
                montage_rows.append(cv2.hconcat(row_views))
            except Exception as e:
                self.logger.error(f"Error hconcatenating row {r}: {e}")
        
        # Assemble grid
        raw_montage = None
        try:
            if montage_rows:
                expected_width = grid_cols * cell_w
                valid_rows = [row for row in montage_rows if row is not None and len(row.shape) == 3 and row.shape[1] == expected_width and row.shape[0] == cell_h]
                if len(valid_rows) == grid_rows:
                    raw_montage = cv2.vconcat(valid_rows)
                else:
                    self.logger.error(f"Inconsitent row shapes for montage")
                    return None
        except Exception as e:
            self.logger.error(f"Error assembling grid: {e}")

        # Scale and pad grid
        try: 
            raw_h, raw_w = raw_montage.shape[:2]
            target_h, target_w = montage_height, montage_width

            scale = min(target_h/raw_h, target_w/raw_w)
            new_h = int(raw_h * scale)
            new_w = int(raw_w * scale)

            # Ensure valid dimensions
            if new_h <= 0 or new_w <= 0:
                self.logger.error(f"Invalid montage dimensions: {new_w} x {new_h}")
                return raw_montage
            
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_montage = cv2.resize(raw_montage, (new_w, new_h), interpolation = interp)

            # Fill out background with black canvas
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            pad_y = target_h - new_h
            pad_x = target_w - new_w
            y_offset = pad_y // 2
            x_offset = pad_x // 2

            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_montage
            return canvas

        except Exception as e:
            self.logger.error(f"Error while resizing montage: {e}")
            return raw_montage
        

    def _process_internal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Core implementation of visualization logic"""
        if not self.is_initialized:
            self.logger.debug(f"Visualization module not initialized, skipping processing")
            return None
        
        output_data: Dict[str, Any] = {}

        
        for key in self._active_views:
            # Pass-Trough
            if key in self._pass_on_views:
                output_data[key] = data.get(key)

            # Colormaps
            elif key in self._colormap_inputs:
                input_key = self._colormap_inputs.get(key)
                if input_key in data:
                    output_data[key] = self._apply_colormap(data.get(input_key))
                else:
                    self.logger.debug(f"Missing input: {input_key}")
            
            # Detection overlays
            elif key in self._detection_views:
                depth_map_key, detection_key = self._detection_views.get(key)
                if depth_map_key:
                    depth_map = self._apply_colormap(data.get(depth_map_key))
                    if depth_map is not None:
                        if detection_key:
                            detections = data.get(detection_key)
                            if detections:
                                output_data[key] = self._draw_detections(depth_map, detections)
                            else:
                                output_data[key] = depth_map
                        else:
                            self.logger.debug(f"Missing input: {detection_key}")
                else:
                    self.logger.debug(f"Missing input: {depth_map_key}")

        # Then montage logic
        view_order = self.get_active_views()
        views_for_montage = []

        for key in view_order:
            view = output_data[key]
            if view is not None:
                resized_view = self._resize_for_montage(view, self._view_target_height)
                views_for_montage.append(resized_view)

        vis_montage = self._create_montage(views_for_montage, self._canvas_width, self._canvas_height)
        output_data[SystemData.VIS_MONTAGE] = vis_montage

        return output_data
    
    def stop(self) -> None:
        self.logger.info(f"Stopping {self.name}...")
        self.is_initialized = False
