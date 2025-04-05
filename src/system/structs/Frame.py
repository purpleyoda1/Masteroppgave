import numpy as np
import time

class Frame:
    """
    A simple and generalised container of a frame instance, with useful functionality included in class
    """
    def __init__(self, 
                 color_frame: np.ndarray= None,
                 depth_frame: np.ndarray= None,
                 timestamp: float= None,
                 metadata: dict= None
            ):
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.metadata = metadata if metadata is not None else {}
