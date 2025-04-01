import open3d as o3d
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PointCloudFK:
    """
    Builds and maintains a continous point cloud using incoming depth and RGB frames, 
    using the robots forward kinematics to retrieve camera pose
    """

    def __init__(self, 
                 intrinsics: o3d.camera.PinholeCameraIntrinsics, 
                 voxel_size: float= 0.05, 
                 depth_cutoff: float= 3.0):
        """
        Initialized a pointcloud
        """
        self.intrincics = intrinsics
        self.voxel_size = voxel_size
        self.depth_cutoff = depth_cutoff
        
        # Global and accumulatic pointcloud
        self.global_pcd = o3d.geometry.PointCloud()

    def rgbd_to_pointcloud(self, color, depth):
        """
        Converts a rgb and depth frame to an open3D pointcloud
        """
        # Convert frames to open3D format
        o3d_depth = o3d.geometry.Image(depth)
        o3d_color = o3d.geometry.Image(color)

        # Construct RGBD
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_trunc= self.depth_cutoff, convert_rgb_to_intensity= False)

        # Create pointcloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd(rgbd, self.intrincics)

        return pcd
    
    def add_frame(self, color, depth, fk_transform):
        """
        Integrates new set of color and depth frames to pointcloud with the help of FK
        """
        # Construct and downsample new pointcloud
        pcd = self.rgbd_to_pointcloud(color, depth)
        pcd_ds = pcd.voxel_down_sample(self.voxel_size)

        # Applyt FK to transform pointcloud to robot base coordinate system
        pcd_ds.transform(fk_transform)

        # Merge current with new pointclkoud, and downsize
        self.global_pcd += pcd_ds
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
        logging.info("New frames succesfully integrated")

        return self.global_pcd

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.global_pc
    
    def visualize(self):
        try:
            o3d.visualization.draw_geometries([self.global_pc])
        except Exception as e:
            logging.error(f"Error visualizing point cloud: {e}")      

         
