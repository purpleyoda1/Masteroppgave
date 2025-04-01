#####################################################################################
#                               UNFINISHED
#####################################################################################

import open3d as o3d
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PointCloudSLAM:
    """
    Builds and maintains a continous point cloud using incoming depth and RGB frames, and the SLAM algorithm
    """

    def __init__(self, 
                 intrinsics: o3d.camera.PinholeCameraIntrinsics, 
                 voxel_size: float= 0.05, 
                 depth_cutoff: float= 3.0, 
                 icp_treshold: float= 0.02,
                 min_fitness: float= 0.3,
                 max_inlier_rmse: float= 0.05,
                 robot_transformation = None):
        """
        Initialized a pointcloud
        """
        self.intrincics = intrinsics
        self.voxel_size = voxel_size
        self.depth_cutoff = depth_cutoff
        self.icp_treshold = icp_treshold
        self.min_fitness = min_fitness
        self.max_inlier_rmse = max_inlier_rmse

        # If no robot transformation is given default to identity
        if robot_transformation is None:
            self.robot_transformation = np.eye(4)
        else:
            self.robot_transformation = robot_transformation
        
        # Model parameters needed for internal logic
        self.global_pcd = o3d.geometry.PointCloud()
        self.previous_pcd = None
        self.accumulated_transformation = np.eye(4)
        self.first_frame = True

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

        return pcd, rgbd


    def register_frame(self, new_pcd, current_pcd) -> None:
        """
        Register new pointcloud to the previous using point-to-plane ICP
        """
        # Estimate normals if they dont exist
        if not new_pcd.has_normals():
            new_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, mac_nn=30))
        if not current_pcd.has_normals():
            current_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, mac_nn=30))
        
        # Perform registration using open3D
        transformation_init = np.eye(4)
        reg_result = o3d.pipelines.registration.registration_icp(new_pcd, current_pcd, self.icp_treshold, transformation_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())

        logging.info(f"ICP registration:\nFitness: {reg_result.fitness}\nInlier RMSE: {reg_result.inlier_rmse}")
        return reg_result
    
    def add_frame(self, color, depth):
        """
        Integrates new set of color and depth frames to pointcloud
        """
        # Construct and downsample new pointcloud
        new_pcd, _ = self.rgbd_to_pointcloud(color, depth)
        new_pcd_ds = new_pcd.voxel_down_sample(self.voxel_size)

        # If first frame, transform with the robot transformation
        if self.first_frame:
            new_pcd_ds.transform(self.robot_transformation)
            self.global_pcd = new_pcd_ds
            self.previous_pcd = new_pcd_ds
            self.first_frame = False
            logging.info("Initialized global pointcloud from first frame")
            return self.global_pcd
        
        # If not, register new pointcloud then update class variables
        reg_result = self.register_frame(new_pcd, self.previous_pcd)

        # Check new pcd for quality
        if reg_result.fitness < self.min_fitness:
            logging.warning("Low registration fitness, pointcloud skipped")
            return self.global_pcd
        if reg_result.inlier_rms > self.max_inlier_rmse:
            logging.warning("High inlier RMSE, pointcloud skipped")
            return self.global_pcd
        
        # Update 
    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.global_pc
    
    def vizualise(self):
        try:
            o3d.visualization.draw_geometries([self.global_pc])
        except Exception as e:
            logging.error(f"Error visualizing point cloud: {e}")      

         
