# workers/pc_rgb.py
import numpy as np
import open3d as o3d
from PySide6.QtCore import QObject, Signal, QThread,Slot
import logging
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RgbPCWorker(QObject):
    """Worker for building and updating RGB point cloud"""
    
    # Signals
    started = Signal()
    point_cloud_updated = Signal(object)  # Emits dict with points and colors
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, max_points=1000000, voxel_size=0.005):
        super().__init__()
        self.is_running = False
        self.max_points = max_points
        self.voxel_size = voxel_size
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Point cloud storage
        self.point_cloud = o3d.geometry.PointCloud()
        self.accumulated_points = []
        self.accumulated_colors = []
        
        # Camera intrinsics (RealSense D455 at 1280x720)
        self.setup_camera_intrinsics()
        
        # Frame buffer for temporal filtering
        self.frame_buffer = deque(maxlen=5)
        
        # Downsampling counter
        self.frame_counter = 0
        self.downsample_rate = 3  # Process every Nth frame
        
    def setup_camera_intrinsics(self):
        """Setup camera intrinsic parameters for RealSense D455"""
        # Typical intrinsics for D455 at 1280x720
        # These should ideally be read from the camera
        width = 1280
        height = 720
        fx = 643.884  # Focal length x
        fy = 643.884  # Focal length y
        cx = 634.661  # Principal point x
        cy = 360.029  # Principal point y
        
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
    
    def transform_to_robot_base(self, points, pose):
        """Transform points from camera frame to robot base frame
        
        Args:
            points: Nx3 array of points in camera frame
            pose: Robot TCP pose [x, y, z, rx, ry, rz]
            
        Returns:
            Nx3 array of transformed points
        """
        # Get transformation matrix from pose
        T = self.pose_to_matrix(pose)
        
        # Add homogeneous coordinate
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform points
        points_transformed = (T @ points_h.T).T[:, :3]
        
        return points_transformed
    
    def pose_to_matrix(self, pose):
        """Convert robot pose to 4x4 transformation matrix"""
        x, y, z, rx, ry, rz = pose
        
        # Rotation vector to rotation matrix (Rodrigues' formula)
        theta = np.sqrt(rx**2 + ry**2 + rz**2)
        if theta > 0:
            k = np.array([rx, ry, rz]) / theta
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        else:
            R = np.eye(3)
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    def create_point_cloud_from_rgbd(self, rgb, depth, pose):
        """Create point cloud from RGB-D data
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image in meters (H, W)
            pose: Robot TCP pose
            
        Returns:
            Tuple of (points, colors)
        """
        # Create RGBD image
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Convert to mm
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, 
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=3.0,  # 3 meter truncation
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud from RGBD
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsic,
            extrinsic=np.eye(4)  # Identity, we'll transform later
        )
        
        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Remove invalid points
        valid_mask = ~np.any(np.isnan(points), axis=1)
        valid_mask &= ~np.any(np.isinf(points), axis=1)
        valid_mask &= np.all(points != 0, axis=1)  # Remove zero points
        
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Transform to robot base frame
        if points.shape[0] > 0:
            points = self.transform_to_robot_base(points, pose)
        
        return points, colors
    
    def downsample_point_cloud(self):
        """Downsample accumulated point cloud using voxel grid"""
        if len(self.accumulated_points) == 0:
            return
        
        # Create temporary point cloud
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(
            np.vstack(self.accumulated_points)
        )
        temp_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack(self.accumulated_colors)
        )
        
        # Voxel downsample
        downsampled = temp_pcd.voxel_down_sample(self.voxel_size)
        
        # Update accumulated points
        self.accumulated_points = [np.asarray(downsampled.points)]
        self.accumulated_colors = [np.asarray(downsampled.colors)]
        
        # Update main point cloud
        self.point_cloud = downsampled
    
    def update_point_cloud(self, points, colors):
        """Update the cumulative point cloud with new points"""
        with self.lock:
            # Add new points
            if points.shape[0] > 0:
                self.accumulated_points.append(points)
                self.accumulated_colors.append(colors)
                
                # Get total point count
                total_points = sum(p.shape[0] for p in self.accumulated_points)
                
                # Downsample if exceeding max points
                if total_points > self.max_points:
                    self.downsample_point_cloud()
                    logger.info(f"Downsampled point cloud to {len(self.accumulated_points[0])} points")
    
    def get_current_point_cloud(self):
        """Get current point cloud data for visualization"""
        with self.lock:
            if len(self.accumulated_points) == 0:
                return {'points': np.array([]), 'colors': np.array([])}
            
            # Combine all accumulated points
            all_points = np.vstack(self.accumulated_points)
            all_colors = np.vstack(self.accumulated_colors)
            
            return {
                'points': all_points,
                'colors': all_colors,
                'num_points': all_points.shape[0]
            }
    
    @Slot()
    def start(self):
        """Start the worker"""
        self.started.emit()
        self.is_running = True
        logger.info("RGB point cloud worker started")
    
    @Slot(object)
    def process_frame(self, frame_data):
        """Process frame data from capture worker
        
        Args:
            frame_data: Dict with 'rgb', 'depth', 'pose' keys
        """
        if not self.is_running:
            return
        
        try:
            # Downsample processing rate
            self.frame_counter += 1
            if self.frame_counter % self.downsample_rate != 0:
                return
            
            # Extract data
            rgb = frame_data['rgb']
            depth = frame_data['depth']
            pose = frame_data['pose']
            
            # Skip if invalid data
            if rgb is None or depth is None or pose is None:
                return
            
            # Create point cloud from RGBD
            points, colors = self.create_point_cloud_from_rgbd(rgb, depth, pose)
            
            # Update cumulative point cloud
            if points.shape[0] > 0:
                self.update_point_cloud(points, colors)
                
                # Emit updated point cloud for visualization
                pc_data = self.get_current_point_cloud()
                self.point_cloud_updated.emit(pc_data)
                
                logger.debug(f"Point cloud updated: {pc_data['num_points']} total points")
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            self.error.emit(f"Point cloud processing error: {str(e)}")
    
    def save_point_cloud(self, filename):
        """Save current point cloud to file"""
        try:
            with self.lock:
                if len(self.accumulated_points) > 0:
                    # Create final point cloud
                    final_pcd = o3d.geometry.PointCloud()
                    final_pcd.points = o3d.utility.Vector3dVector(
                        np.vstack(self.accumulated_points)
                    )
                    final_pcd.colors = o3d.utility.Vector3dVector(
                        np.vstack(self.accumulated_colors)
                    )
                    
                    # Estimate normals
                    final_pcd.estimate_normals()
                    
                    # Save to file
                    o3d.io.write_point_cloud(filename, final_pcd)
                    logger.info(f"Saved point cloud to {filename}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error saving point cloud: {str(e)}")
            return False
    
    def clear_point_cloud(self):
        """Clear accumulated point cloud"""
        with self.lock:
            self.accumulated_points.clear()
            self.accumulated_colors.clear()
            self.point_cloud.clear()
            self.frame_counter = 0
            logger.info("Point cloud cleared")
    
    @Slot()
    def stop(self):
        """Stop the worker"""
        self.is_running = False
        
        # Optionally save final point cloud
        timestamp = np.datetime64('now').astype(str).replace(':', '-')
        filename = f"data/point_clouds/rgb_pc_{timestamp}.ply"
        self.save_point_cloud(filename)
        
        self.finished.emit()
        logger.info("RGB point cloud worker stopped")