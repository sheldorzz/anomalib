import sys
import numpy as np
import cv2
from pathlib import Path
import json
import logging
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
import matplotlib.cm as cm

# PyVista for 3D visualization - Install with: pip install pyvista pyvistaqt
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as e:
    print("Error: PyVista not found. Please install with: pip install pyvista pyvistaqt")
    sys.exit(1)

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QLabel, QProgressBar, QTextEdit, QGroupBox,
    QSlider, QSpinBox, QCheckBox, QGridLayout,
    QFileDialog, QMessageBox, QSplitter, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPixmap, QImage

# Import our workers
from capture_core import CaptureWorker
from infer_core import InferWorker
from train_core import TrainWorker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudGenerator:
    """Generate point clouds from RGB-D data and camera parameters"""
    
    def __init__(self):
        self.max_depth = 3.0  # Maximum depth in meters
        self.min_depth = 0.1  # Minimum depth in meters
        self.anomaly_colormap = cm.get_cmap('jet')  # For anomaly heatmap
        
        # Hardcoded Volume of Interest (meters)
        self.voi = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.1, 'z_max': 2.0
        }
        
    def create_point_cloud(self, rgb_image, depth_image, camera_matrix, transform_matrix=None):
        """
        Create point cloud from RGB-D image
        
        Args:
            rgb_image: RGB image (H, W, 3) 
            depth_image: Depth image in meters (H, W)
            camera_matrix: 3x3 camera intrinsic matrix
            transform_matrix: 4x4 transformation matrix (optional)
            
        Returns:
            PyVista PolyData point cloud
        """
        height, width = depth_image.shape
        
        # Create intrinsic parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # Create 2D pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth points
        valid_depth = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        # Get valid coordinates
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        z_valid = depth_image[valid_depth]
        
        # Convert to 3D points (camera coordinates)
        x_3d = (u_valid - cx) * z_valid / fx
        y_3d = (v_valid - cy) * z_valid / fy
        z_3d = z_valid
        
        # Create points array
        points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        # Apply transformation if provided
        if transform_matrix is not None:
            # Convert to homogeneous coordinates
            points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
            # Apply transformation
            points_transformed = (transform_matrix @ points_homo.T).T
            points_3d = points_transformed[:, :3]
        
        # Filter by Volume of Interest
        voi_mask = (
            (points_3d[:, 0] >= self.voi['x_min']) & (points_3d[:, 0] <= self.voi['x_max']) &
            (points_3d[:, 1] >= self.voi['y_min']) & (points_3d[:, 1] <= self.voi['y_max']) &
            (points_3d[:, 2] >= self.voi['z_min']) & (points_3d[:, 2] <= self.voi['z_max'])
        )
        
        points_3d = points_3d[voi_mask]
        
        # Get RGB colors for filtered points
        if rgb_image.shape[2] == 3:
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_image
        rgb_valid = rgb[valid_depth]
        rgb_valid = rgb_valid[voi_mask]
        
        # Create PyVista point cloud
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            cloud["RGB"] = rgb_valid
            return cloud
        else:
            return pv.PolyData()
    
    def create_anomaly_point_cloud_heatmap(self, rgb_image, depth_image, camera_matrix, 
                                          anomaly_map, transform_matrix=None):
        """
        Create point cloud with full anomaly heatmap visualization
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image in meters (H, W)
            camera_matrix: 3x3 camera intrinsic matrix
            anomaly_map: Normalized anomaly scores (0-1)
            transform_matrix: 4x4 transformation matrix (optional)
            
        Returns:
            PyVista PolyData point cloud with anomaly heatmap colors
        """
        height, width = depth_image.shape
        
        # Resize anomaly map to match depth image dimensions if needed
        if anomaly_map.shape != (height, width):
            anomaly_map_resized = cv2.resize(
                anomaly_map.astype(np.float32), 
                (width, height), 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            anomaly_map_resized = anomaly_map.astype(np.float32)
        
        # Create intrinsic parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # Create 2D pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth points
        valid_depth = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        # Get valid coordinates
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        z_valid = depth_image[valid_depth]
        anomaly_valid = anomaly_map_resized[valid_depth]
        
        # Convert to 3D points (camera coordinates)
        x_3d = (u_valid - cx) * z_valid / fx
        y_3d = (v_valid - cy) * z_valid / fy
        z_3d = z_valid
        
        # Create points array
        points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        # Apply transformation if provided
        if transform_matrix is not None:
            # Convert to homogeneous coordinates
            points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
            # Apply transformation
            points_transformed = (transform_matrix @ points_homo.T).T
            points_3d = points_transformed[:, :3]
        
        # Filter by Volume of Interest
        voi_mask = (
            (points_3d[:, 0] >= self.voi['x_min']) & (points_3d[:, 0] <= self.voi['x_max']) &
            (points_3d[:, 1] >= self.voi['y_min']) & (points_3d[:, 1] <= self.voi['y_max']) &
            (points_3d[:, 2] >= self.voi['z_min']) & (points_3d[:, 2] <= self.voi['z_max'])
        )
        
        points_3d = points_3d[voi_mask]
        anomaly_valid = anomaly_valid[voi_mask]
        
        # Create PyVista point cloud
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            
            # Apply colormap to anomaly scores
            # Ensure anomaly scores are in [0, 1] range
            anomaly_clipped = np.clip(anomaly_valid, 0, 1)
            
            # Get colors from colormap (returns RGBA, we need RGB)
            colors_rgba = self.anomaly_colormap(anomaly_clipped)
            colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
            
            cloud["RGB"] = colors_rgb
            cloud["Anomaly_Score"] = anomaly_valid
            
            return cloud
        else:
            return pv.PolyData()
    
    def create_anomaly_point_cloud(self, rgb_image, depth_image, camera_matrix, 
                                   anomaly_mask, transform_matrix=None):
        """Original method for binary anomaly visualization (kept for compatibility)"""
        return self.create_anomaly_point_cloud_heatmap(
            rgb_image, depth_image, camera_matrix, 
            anomaly_mask.astype(np.float32), transform_matrix
        )


class PyVistaViewer(QWidget):
    """PyVista-based 3D viewer widget with view persistence"""
    
    def __init__(self, title="3D View"):
        super().__init__()
        self.title = title
        self.current_cloud = pv.PolyData()
        self.voxel_size = 0.005  # 5mm voxels
        self.grid_size = 2.0  # 2m grid
        self.show_grid = True
        self.point_count = 0
        self.camera_position = None  # Store camera position
        self.bounding_boxes = []  # Store bounding box actors
        self.is_voxelized = False
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Info label
        self.info_label = QLabel("No data")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.reset_view_btn = QPushButton("Reset View")
        self.toggle_grid_btn = QPushButton("Hide Grid")
        
        self.clear_btn.clicked.connect(self.clear_point_cloud)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.toggle_grid_btn.clicked.connect(self.toggle_grid)
        
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.reset_view_btn)
        control_layout.addWidget(self.toggle_grid_btn)
        layout.addLayout(control_layout)
        
        # Create PyVista Qt widget
        self.plotter = QtInteractor(self)
        self.plotter.setMinimumSize(400, 300)
        layout.addWidget(self.plotter)
        
        # Configure plotter
        self.plotter.set_background([0.1, 0.1, 0.1])
        self.setup_grid()
        
        # Set layout
        self.setLayout(layout)
        
    def setup_grid(self):
        """Setup 3D coordinate grid"""
        if self.show_grid:
            # Add coordinate axes at origin
            self.plotter.add_axes_at_origin(labels_off=False)
            
            # Create grid planes matching VOI size
            # XY plane (Z=0)
            xy_grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), 
                              i_size=self.grid_size, j_size=self.grid_size,
                              i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(xy_grid, style='wireframe', color='gray', 
                                line_width=1, opacity=0.3, name="xy_grid")
            
            # XZ plane (Y=0)
            xz_grid = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0),
                              i_size=self.grid_size, j_size=2.0,  # Z range is 0.1 to 2.0
                              i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(xz_grid, style='wireframe', color='lightblue',
                                line_width=1, opacity=0.2, name="xz_grid")
            
            # YZ plane (X=0)
            yz_grid = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0),
                              i_size=self.grid_size, j_size=2.0,  # Z range is 0.1 to 2.0
                              i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(yz_grid, style='wireframe', color='lightcoral',
                                line_width=1, opacity=0.2, name="yz_grid")
        
    def downsample_point_cloud(self, cloud, max_points=50000):
        """Downsample point cloud using voxel grid or random sampling"""
        if cloud.n_points <= max_points:
            return cloud
            
        try:
            # Try voxel-based downsampling first (better spatial distribution)
            voxel_size = max(self.voxel_size * 2, 0.01)  # Use larger voxels for downsampling, minimum 1cm
            
            # Create a simple voxel grid downsampling
            points = cloud.points
            
            # Calculate voxel indices
            voxel_indices = np.floor(points / voxel_size).astype(int)
            
            # Find unique voxels
            unique_voxels, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            
            # If we still have too many points, use random sampling
            if len(unique_indices) > max_points:
                unique_indices = np.random.choice(unique_indices, size=max_points, replace=False)
            
            # Create downsampled cloud
            sampled_points = points[unique_indices]
            downsampled_cloud = pv.PolyData(sampled_points)
            
            # Copy array data if it exists
            for array_name in cloud.point_data.keys():
                try:
                    downsampled_cloud[array_name] = cloud[array_name][unique_indices]
                except Exception as e:
                    logger.warning(f"Failed to copy array {array_name}: {e}")
                
            return downsampled_cloud
            
        except Exception as e:
            logger.warning(f"Voxel downsampling failed, using random sampling: {e}")
            
            # Fallback to random sampling
            try:
                n_sample = min(max_points, cloud.n_points)
                if n_sample <= 0:
                    return pv.PolyData()
                    
                indices = np.random.choice(cloud.n_points, size=n_sample, replace=False)
                sampled_points = cloud.points[indices]
                sampled_cloud = pv.PolyData(sampled_points)
                
                # Copy array data
                for array_name in cloud.point_data.keys():
                    try:
                        sampled_cloud[array_name] = cloud[array_name][indices]
                    except Exception as e:
                        logger.warning(f"Failed to copy array {array_name} in fallback: {e}")
                    
                return sampled_cloud
                
            except Exception as e2:
                logger.error(f"Both downsampling methods failed: {e2}")
                return cloud  # Return original if all else fails

    def save_camera_position(self):
        """Save current camera position"""
        if hasattr(self.plotter, 'camera_position'):
            self.camera_position = self.plotter.camera_position

    def restore_camera_position(self):
        """Restore saved camera position"""
        if self.camera_position is not None:
            try:
                self.plotter.camera_position = self.camera_position
            except:
                pass  # Ignore if restoration fails

    def add_point_cloud(self, cloud, clear_previous=False):
        """Add point cloud to visualization with view persistence"""
        try:
            # Save camera position before update
            self.save_camera_position()
            
            if clear_previous:
                self.current_cloud = pv.PolyData()
                
            # Combine with existing cloud
            if cloud.n_points > 0:
                if self.current_cloud.n_points > 0:
                    self.current_cloud = self.current_cloud + cloud
                else:
                    self.current_cloud = cloud.copy()
                
                # Downsample if too many points using proper point cloud methods
                if self.current_cloud.n_points > 100000:
                    self.current_cloud = self.downsample_point_cloud(self.current_cloud, max_points=50000)
                
                # Update visualization
                try:
                    self.plotter.add_mesh(self.current_cloud, scalars="RGB", rgb=True, 
                                        point_size=3, name="point_cloud", render_points_as_spheres=True)
                except Exception as e:
                    logger.warning(f"Failed to add mesh with RGB, trying without: {e}")
                    # Fallback without RGB if it fails
                    self.plotter.add_mesh(self.current_cloud, point_size=3, name="point_cloud", 
                                        render_points_as_spheres=True)
                
                # Update info
                self.point_count = self.current_cloud.n_points
                self.info_label.setText(f"Points: {self.point_count:,}")
                
                # Restore camera position after update
                self.restore_camera_position()
                
        except Exception as e:
            logger.error(f"Failed to add point cloud: {e}")
            self.info_label.setText(f"Error: {e}")
    
    def clear_point_cloud(self):
        """Clear point cloud but keep grid"""
        try:
            self.current_cloud = pv.PolyData()
            self.point_count = 0
            self.is_voxelized = False
            
            # Remove point cloud and bounding boxes
            try:
                self.plotter.remove_actor("point_cloud", render=False)
            except:
                pass  # Actor might not exist
                
            # Clear bounding boxes
            self.clear_bounding_boxes()
            
            self.plotter.render()
            self.info_label.setText("Cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear point cloud: {e}")
    
    def clear_bounding_boxes(self):
        """Clear all bounding boxes"""
        for i in range(len(self.bounding_boxes)):
            try:
                self.plotter.remove_actor(f"bbox_{i}", render=False)
            except:
                pass
        self.bounding_boxes.clear()
    
    def add_bounding_boxes(self, bounding_boxes):
        """Add 3D bounding boxes to the visualization"""
        # Save camera position
        self.save_camera_position()
        
        self.clear_bounding_boxes()
        
        for i, bbox in enumerate(bounding_boxes):
            try:
                min_pt = np.array(bbox['min'])
                max_pt = np.array(bbox['max'])
                
                # Create box mesh
                box = pv.Box(bounds=[
                    min_pt[0], max_pt[0],
                    min_pt[1], max_pt[1],
                    min_pt[2], max_pt[2]
                ])
                
                # Color based on anomaly score
                color = 'red' if bbox['mean_score'] > 0.9 else 'orange'
                
                # Store the box for re-rendering after voxelization
                bbox['box_mesh'] = box
                bbox['color'] = color
                
                self.plotter.add_mesh(
                    box, style='wireframe', color=color, 
                    line_width=3, opacity=0.8, name=f"bbox_{i}",
                    render_points_as_spheres=False
                )
                
                self.bounding_boxes.append(bbox)
                
            except Exception as e:
                logger.error(f"Failed to add bounding box {i}: {e}")
        
        # Restore camera position
        self.restore_camera_position()
        
        # Force render
        self.plotter.render()
        
        logger.info(f"Added {len(self.bounding_boxes)} bounding boxes to {self.title}")
    
    def voxelize_point_cloud(self):
        """Convert point cloud to voxel representation"""
        if self.current_cloud.n_points == 0:
            logger.warning("No points to voxelize")
            return
            
        try:
            # Save camera position
            self.save_camera_position()
            
            logger.info(f"Voxelizing {self.current_cloud.n_points} points with voxel size {self.voxel_size}m")
            
            # Create voxel grid
            voxel_grid = pv.voxelize(self.current_cloud, cell_size=self.voxel_size)
            
            logger.info(f"Created voxel grid with {voxel_grid.n_cells} voxels")
            
            # Update visualization
            self.plotter.remove_actor("point_cloud", render=False)
            
            # Check if voxel grid has RGB data
            has_rgb = False
            if "RGB" in voxel_grid.point_data:
                has_rgb = True
                logger.info("Voxel grid has RGB data")
            elif "RGB" in voxel_grid.cell_data:
                has_rgb = True
                logger.info("Voxel grid has RGB cell data")
                # Move cell data to point data for visualization
                voxel_grid = voxel_grid.cell_data_to_point_data()
            
            if has_rgb:
                try:
                    self.plotter.add_mesh(
                        voxel_grid, scalars="RGB", rgb=True, 
                        opacity=0.8, name="point_cloud"
                    )
                except Exception as e:
                    logger.warning(f"Failed to render with RGB: {e}")
                    # Fallback to no RGB
                    self.plotter.add_mesh(
                        voxel_grid, opacity=0.8, name="point_cloud", color='gray'
                    )
            else:
                logger.info("No RGB data in voxel grid, using default color")
                self.plotter.add_mesh(
                    voxel_grid, opacity=0.8, name="point_cloud", color='gray'
                )
            
            self.is_voxelized = True
            self.info_label.setText(f"Voxels: {voxel_grid.n_cells:,}")
            
            # Re-add bounding boxes on top of voxels
            if self.bounding_boxes:
                logger.info(f"Re-rendering {len(self.bounding_boxes)} bounding boxes")
                for i, bbox in enumerate(self.bounding_boxes):
                    # Re-add the bounding box mesh
                    if 'box_mesh' in bbox and 'color' in bbox:
                        try:
                            self.plotter.add_mesh(
                                bbox['box_mesh'], style='wireframe', color=bbox['color'], 
                                line_width=3, opacity=0.8, name=f"bbox_{i}",
                                render_points_as_spheres=False
                            )
                        except Exception as e:
                            logger.error(f"Failed to re-add bounding box {i}: {e}")
            
            # Restore camera position
            self.restore_camera_position()
            
            # Force final render
            self.plotter.render()
            
            logger.info("Voxelization complete")
            
        except Exception as e:
            logger.error(f"Failed to voxelize: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.info_label.setText(f"Voxelize error: {str(e)[:50]}")
    
    def toggle_grid(self):
        """Toggle grid visibility"""
        self.show_grid = not self.show_grid
        self.toggle_grid_btn.setText("Show Grid" if not self.show_grid else "Hide Grid")
        
        if self.show_grid:
            self.setup_grid()
        else:
            # Remove grid elements (axes are handled separately by PyVista)
            for name in ["xy_grid", "xz_grid", "yz_grid"]:
                try:
                    self.plotter.remove_actor(name, render=False)
                except:
                    pass  # Actor might not exist
        
        self.plotter.render()
        logger.info(f"Grid {'shown' if self.show_grid else 'hidden'}")
    
    def reset_view(self):
        """Reset camera view"""
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.camera_position = None
        
    def set_voxel_size(self, size_mm):
        """Set voxel size in millimeters"""
        self.voxel_size = size_mm / 1000.0
        logger.info(f"Voxel size set to {size_mm}mm")
        
    def set_grid_size(self, size_m):
        """Set grid size in meters"""
        self.grid_size = float(size_m)
        
        # Remove old grid (but not axes - they stay the same)
        for name in ["xy_grid", "xz_grid", "yz_grid"]:
            try:
                self.plotter.remove_actor(name, render=False)
            except:
                pass  # Actor might not exist
            
        # Add new grid
        if self.show_grid:
            self.setup_grid()
            
        self.plotter.render()
        logger.info(f"Grid size set to {size_m}m")


class LiveRGBWidget(QWidget):
    """Widget for displaying live RGB feed"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Live RGB Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setMaximumSize(640, 480)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: 2px solid #333;")
        self.image_label.setText("No feed")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Info label
        self.info_label = QLabel("Waiting for feed...")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
    def update_image(self, bgr_image):
        """Update displayed image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            height, width, channel = rgb_image.shape
            if width > 640 or height > 480:
                scale = min(640/width, 480/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Display
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            
            self.info_label.setText(f"Feed: {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to update RGB image: {e}")


class RobotVision3DApp(QMainWindow):
    """Main application for 3D robot vision with anomaly detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Vision 3D - Anomaly Detection")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1400, 800)
        
        # Workers and threads
        self.capture_worker = None
        self.capture_thread = None
        self.infer_worker = None
        self.infer_thread = None
        
        # Point cloud generator
        self.pc_generator = PointCloudGenerator()
        
        # Configuration
        self.waypoints_config = None
        self.model_path = "models/weights/torch/patchcore.pt"
        
        # Capture state
        self.is_capturing = False
        self.capture_complete = False
        
        self.setup_ui()
        self.setup_connections()
        
        # Log VOI settings
        voi = self.pc_generator.voi
        self.log(f"Volume of Interest: X=[{voi['x_min']:.1f}, {voi['x_max']:.1f}]m, " +
                 f"Y=[{voi['y_min']:.1f}, {voi['y_max']:.1f}]m, " +
                 f"Z=[{voi['z_min']:.1f}, {voi['z_max']:.1f}]m")
        
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls and RGB feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Control panel
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel)
        
        # Live RGB feed
        self.rgb_widget = LiveRGBWidget()
        left_layout.addWidget(self.rgb_widget)
        
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - 3D Views
        visualization_panel = self.create_visualization_panel()
        main_layout.addWidget(visualization_panel, 3)
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File selection
        file_group = QGroupBox("Configuration")
        file_layout = QVBoxLayout(file_group)
        
        self.waypoints_btn = QPushButton("Load Waypoints Config")
        self.model_btn = QPushButton("Load Model")
        self.waypoints_label = QLabel("No waypoints loaded")
        self.model_label = QLabel("No model loaded")
        
        file_layout.addWidget(self.waypoints_btn)
        file_layout.addWidget(self.waypoints_label)
        file_layout.addWidget(self.model_btn)
        file_layout.addWidget(self.model_label)
        
        # Capture controls
        capture_group = QGroupBox("Data Capture")
        capture_layout = QVBoxLayout(capture_group)
        
        self.start_capture_btn = QPushButton("Start Capture")
        self.stop_capture_btn = QPushButton("Stop Capture")
        self.stop_capture_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        
        capture_layout.addWidget(self.start_capture_btn)
        capture_layout.addWidget(self.stop_capture_btn)
        capture_layout.addWidget(self.progress_bar)
        capture_layout.addWidget(self.status_label)
        
        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add all groups to layout
        layout.addWidget(file_group)
        layout.addWidget(capture_group)
        layout.addWidget(log_group)
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create visualization panel with split view"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("3D Point Cloud Visualization")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Split view
        splitter = QSplitter(Qt.Horizontal)
        
        self.normal_viewer = PyVistaViewer("Normal RGB Point Cloud")
        self.anomaly_viewer = PyVistaViewer("Anomaly Heatmap Point Cloud")
        
        splitter.addWidget(self.normal_viewer)
        splitter.addWidget(self.anomaly_viewer)
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter)
        
        return panel
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        # File buttons
        self.waypoints_btn.clicked.connect(self.load_waypoints_config)
        self.model_btn.clicked.connect(self.load_model)
        
        # Capture buttons
        self.start_capture_btn.clicked.connect(self.start_capture)
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        
    def load_waypoints_config(self):
        """Load waypoints configuration"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Waypoints Config", "", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.waypoints_config = json.load(f)
                self.waypoints_label.setText(f"Loaded: {Path(filename).name}")
                self.log(f"Loaded waypoints: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load waypoints: {e}")
                
    def load_model(self):
        """Load anomaly detection model"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "PyTorch Files (*.pt);;All Files (*)"
        )
        if filename:
            self.model_path = filename
            self.model_label.setText(f"Loaded: {Path(filename).name}")
            self.log(f"Loaded model: {filename}")
    
    def start_capture(self):
        """Start capture and inference"""
        if not self.waypoints_config:
            QMessageBox.warning(self, "Warning", "Please load waypoints configuration first")
            return
            
        if not Path(self.model_path).exists():
            QMessageBox.warning(self, "Warning", f"Model not found: {self.model_path}")
            return
        
        try:
            # Reset state
            self.is_capturing = True
            self.capture_complete = False
            self._waypoint_count = 0  # Reset waypoint counter
            
            # Clear existing point clouds
            self.normal_viewer.clear_point_cloud()
            self.anomaly_viewer.clear_point_cloud()
            self.log("Cleared existing point clouds")
            
            # Setup inference worker first
            self.log("Setting up inference worker...")
            self.setup_inference_worker()
            
            # Setup capture worker
            self.log("Setting up capture worker...")
            self.setup_capture_worker()
            
            # Verify signal connections
            self.log("Signal connections established")
            
            # Start inference worker
            self.log("Starting inference thread...")
            self.infer_thread.start()
            
            # Wait for inference to be ready
            self.log("Waiting for inference to initialize...")
            QTimer.singleShot(500, lambda: self.start_capture_delayed())
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to start capture: {e}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log(error_msg)
            
            # Reset state on error
            self.is_capturing = False
            self.start_capture_btn.setEnabled(True)
            self.stop_capture_btn.setEnabled(False)
            
    def start_capture_delayed(self):
        """Start capture after delay"""
        try:
            self.log("Starting capture thread...")
            self.capture_thread.start()
            
            # Update UI
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(True)
            self.status_label.setText("Capturing...")
            self.log("Capture and inference started successfully")
        except Exception as e:
            error_msg = f"Failed to start capture thread: {e}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
    
    def stop_capture(self):
        """Stop capture and inference"""
        try:
            self.is_capturing = False
            
            if self.capture_worker is not None:
                self.capture_worker.stop_capture()
            
            # Don't stop inference here - let it finish processing
            # It will be stopped when capture finishes
            
            self.status_label.setText("Stopping...")
            self.log("Stopping capture...")
            
        except Exception as e:
            self.log(f"Error stopping capture: {e}")
    
    def setup_capture_worker(self):
        """Setup capture worker and thread"""
        # Clean up any existing threads
        if hasattr(self, 'capture_thread') and self.capture_thread is not None:
            if self.capture_thread.isRunning():
                self.capture_thread.quit()
                self.capture_thread.wait()
            
        self.capture_thread = QThread()
        self.capture_worker = CaptureWorker(self.waypoints_config)
        self.capture_worker.moveToThread(self.capture_thread)
        
        # Connect signals
        self.capture_thread.started.connect(self.capture_worker.start_capture)
        self.capture_worker.progress.connect(self.progress_bar.setValue)
        self.capture_worker.error.connect(self.log)
        self.capture_worker.live_frame.connect(self.process_live_frame)
        self.capture_worker.finished.connect(self.on_capture_finished)  # Call this first
        self.capture_worker.finished.connect(self.capture_thread.quit)  # Then quit thread
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        
    def setup_inference_worker(self):
        """Setup inference worker and thread"""
        # Clean up any existing threads
        if hasattr(self, 'infer_thread') and self.infer_thread is not None:
            if self.infer_thread.isRunning():
                self.infer_thread.quit()
                self.infer_thread.wait()
            
        self.infer_thread = QThread()
        self.infer_worker = InferWorker(self.model_path)
        self.infer_worker.moveToThread(self.infer_thread)
        
        # Enable real-time mode
        self.infer_worker.enable_real_time_mode(True)
        
        # Set hardcoded VOI (same as in PointCloudGenerator)
        hardcoded_voi = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.1, 'z_max': 2.0
        }
        self.infer_worker.set_volume_of_interest(hardcoded_voi)
        
        # Connect signals
        self.infer_thread.started.connect(self.infer_worker.start_inference)
        self.infer_worker.error.connect(self.log)
        self.infer_worker.live_inference_result.connect(self.update_3d_visualization)
        self.infer_worker.anomaly_accumulation_complete.connect(self.on_anomaly_accumulation_complete)
        self.infer_worker.finished.connect(self.infer_thread.quit)
        self.infer_worker.finished.connect(self.on_inference_finished)
        self.infer_thread.finished.connect(self.infer_thread.deleteLater)
    
    @Slot(object)
    def process_live_frame(self, frame_data):
        """Process live frame from capture worker"""
        # Update RGB display
        if 'rgb' in frame_data:
            self.rgb_widget.update_image(frame_data['rgb'])
        
        # Forward to inference worker
        if (self.infer_worker is not None and 
            hasattr(self.infer_worker, '_running') and 
            self.infer_worker._running):
            self.infer_worker.process_live_frame(frame_data)
        else:
            logger.warning("Inference worker not ready to process frame")
    
    @Slot(object)
    def update_3d_visualization(self, inference_result):
        """Update 3D visualization with inference results"""
        try:
            frame_data = inference_result["frame_data"]
            normalized_anomaly_map = inference_result.get("normalized_anomaly_map")
            has_anomaly = inference_result["has_anomaly"]
            
            # Extract data
            rgb_image = frame_data["rgb"]
            depth_image = frame_data["depth"]
            camera_matrix = frame_data["camera_matrix"]
            transform_matrix = frame_data["transform_matrix"]
            
            # Validate dimensions
            if rgb_image is None or depth_image is None:
                self.log("Warning: Invalid frame data received")
                return
            
            # Generate normal point cloud
            normal_cloud = self.pc_generator.create_point_cloud(
                rgb_image, depth_image, camera_matrix, transform_matrix
            )
            
            # Generate anomaly heatmap point cloud
            anomaly_cloud = self.pc_generator.create_anomaly_point_cloud_heatmap(
                rgb_image, depth_image, camera_matrix, normalized_anomaly_map, transform_matrix
            )
            
            # Check if point clouds are valid
            if normal_cloud.n_points == 0:
                logger.warning("Empty normal point cloud generated")
                return
                
            # Add to viewers
            self.normal_viewer.add_point_cloud(normal_cloud)
            self.anomaly_viewer.add_point_cloud(anomaly_cloud)
            
            # Log progress every 10 waypoints
            waypoint_id = frame_data.get("waypoint_id", "unknown")
            if hasattr(self, '_waypoint_count'):
                self._waypoint_count += 1
            else:
                self._waypoint_count = 1
                
            if self._waypoint_count % 10 == 0:
                normal_total = self.normal_viewer.point_count
                anomaly_total = self.anomaly_viewer.point_count
                self.log(f"Progress: {self._waypoint_count} waypoints, Normal={normal_total:,} pts, Anomaly={anomaly_total:,} pts")
            
            # Log anomaly detection
            if has_anomaly:
                score = inference_result["anomaly_score"]
                self.log(f"🚨 ANOMALY at {waypoint_id}: score={score:.3f}")
                
        except Exception as e:
            import traceback
            self.log(f"Error updating 3D visualization: {e}")
            logger.error(f"3D visualization error: {e}\n{traceback.format_exc()}")
    
    @Slot(object)
    def on_anomaly_accumulation_complete(self, result):
        """Handle completed anomaly accumulation and bounding box computation"""
        try:
            self.log("Anomaly accumulation complete signal received")
            
            bounding_boxes = result.get('bounding_boxes', [])
            
            self.log(f"Anomaly processing complete: {len(bounding_boxes)} regions detected")
            
            if len(bounding_boxes) > 0:
                # Add bounding boxes to viewers
                self.normal_viewer.add_bounding_boxes(bounding_boxes)
                self.anomaly_viewer.add_bounding_boxes(bounding_boxes)
                
                # Log RGB image associations
                for i, bbox in enumerate(bounding_boxes):
                    if bbox.get('best_view'):
                        self.log(f"BBox {i}: Best view from {bbox['best_view']} (score: {bbox['max_score']:.3f})")
            else:
                self.log("No anomalies detected within VOI thresholds")
            
            # Always voxelize point clouds after processing
            self.log("Starting voxelization...")
            
            # Check if we have point clouds to voxelize
            if self.normal_viewer.current_cloud.n_points > 0:
                self.log(f"Voxelizing normal view ({self.normal_viewer.current_cloud.n_points} points)")
                self.normal_viewer.voxelize_point_cloud()
            else:
                self.log("Warning: No points in normal viewer to voxelize")
                
            if self.anomaly_viewer.current_cloud.n_points > 0:
                self.log(f"Voxelizing anomaly view ({self.anomaly_viewer.current_cloud.n_points} points)")
                self.anomaly_viewer.voxelize_point_cloud()
            else:
                self.log("Warning: No points in anomaly viewer to voxelize")
            
            self.status_label.setText("Post-processing complete")
            self.log("Post-processing complete")
            
        except Exception as e:
            self.log(f"Error in post-processing: {e}")
            logger.error(f"Anomaly accumulation error: {traceback.format_exc()}")
    
    def on_capture_finished(self):
        """Handle capture finished"""
        self.capture_complete = True
        self.is_capturing = False
        
        # Clean up capture references
        self.capture_worker = None
        self.capture_thread = None
        
        self.status_label.setText("Capture completed - Processing anomalies...")
        self.log("Capture completed, processing anomalies...")
        
        # Add a small delay to ensure all frames are processed
        QTimer.singleShot(1000, self.trigger_anomaly_processing)
        
    def trigger_anomaly_processing(self):
        """Trigger anomaly accumulation processing after delay"""
        if self.infer_worker is not None:
            # Log current state
            if hasattr(self.infer_worker, 'accumulated_anomaly_maps'):
                self.log(f"Triggering anomaly processing with {len(self.infer_worker.accumulated_anomaly_maps)} accumulated frames")
            else:
                self.log("Triggering anomaly processing...")
                
            # Stop inference which will trigger accumulated anomaly processing
            self.infer_worker.stop_inference()
        else:
            # If no inference worker, just update UI
            self.start_capture_btn.setEnabled(True)
            self.stop_capture_btn.setEnabled(False)
            self.status_label.setText("Capture completed")
    
    def on_inference_finished(self):
        """Handle inference worker finished"""
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.status_label.setText("Ready")
        self.log("Inference processing finished")
        
        # Clean up references
        self.infer_worker = None
        self.infer_thread = None
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        try:
            # Stop any running captures
            if self.is_capturing:
                self.stop_capture()
                
            # Wait for threads to finish
            if hasattr(self, 'capture_thread') and self.capture_thread is not None:
                if self.capture_thread.isRunning():
                    self.capture_thread.quit()
                    self.capture_thread.wait(2000)
                
            if hasattr(self, 'infer_thread') and self.infer_thread is not None:
                if self.infer_thread.isRunning():
                    self.infer_thread.quit()
                    self.infer_thread.wait(2000)
                
            # Close PyVista plotters
            if hasattr(self, 'normal_viewer') and self.normal_viewer.plotter:
                self.normal_viewer.plotter.close()
            if hasattr(self, 'anomaly_viewer') and self.anomaly_viewer.plotter:
                self.anomaly_viewer.plotter.close()
        except:
            pass
        event.accept()


def main():
    """Main function"""
    try:
        # Required for PyVista Qt integration
        pv.global_theme.allow_empty_mesh = True
    except Exception as e:
        print(f"Warning: Could not configure PyVista theme: {e}")
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Robot Vision 3D")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Robot Vision Lab")
    
    try:
        # Create main window
        window = RobotVision3DApp()
        window.show()
        
        # Run application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure you have installed: pip install pyvista pyvistaqt")
        sys.exit(1)


if __name__ == "__main__":
    main()