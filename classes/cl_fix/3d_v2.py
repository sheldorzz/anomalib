import sys
import numpy as np
import cv2
from pathlib import Path
import json
import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

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
    QFileDialog, QMessageBox, QSplitter, QSizePolicy, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPixmap, QImage

# Import our workers
from capture_core import CaptureWorker
from infer_core import InferWorker
from train_core import TrainWorker

# Try to import sklearn for clustering
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Anomaly region clustering will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudGenerator:
    """Generate point clouds from RGB-D data and camera parameters"""
    
    def __init__(self):
        self.max_depth = 3.0  # Maximum depth in meters
        self.min_depth = 0.1  # Minimum depth in meters
        
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
        
        # Get RGB colors
        if rgb_image.shape[2] == 3:
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_image
        rgb_valid = rgb[valid_depth]
        
        # Create PyVista point cloud
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            cloud["RGB"] = rgb_valid
            return cloud
        else:
            return pv.PolyData()
    
    def create_anomaly_point_cloud(self, rgb_image, depth_image, camera_matrix, 
                                   anomaly_map, transform_matrix=None, threshold=None):
        """
        Create point cloud with full anomaly heatmap visualization
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image in meters (H, W)
            camera_matrix: 3x3 camera intrinsic matrix
            anomaly_map: Normalized anomaly map (0-1) - NOT binary mask
            transform_matrix: 4x4 transformation matrix (optional)
            threshold: Optional threshold for filtering (if None, show all points)
            
        Returns:
            PyVista PolyData point cloud with anomaly heatmap
        """
        height, width = depth_image.shape
        
        # Resize anomaly map to match depth image dimensions if needed
        if anomaly_map.shape != (height, width):
            anomaly_map_resized = cv2.resize(
                anomaly_map.astype(np.float32), 
                (width, height), 
                interpolation=cv2.INTER_LINEAR  # Use linear for smooth interpolation
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
        
        # Optional: filter by anomaly threshold
        if threshold is not None:
            valid_depth = valid_depth & (anomaly_map_resized > threshold)
        
        # Get valid coordinates
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        z_valid = depth_image[valid_depth]
        anomaly_scores = anomaly_map_resized[valid_depth]
        
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
        
        # Create PyVista point cloud
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            
            # Create color map based on anomaly scores
            # Use matplotlib colormap for better visualization
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # Use 'jet' colormap: blue (low) -> red (high)
            cmap = cm.get_cmap('jet')
            colors = cmap(anomaly_scores)[:, :3]  # Get RGB, ignore alpha
            colors = (colors * 255).astype(np.uint8)
            
            cloud["RGB"] = colors
            cloud["Anomaly_Score"] = anomaly_scores
            
            return cloud
        else:
            return pv.PolyData()


class PyVistaViewer(QWidget):
    """PyVista-based 3D viewer widget"""
    
    def __init__(self, title="3D View"):
        super().__init__()
        self.title = title
        self.current_cloud = pv.PolyData()
        self.voxel_size = 0.005  # 5mm voxels
        self.grid_size = 2.0  # 2m grid
        self.show_grid = True
        self.point_count = 0
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
            
            # Create grid planes
            # XY plane (Z=0)
            xy_grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), 
                              i_size=self.grid_size, j_size=self.grid_size,
                              i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(xy_grid, style='wireframe', color='gray', 
                                line_width=1, opacity=0.3, name="xy_grid")
            
            # XZ plane (Y=0)
            xz_grid = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0),
                              i_size=self.grid_size, j_size=self.grid_size,
                              i_resolution=20, j_resolution=20)
            self.plotter.add_mesh(xz_grid, style='wireframe', color='lightblue',
                                line_width=1, opacity=0.2, name="xz_grid")
            
            # YZ plane (X=0)
            yz_grid = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0),
                              i_size=self.grid_size, j_size=self.grid_size,
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

    def add_point_cloud(self, cloud, clear_previous=False, maintain_view=True):
        """Add point cloud to visualization
        
        Args:
            cloud: PyVista point cloud
            clear_previous: Whether to clear previous cloud
            maintain_view: Whether to maintain camera position
        """
        try:
            # Store camera position if maintaining view
            camera_position = None
            if maintain_view and hasattr(self.plotter, 'camera_position'):
                camera_position = self.plotter.camera_position
            
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
                
                # Restore camera position if maintaining view
                if maintain_view and camera_position is not None:
                    self.plotter.camera_position = camera_position
                
                # Update info
                self.point_count = self.current_cloud.n_points
                self.info_label.setText(f"Points: {self.point_count:,}")
                
        except Exception as e:
            logger.error(f"Failed to add point cloud: {e}")
            self.info_label.setText(f"Error: {e}")
    
    def clear_point_cloud(self):
        """Clear point cloud but keep grid"""
        try:
            self.current_cloud = pv.PolyData()
            self.point_count = 0
            
            # Remove only point cloud, keep grid elements
            try:
                self.plotter.remove_actor("point_cloud", render=False)
            except:
                pass  # Actor might not exist
            self.plotter.render()
            
            self.info_label.setText("Cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear point cloud: {e}")
    
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


class RobotVision3DApp(QMainWindow):
    """Main application for 3D robot vision with anomaly detection"""
    
    # Add signal to trigger processing
    trigger_processing = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Vision 3D - Anomaly Detection")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1200, 700)
        
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
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
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
        
        # Live RGB Feed
        rgb_group = QGroupBox("Live RGB Feed")
        rgb_layout = QVBoxLayout(rgb_group)
        
        self.rgb_label = QLabel()
        self.rgb_label.setMinimumSize(320, 240)
        self.rgb_label.setMaximumSize(640, 480)
        self.rgb_label.setScaledContents(True)
        self.rgb_label.setStyleSheet("QLabel { background-color: black; }")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setText("No feed")
        
        rgb_layout.addWidget(self.rgb_label)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QGridLayout(viz_group)
        
        self.voxel_size_spin = QSpinBox()
        self.voxel_size_spin.setRange(1, 20)
        self.voxel_size_spin.setValue(5)
        self.voxel_size_spin.setSuffix(" mm")
        
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(1, 10)
        self.grid_size_spin.setValue(2)
        self.grid_size_spin.setSuffix(" m")
        
        viz_layout.addWidget(QLabel("Voxel Size:"), 0, 0)
        viz_layout.addWidget(self.voxel_size_spin, 0, 1)
        viz_layout.addWidget(QLabel("Grid Size:"), 1, 0)
        viz_layout.addWidget(self.grid_size_spin, 1, 1)
        
        # Anomaly detection controls
        anomaly_group = QGroupBox("Anomaly Detection")
        anomaly_layout = QGridLayout(anomaly_group)
        
        self.image_threshold_slider = QSlider(Qt.Horizontal)
        self.image_threshold_slider.setRange(0, 100)
        self.image_threshold_slider.setValue(80)
        self.image_threshold_label = QLabel("0.80")
        
        self.pixel_threshold_slider = QSlider(Qt.Horizontal)
        self.pixel_threshold_slider.setRange(0, 100)
        self.pixel_threshold_slider.setValue(80)
        self.pixel_threshold_label = QLabel("0.80")
        
        anomaly_layout.addWidget(QLabel("Image Threshold:"), 0, 0)
        anomaly_layout.addWidget(self.image_threshold_slider, 0, 1)
        anomaly_layout.addWidget(self.image_threshold_label, 0, 2)
        
        anomaly_layout.addWidget(QLabel("Pixel Threshold:"), 1, 0)
        anomaly_layout.addWidget(self.pixel_threshold_slider, 1, 1)
        anomaly_layout.addWidget(self.pixel_threshold_label, 1, 2)
        
        # Volume of Interest controls
        voi_group = QGroupBox("Volume of Interest")
        voi_layout = QGridLayout(voi_group)
        
        self.voi_x_min = QDoubleSpinBox()
        self.voi_x_max = QDoubleSpinBox()
        self.voi_y_min = QDoubleSpinBox()
        self.voi_y_max = QDoubleSpinBox()
        self.voi_z_min = QDoubleSpinBox()
        self.voi_z_max = QDoubleSpinBox()
        
        for spinbox in [self.voi_x_min, self.voi_x_max, self.voi_y_min, 
                        self.voi_y_max, self.voi_z_min, self.voi_z_max]:
            spinbox.setRange(-5.0, 5.0)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)
            spinbox.setSuffix(" m")
        
        # Set default VOI
        self.voi_x_min.setValue(-1.0)
        self.voi_x_max.setValue(1.0)
        self.voi_y_min.setValue(-1.0)
        self.voi_y_max.setValue(1.0)
        self.voi_z_min.setValue(0.1)
        self.voi_z_max.setValue(2.0)
        
        voi_layout.addWidget(QLabel("X Min:"), 0, 0)
        voi_layout.addWidget(self.voi_x_min, 0, 1)
        voi_layout.addWidget(QLabel("X Max:"), 0, 2)
        voi_layout.addWidget(self.voi_x_max, 0, 3)
        
        voi_layout.addWidget(QLabel("Y Min:"), 1, 0)
        voi_layout.addWidget(self.voi_y_min, 1, 1)
        voi_layout.addWidget(QLabel("Y Max:"), 1, 2)
        voi_layout.addWidget(self.voi_y_max, 1, 3)
        
        voi_layout.addWidget(QLabel("Z Min:"), 2, 0)
        voi_layout.addWidget(self.voi_z_min, 2, 1)
        voi_layout.addWidget(QLabel("Z Max:"), 2, 2)
        voi_layout.addWidget(self.voi_z_max, 2, 3)
        
        # Anomaly size filtering
        self.min_anomaly_spin = QSpinBox()
        self.min_anomaly_spin.setRange(10, 10000)
        self.min_anomaly_spin.setValue(100)
        self.min_anomaly_spin.setSuffix(" voxels")
        
        self.max_anomaly_spin = QSpinBox()
        self.max_anomaly_spin.setRange(100, 100000)
        self.max_anomaly_spin.setValue(50000)
        self.max_anomaly_spin.setSuffix(" voxels")
        
        voi_layout.addWidget(QLabel("Min Anomaly:"), 3, 0)
        voi_layout.addWidget(self.min_anomaly_spin, 3, 1)
        voi_layout.addWidget(QLabel("Max Anomaly:"), 3, 2)
        voi_layout.addWidget(self.max_anomaly_spin, 3, 3)
        
        # Clear buttons
        clear_group = QGroupBox("Clear")
        clear_layout = QVBoxLayout(clear_group)
        
        self.clear_normal_btn = QPushButton("Clear Normal View")
        self.clear_anomaly_btn = QPushButton("Clear Anomaly View")
        self.clear_all_btn = QPushButton("Clear All")
        
        clear_layout.addWidget(self.clear_normal_btn)
        clear_layout.addWidget(self.clear_anomaly_btn)
        clear_layout.addWidget(self.clear_all_btn)
        
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
        layout.addWidget(rgb_group)
        layout.addWidget(viz_group)
        layout.addWidget(anomaly_group)
        layout.addWidget(voi_group)
        layout.addWidget(clear_group)
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
        
        # Status label for capture state
        self.capture_status_label = QLabel("Live Point Cloud Accumulation")
        self.capture_status_label.setAlignment(Qt.AlignCenter)
        self.capture_status_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.capture_status_label)
        
        # Split view
        splitter = QSplitter(Qt.Horizontal)
        
        self.normal_viewer = PyVistaViewer("Normal RGB Point Cloud")
        self.anomaly_viewer = PyVistaViewer("Anomaly Highlighted Point Cloud")
        
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
        
        # Clear buttons
        self.clear_normal_btn.clicked.connect(self.normal_viewer.clear_point_cloud)
        self.clear_anomaly_btn.clicked.connect(self.anomaly_viewer.clear_point_cloud)
        self.clear_all_btn.clicked.connect(self.clear_all_views)
        
        # Threshold sliders
        self.image_threshold_slider.valueChanged.connect(self.update_image_threshold)
        self.pixel_threshold_slider.valueChanged.connect(self.update_pixel_threshold)
        
        # Visualization controls
        self.voxel_size_spin.valueChanged.connect(self.update_voxel_size)
        self.grid_size_spin.valueChanged.connect(self.update_grid_size)
        
        # VOI controls
        self.voi_x_min.valueChanged.connect(self.update_voi)
        self.voi_x_max.valueChanged.connect(self.update_voi)
        self.voi_y_min.valueChanged.connect(self.update_voi)
        self.voi_y_max.valueChanged.connect(self.update_voi)
        self.voi_z_min.valueChanged.connect(self.update_voi)
        self.voi_z_max.valueChanged.connect(self.update_voi)
        
        # Anomaly size filters
        self.min_anomaly_spin.valueChanged.connect(self.update_anomaly_size_filters)
        self.max_anomaly_spin.valueChanged.connect(self.update_anomaly_size_filters)
        
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
            # Setup inference worker first
            self.setup_inference_worker()
            
            # Setup capture worker
            self.setup_capture_worker()
            
            # Start inference worker
            self.infer_thread.start()
            
            # Start capture worker
            self.capture_thread.start()
            
            # Update UI
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(True)
            self.status_label.setText("Capturing...")
            self.capture_status_label.setText("Live Point Cloud Accumulation")
            self.log("Started capture and inference")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start capture: {e}")
            self.log(f"Error starting capture: {e}")
    
    def stop_capture(self):
        """Stop capture and trigger processing"""
        try:
            # Only stop the capture worker, not the inference worker
            if self.capture_worker:
                self.capture_worker.stop_capture()
                
            self.start_capture_btn.setEnabled(True)
            self.stop_capture_btn.setEnabled(False)
            self.status_label.setText("Stopped - Processing...")
            self.log("Stopped capture")
            
            # Trigger final processing after stopping capture
            # The inference worker will stop itself after processing
            if self.infer_worker and self.infer_thread.isRunning():
                self.log("Waiting before triggering final anomaly processing...")
                # Give a small delay to ensure all frames are processed
                QTimer.singleShot(1000, self.trigger_final_processing)
            
        except Exception as e:
            self.log(f"Error stopping capture: {e}")
    
    def setup_capture_worker(self):
        """Setup capture worker and thread"""
        self.capture_thread = QThread()
        self.capture_worker = CaptureWorker(self.waypoints_config)
        self.capture_worker.moveToThread(self.capture_thread)
        
        # Connect signals
        self.capture_thread.started.connect(self.capture_worker.start_capture)
        self.capture_worker.progress.connect(self.progress_bar.setValue)
        self.capture_worker.error.connect(self.log)
        self.capture_worker.live_frame.connect(self.process_live_frame)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.on_capture_finished)
        
    def setup_inference_worker(self):
        """Setup inference worker and thread"""
        self.infer_thread = QThread()
        self.infer_worker = InferWorker(self.model_path)
        self.infer_worker.moveToThread(self.infer_thread)
        
        # Enable real-time mode
        self.infer_worker.enable_real_time_mode(True)
        
        # Set VOI and anomaly size filters
        self.update_voi()
        self.update_anomaly_size_filters()
        
        # Connect signals
        self.infer_thread.started.connect(self.infer_worker.start_inference)
        self.infer_worker.error.connect(self.log)
        self.infer_worker.live_inference_result.connect(self.update_3d_visualization)
        self.infer_worker.capture_completed.connect(self.on_capture_completed)
        self.infer_worker.finished.connect(self.infer_thread.quit)
        
        # Connect trigger for final processing
        self.trigger_processing.connect(self.infer_worker.trigger_final_processing)
        
        self.log("Inference worker signals connected")
    
    @Slot(object)
    def process_live_frame(self, frame_data):
        """Process live frame from capture worker"""
        # Update RGB feed
        self.update_rgb_feed(frame_data["rgb"])
        
        # Pass to inference worker
        if self.infer_worker:
            self.infer_worker.process_live_frame(frame_data)
    
    def update_rgb_feed(self, bgr_image):
        """Update live RGB feed display"""
        try:
            # Convert BGR to RGB for display
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.rgb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.rgb_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error updating RGB feed: {e}")
    
    @Slot(object)
    def update_3d_visualization(self, inference_result):
        """Update 3D visualization with inference results"""
        try:
            frame_data = inference_result["frame_data"]
            normalized_anomaly_map = inference_result["normalized_anomaly_map"]
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
            
            # Generate anomaly point cloud with full normalized anomaly map
            anomaly_cloud = self.pc_generator.create_anomaly_point_cloud(
                rgb_image, depth_image, camera_matrix, 
                normalized_anomaly_map,  # Use normalized map instead of binary mask
                transform_matrix
            )
            
            # Check if point clouds are valid
            if normal_cloud.n_points == 0:
                self.log("Warning: Empty normal point cloud generated")
                return
                
            # Add to viewers (maintain camera view)
            self.normal_viewer.add_point_cloud(normal_cloud, maintain_view=True)
            self.anomaly_viewer.add_point_cloud(anomaly_cloud, maintain_view=True)
            
            # Log progress
            waypoint_id = frame_data.get("waypoint_id", "unknown")
            normal_total = self.normal_viewer.point_count
            anomaly_total = self.anomaly_viewer.point_count
            
            self.log(f"Added {waypoint_id}: Normal={normal_total:,} pts, Anomaly={anomaly_total:,} pts")
            
            # Log anomaly detection
            if has_anomaly:
                score = inference_result["anomaly_score"]
                self.log(f"🚨 ANOMALY at {waypoint_id}: score={score:.3f}")
                
        except Exception as e:
            import traceback
            self.log(f"Error updating 3D visualization: {e}")
            logger.error(f"3D visualization error: {e}\n{traceback.format_exc()}")
    
    def update_image_threshold(self, value):
        """Update image threshold"""
        threshold = value / 100.0
        self.image_threshold_label.setText(f"{threshold:.2f}")
        if self.infer_worker:
            self.infer_worker.update_thresholds(image_threshold=threshold)
    
    def update_pixel_threshold(self, value):
        """Update pixel threshold"""
        threshold = value / 100.0
        self.pixel_threshold_label.setText(f"{threshold:.2f}")
        if self.infer_worker:
            self.infer_worker.update_thresholds(pixel_threshold=threshold)
            
    def update_voxel_size(self, value):
        """Update voxel size for both viewers"""
        self.normal_viewer.set_voxel_size(value)
        self.anomaly_viewer.set_voxel_size(value)
        self.log(f"Voxel size updated to {value}mm")
        
    def update_grid_size(self, value):
        """Update grid size for both viewers"""
        self.normal_viewer.set_grid_size(value)
        self.anomaly_viewer.set_grid_size(value)
        self.log(f"Grid size updated to {value}m")
    
    def update_voi(self):
        """Update volume of interest"""
        bounds = (
            self.voi_x_min.value(), self.voi_x_max.value(),
            self.voi_y_min.value(), self.voi_y_max.value(),
            self.voi_z_min.value(), self.voi_z_max.value()
        )
        if self.infer_worker:
            self.infer_worker.set_volume_of_interest(bounds)
            self.log(f"VOI updated: X[{bounds[0]:.2f}, {bounds[1]:.2f}], Y[{bounds[2]:.2f}, {bounds[3]:.2f}], Z[{bounds[4]:.2f}, {bounds[5]:.2f}]")
    
    def update_anomaly_size_filters(self):
        """Update anomaly size filters"""
        min_vol = self.min_anomaly_spin.value()
        max_vol = self.max_anomaly_spin.value()
        if self.infer_worker:
            self.infer_worker.set_anomaly_size_filters(min_vol, max_vol)
            self.log(f"Anomaly size filters updated: {min_vol} - {max_vol} voxels")
    
    @Slot(object)
    def on_capture_completed(self, accumulated_data):
        """Handle capture completion and process accumulated data"""
        self.log("=== CAPTURE COMPLETED SIGNAL RECEIVED ===")
        self.log("Capture completed, processing accumulated data...")
        
        try:
            num_frames = len(accumulated_data["anomaly_maps"])
            self.log(f"Processing {num_frames} frames for 3D anomaly detection...")
            
            # Process accumulated data to generate voxelized representation and bounding boxes
            self.process_final_3d_anomalies(accumulated_data)
            
        except Exception as e:
            self.log(f"Error in post-capture processing: {e}")
            logger.error(traceback.format_exc())
    
    def process_final_3d_anomalies(self, accumulated_data):
        """Process accumulated data to create voxelized representation and detect 3D anomaly regions"""
        
        # Extract accumulated data
        anomaly_maps = accumulated_data["normalized_maps"]
        rgb_images = accumulated_data["rgb_images"]
        depth_images = accumulated_data["depth_images"]
        transforms = accumulated_data["transforms"]
        waypoint_ids = accumulated_data["waypoint_ids"]
        frame_data_list = accumulated_data["frame_data"]
        pixel_threshold = accumulated_data["pixel_threshold"]
        voi_bounds = accumulated_data["voi_bounds"]
        min_anomaly_vol = accumulated_data["min_anomaly_volume"]
        max_anomaly_vol = accumulated_data["max_anomaly_volume"]
        
        self.log("Creating combined point clouds...")
        
        # Create combined point clouds
        all_normal_points = []
        all_anomaly_points = []
        all_anomaly_scores = []
        
        # Process each frame
        for i, (amap, rgb, depth, transform, frame_data) in enumerate(
            zip(anomaly_maps, rgb_images, depth_images, transforms, frame_data_list)
        ):
            camera_matrix = frame_data["camera_matrix"]
            
            # Generate normal point cloud
            normal_cloud = self.pc_generator.create_point_cloud(
                rgb, depth, camera_matrix, transform
            )
            
            # Generate anomaly point cloud with scores
            anomaly_cloud = self.pc_generator.create_anomaly_point_cloud(
                rgb, depth, camera_matrix, amap, transform
            )
            
            if normal_cloud.n_points > 0:
                all_normal_points.append(normal_cloud.points)
            
            if anomaly_cloud.n_points > 0:
                all_anomaly_points.append(anomaly_cloud.points)
                all_anomaly_scores.append(anomaly_cloud["Anomaly_Score"])
        
        # Combine all points
        if all_normal_points:
            combined_normal_points = np.vstack(all_normal_points)
        else:
            combined_normal_points = np.array([])
            
        if all_anomaly_points:
            combined_anomaly_points = np.vstack(all_anomaly_points)
            combined_anomaly_scores = np.hstack(all_anomaly_scores)
        else:
            combined_anomaly_points = np.array([])
            combined_anomaly_scores = np.array([])
        
        self.log(f"Combined point clouds: Normal={len(combined_normal_points):,}, Anomaly={len(combined_anomaly_points):,}")
        
        # Voxelize the point clouds
        voxel_size = self.voxel_size_spin.value() / 1000.0  # Convert mm to meters
        
        self.log(f"Voxelizing with voxel size: {voxel_size*1000:.1f}mm")
        
        # Create voxelized representations
        normal_voxel_cloud = self.voxelize_points(combined_normal_points, voxel_size)
        anomaly_voxel_cloud = self.voxelize_anomaly_points(
            combined_anomaly_points, combined_anomaly_scores, voxel_size, pixel_threshold
        )
        
        # Detect 3D anomaly regions (bounding boxes)
        self.log("Detecting 3D anomaly regions...")
        bounding_boxes = self.detect_3d_anomaly_regions(
            anomaly_voxel_cloud, voxel_size, voi_bounds, min_anomaly_vol, max_anomaly_vol
        )
        
        self.log(f"Found {len(bounding_boxes)} anomaly regions")
        
        # Associate RGB images with bounding boxes
        self.associate_rgb_images(bounding_boxes, rgb_images, depth_images, 
                                 transforms, waypoint_ids, frame_data_list)
        
        # Update viewers with voxelized data and bounding boxes
        self.update_final_visualization(normal_voxel_cloud, anomaly_voxel_cloud, bounding_boxes)
    
    def voxelize_points(self, points, voxel_size):
        """Voxelize point cloud"""
        if len(points) == 0:
            return pv.PolyData()
        
        # Create PyVista point cloud
        cloud = pv.PolyData(points)
        
        # Voxelize using PyVista
        # Create a uniform grid that encompasses all points
        bounds = cloud.bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Calculate grid dimensions
        nx = int(np.ceil((x_max - x_min) / voxel_size))
        ny = int(np.ceil((y_max - y_min) / voxel_size))
        nz = int(np.ceil((z_max - z_min) / voxel_size))
        
        # Create voxel grid
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(voxel_size, voxel_size, voxel_size),
            origin=(x_min, y_min, z_min)
        )
        
        # Map points to voxels
        voxel_centers = []
        
        # Get voxel indices for each point
        voxel_indices = np.floor((points - [x_min, y_min, z_min]) / voxel_size).astype(int)
        
        # Get unique voxels
        unique_voxels = np.unique(voxel_indices, axis=0)
        
        # Convert back to world coordinates (voxel centers)
        for voxel_idx in unique_voxels:
            center = [x_min, y_min, z_min] + (voxel_idx + 0.5) * voxel_size
            voxel_centers.append(center)
        
        if voxel_centers:
            voxel_cloud = pv.PolyData(np.array(voxel_centers))
            
            # Create cube glyphs for visualization
            cube = pv.Cube(center=(0, 0, 0), x_length=voxel_size, 
                          y_length=voxel_size, z_length=voxel_size)
            voxel_mesh = voxel_cloud.glyph(geom=cube)
            
            return voxel_mesh
        else:
            return pv.PolyData()
    
    def voxelize_anomaly_points(self, points, scores, voxel_size, threshold):
        """Voxelize anomaly points with scores"""
        if len(points) == 0:
            return pv.PolyData()
        
        # Filter by threshold
        mask = scores > threshold
        filtered_points = points[mask]
        filtered_scores = scores[mask]
        
        if len(filtered_points) == 0:
            return pv.PolyData()
        
        # Create PyVista point cloud
        cloud = pv.PolyData(filtered_points)
        
        # Get bounds
        bounds = cloud.bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Calculate voxel indices
        voxel_indices = np.floor((filtered_points - [x_min, y_min, z_min]) / voxel_size).astype(int)
        
        # Aggregate scores by voxel (take maximum score per voxel)
        voxel_dict = {}
        for i, (idx, score) in enumerate(zip(voxel_indices, filtered_scores)):
            key = tuple(idx)
            if key not in voxel_dict or score > voxel_dict[key]:
                voxel_dict[key] = score
        
        # Create voxel centers and scores
        voxel_centers = []
        voxel_scores = []
        
        for voxel_idx, score in voxel_dict.items():
            center = [x_min, y_min, z_min] + (np.array(voxel_idx) + 0.5) * voxel_size
            voxel_centers.append(center)
            voxel_scores.append(score)
        
        if voxel_centers:
            voxel_cloud = pv.PolyData(np.array(voxel_centers))
            voxel_cloud["Anomaly_Score"] = np.array(voxel_scores)
            
            # Create cube glyphs
            cube = pv.Cube(center=(0, 0, 0), x_length=voxel_size, 
                          y_length=voxel_size, z_length=voxel_size)
            voxel_mesh = voxel_cloud.glyph(geom=cube)
            
            # Transfer scores to mesh
            voxel_mesh["Anomaly_Score"] = np.repeat(voxel_scores, cube.n_points)
            
            return voxel_mesh
        else:
            return pv.PolyData()
    
    def detect_3d_anomaly_regions(self, anomaly_voxel_cloud, voxel_size, 
                                  voi_bounds, min_vol, max_vol):
        """Detect 3D bounding boxes for anomaly regions"""
        bounding_boxes = []
        
        if anomaly_voxel_cloud.n_points == 0:
            return bounding_boxes
        
        # Get voxel centers (every nth point where n is points per voxel)
        # For a cube, there are 8 vertices
        points_per_voxel = 8
        voxel_centers = anomaly_voxel_cloud.points[::points_per_voxel]
        anomaly_scores = anomaly_voxel_cloud["Anomaly_Score"][::points_per_voxel]
        
        # Filter by VOI if provided
        if voi_bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = voi_bounds
            voi_mask = (
                (voxel_centers[:, 0] >= x_min) & (voxel_centers[:, 0] <= x_max) &
                (voxel_centers[:, 1] >= y_min) & (voxel_centers[:, 1] <= y_max) &
                (voxel_centers[:, 2] >= z_min) & (voxel_centers[:, 2] <= z_max)
            )
            voxel_centers = voxel_centers[voi_mask]
            anomaly_scores = anomaly_scores[voi_mask]
        
        if len(voxel_centers) == 0:
            return bounding_boxes
        
        # Use DBSCAN clustering to find connected anomaly regions
        # Distance threshold is slightly larger than voxel diagonal
        eps = voxel_size * np.sqrt(3) * 1.1
        
        if HAS_SKLEARN:
            clustering = DBSCAN(eps=eps, min_samples=1).fit(voxel_centers)
            labels = clustering.labels_
        else:
            # Fallback: treat all voxels as one cluster
            self.log("Warning: sklearn not available, treating all anomalies as one region")
            labels = np.zeros(len(voxel_centers), dtype=int)
        
        # Process each cluster
        unique_labels = np.unique(labels[labels >= 0])
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = voxel_centers[cluster_mask]
            cluster_scores = anomaly_scores[cluster_mask]
            
            # Check volume constraints
            num_voxels = len(cluster_points)
            if num_voxels < min_vol or num_voxels > max_vol:
                continue
            
            # Calculate bounding box
            bbox_min = np.min(cluster_points, axis=0) - voxel_size/2
            bbox_max = np.max(cluster_points, axis=0) + voxel_size/2
            center = (bbox_min + bbox_max) / 2
            dimensions = bbox_max - bbox_min
            
            # Calculate average anomaly score
            avg_score = np.mean(cluster_scores)
            max_score = np.max(cluster_scores)
            
            bbox_info = {
                "id": len(bounding_boxes),
                "center": center,
                "dimensions": dimensions,
                "min": bbox_min,
                "max": bbox_max,
                "num_voxels": num_voxels,
                "avg_score": avg_score,
                "max_score": max_score,
                "points": cluster_points,
                "rgb_image": None  # Will be filled by associate_rgb_images
            }
            
            bounding_boxes.append(bbox_info)
        
        # Sort by average score (highest first)
        bounding_boxes.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return bounding_boxes
    
    def associate_rgb_images(self, bounding_boxes, rgb_images, depth_images, 
                            transforms, waypoint_ids, frame_data_list):
        """Associate each bounding box with the best RGB image"""
        
        for bbox in bounding_boxes:
            best_image_idx = None
            best_visibility_score = 0
            bbox_center = bbox["center"]
            
            # Check visibility from each camera position
            for i, (transform, depth, frame_data) in enumerate(
                zip(transforms, depth_images, frame_data_list)
            ):
                # Transform bbox center to camera coordinates
                bbox_center_homo = np.append(bbox_center, 1)
                transform_inv = np.linalg.inv(transform)
                bbox_cam = transform_inv @ bbox_center_homo
                bbox_cam = bbox_cam[:3]
                
                # Check if point is in front of camera
                if bbox_cam[2] <= 0:
                    continue
                
                # Project to image coordinates
                camera_matrix = frame_data["camera_matrix"]
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                
                u = int(fx * bbox_cam[0] / bbox_cam[2] + cx)
                v = int(fy * bbox_cam[1] / bbox_cam[2] + cy)
                
                # Check if within image bounds
                h, w = depth.shape
                if 0 <= u < w and 0 <= v < h:
                    # Check depth consistency
                    expected_depth = bbox_cam[2]
                    actual_depth = depth[v, u]
                    
                    if actual_depth > 0 and abs(actual_depth - expected_depth) < 0.1:
                        # Calculate visibility score (inverse of distance)
                        visibility_score = 1.0 / expected_depth
                        
                        if visibility_score > best_visibility_score:
                            best_visibility_score = visibility_score
                            best_image_idx = i
            
            # Assign best image
            if best_image_idx is not None:
                bbox["rgb_image"] = waypoint_ids[best_image_idx]
                bbox["rgb_image_path"] = f"wp{waypoint_ids[best_image_idx]}_rgb.png"
                self.log(f"Anomaly bbox {bbox['id']} associated with image: {bbox['rgb_image_path']}")
    
    def update_final_visualization(self, normal_voxel_cloud, anomaly_voxel_cloud, bounding_boxes):
        """Update viewers with voxelized data and bounding boxes"""
        self.log("Updating visualization with voxelized data...")
        
        # Clear current point clouds
        self.normal_viewer.clear_point_cloud()
        self.anomaly_viewer.clear_point_cloud()
        
        # Add voxelized normal cloud
        if normal_voxel_cloud.n_points > 0:
            self.normal_viewer.plotter.add_mesh(
                normal_voxel_cloud, 
                color='lightblue', 
                opacity=0.8,
                name="normal_voxels"
            )
            self.normal_viewer.info_label.setText(f"Voxels: {normal_voxel_cloud.n_cells:,}")
        
        # Add voxelized anomaly cloud with color mapping
        if anomaly_voxel_cloud.n_points > 0:
            self.anomaly_viewer.plotter.add_mesh(
                anomaly_voxel_cloud,
                scalars="Anomaly_Score",
                cmap='jet',
                opacity=0.9,
                name="anomaly_voxels"
            )
            self.anomaly_viewer.info_label.setText(f"Anomaly Voxels: {anomaly_voxel_cloud.n_cells:,}")
        
        # Add bounding boxes to both viewers
        for bbox in bounding_boxes:
            # Create box mesh
            box = pv.Box(bounds=[
                bbox["min"][0], bbox["max"][0],
                bbox["min"][1], bbox["max"][1],
                bbox["min"][2], bbox["max"][2]
            ])
            
            # Add to normal viewer (green boxes)
            self.normal_viewer.plotter.add_mesh(
                box, 
                color='green', 
                style='wireframe',
                line_width=3,
                name=f"bbox_normal_{bbox['id']}"
            )
            
            # Add to anomaly viewer (red boxes with opacity based on score)
            opacity = 0.3 + 0.7 * bbox["avg_score"]  # Scale opacity by score
            self.anomaly_viewer.plotter.add_mesh(
                box,
                color='red',
                style='surface',
                opacity=opacity,
                name=f"bbox_anomaly_{bbox['id']}"
            )
            
            # Add text label
            label = f"A{bbox['id']}: {bbox['avg_score']:.2f}"
            self.anomaly_viewer.plotter.add_text(
                label,
                position=bbox["center"],
                font_size=8,
                name=f"label_{bbox['id']}"
            )
        
        # Log summary
        self.log(f"Visualization complete: {len(bounding_boxes)} anomaly regions detected")
        for bbox in bounding_boxes:
            self.log(f"  - Region {bbox['id']}: {bbox['num_voxels']} voxels, "
                    f"score={bbox['avg_score']:.3f}, image={bbox.get('rgb_image_path', 'None')}")
        
        # Update status
        self.capture_status_label.setText(f"Voxelized View - {len(bounding_boxes)} Anomaly Regions Detected")
        
        # Render
        self.normal_viewer.plotter.render()
        self.anomaly_viewer.plotter.render()
        
        # Clean up inference worker
        if self.infer_worker and self.infer_thread.isRunning():
            self.log("Stopping inference worker...")
            self.infer_worker.stop_inference()
    
    def clear_all_views(self):
        """Clear all views"""
        self.normal_viewer.clear_point_cloud()
        self.anomaly_viewer.clear_point_cloud()
        self.log("Cleared all views")
    
    def on_capture_finished(self):
        """Handle capture finished"""
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.status_label.setText("Capture completed")
        self.capture_status_label.setText("Processing for voxelization...")
        self.log("Capture completed")
        
        # Trigger post-processing when capture finishes naturally
        # Add a small delay to ensure all frames have been processed
        if self.infer_worker and self.infer_thread.isRunning():
            self.log("Waiting for all frames to be processed...")
            # Use QTimer to delay the trigger
            QTimer.singleShot(1000, self.trigger_final_processing)
        else:
            self.log("Warning: Inference worker not ready for processing")
    
    def trigger_final_processing(self):
        """Trigger final processing of accumulated data"""
        if self.infer_worker and self.infer_thread.isRunning():
            self.log("Triggering anomaly processing...")
            self.trigger_processing.emit()
        else:
            self.log("Warning: Inference worker not available")
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        try:
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