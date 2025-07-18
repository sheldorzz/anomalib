import sys
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from datetime import datetime

# PyVista for 3D visualization
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

# Import workers
from capture_core import CaptureWorker
from infer_core import InferWorker

# Try sklearn for clustering
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Will use simple clustering.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudGenerator:
    """Generate point clouds from RGB-D data"""
    
    def __init__(self):
        self.max_depth = 3.0
        self.min_depth = 0.1
        
    def create_point_cloud(self, rgb_image, depth_image, camera_matrix, transform_matrix=None):
        """Create point cloud from RGB-D image"""
        height, width = depth_image.shape
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        valid_depth = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        z_valid = depth_image[valid_depth]
        
        x_3d = (u_valid - cx) * z_valid / fx
        y_3d = (v_valid - cy) * z_valid / fy
        z_3d = z_valid
        
        points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        if transform_matrix is not None:
            points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
            points_transformed = (transform_matrix @ points_homo.T).T
            points_3d = points_transformed[:, :3]
        
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) if rgb_image.shape[2] == 3 else rgb_image
        rgb_valid = rgb[valid_depth]
        
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            cloud["RGB"] = rgb_valid
            return cloud
        else:
            return pv.PolyData()
    
    def create_anomaly_point_cloud(self, rgb_image, depth_image, camera_matrix, 
                                   anomaly_map, transform_matrix=None):
        """Create point cloud colored by anomaly scores"""
        height, width = depth_image.shape
        
        # Ensure anomaly map is 2D
        if anomaly_map.ndim > 2:
            anomaly_map = anomaly_map.squeeze()
        
        # Always resize anomaly map to match depth image dimensions
        if anomaly_map.shape != (height, width):
            try:
                anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (width, height), 
                                        interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.error(f"Failed to resize anomaly map: {e}")
                # Return empty cloud on error
                return pv.PolyData()
        else:
            anomaly_map = anomaly_map.astype(np.float32)
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        valid_depth = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        z_valid = depth_image[valid_depth]
        anomaly_scores = anomaly_map[valid_depth]
        
        x_3d = (u_valid - cx) * z_valid / fx
        y_3d = (v_valid - cy) * z_valid / fy
        z_3d = z_valid
        
        points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        if transform_matrix is not None:
            points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
            points_transformed = (transform_matrix @ points_homo.T).T
            points_3d = points_transformed[:, :3]
        
        if len(points_3d) > 0:
            cloud = pv.PolyData(points_3d)
            
            # Color by anomaly score
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')
            colors = cmap(anomaly_scores)[:, :3]
            colors = (colors * 255).astype(np.uint8)
            
            cloud["RGB"] = colors
            cloud["Anomaly_Score"] = anomaly_scores
            
            return cloud
        else:
            return pv.PolyData()


class PyVistaViewer(QWidget):
    """PyVista 3D viewer widget"""
    
    def __init__(self, title="3D View"):
        super().__init__()
        self.title = title
        self.current_cloud = pv.PolyData()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.info_label = QLabel("No data")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        self.plotter = QtInteractor(self)
        self.plotter.setMinimumSize(400, 300)
        layout.addWidget(self.plotter)
        
        self.plotter.set_background([0.1, 0.1, 0.1])
        self.plotter.add_axes_at_origin(labels_off=False)
        
    def add_point_cloud(self, cloud, maintain_view=True):
        """Add point cloud to current accumulation"""
        try:
            camera_position = None
            if maintain_view and hasattr(self.plotter, 'camera_position'):
                camera_position = self.plotter.camera_position
            
            if self.current_cloud.n_points > 0:
                self.current_cloud = self.current_cloud + cloud
            else:
                self.current_cloud = cloud.copy()
            
            # Downsample if needed
            if self.current_cloud.n_points > 100000:
                indices = np.random.choice(self.current_cloud.n_points, 100000, replace=False)
                downsampled = pv.PolyData(self.current_cloud.points[indices])
                for array_name in self.current_cloud.point_data.keys():
                    downsampled[array_name] = self.current_cloud[array_name][indices]
                self.current_cloud = downsampled
            
            self.plotter.add_mesh(self.current_cloud, scalars="RGB", rgb=True, 
                                point_size=3, name="point_cloud", render_points_as_spheres=True)
            
            if maintain_view and camera_position is not None:
                self.plotter.camera_position = camera_position
            
            self.info_label.setText(f"Points: {self.current_cloud.n_points:,}")
                
        except Exception as e:
            logger.error(f"Failed to add point cloud: {e}")
            
    def clear(self):
        """Clear all meshes"""
        self.current_cloud = pv.PolyData()
        self.plotter.clear()
        self.plotter.add_axes_at_origin(labels_off=False)
        self.info_label.setText("Cleared")
        self.plotter.render()
        
    def show_voxelized(self, voxel_mesh, color='lightblue', opacity=0.8):
        """Show voxelized mesh"""
        self.clear()
        if voxel_mesh.n_points > 0:
            self.plotter.add_mesh(voxel_mesh, color=color, opacity=opacity, name="voxels")
            self.info_label.setText(f"Voxels: {voxel_mesh.n_cells:,}")
            self.plotter.reset_camera()
            self.plotter.render()
    
    def show_anomaly_voxels(self, voxel_mesh):
        """Show anomaly voxels with color mapping"""
        self.clear()
        if voxel_mesh.n_points > 0:
            self.plotter.add_mesh(voxel_mesh, scalars="Anomaly_Score", cmap='jet', 
                                opacity=0.9, name="anomaly_voxels")
            self.info_label.setText(f"Anomaly Voxels: {voxel_mesh.n_cells:,}")
            self.plotter.reset_camera()
            self.plotter.render()
    
    def add_bounding_box(self, bbox_id, bounds, color='red', opacity=0.5):
        """Add a bounding box"""
        box = pv.Box(bounds=bounds)
        self.plotter.add_mesh(box, color=color, opacity=opacity, 
                            name=f"bbox_{bbox_id}")
    
    def add_bbox_label(self, bbox_id, text, position):
        """Add text label for bounding box"""
        self.plotter.add_text(text, position=position, font_size=8, 
                            name=f"label_{bbox_id}")


class RobotVision3DApp(QMainWindow):
    """Main application"""
    
    trigger_processing = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Vision 3D - Anomaly Detection")
        self.setGeometry(100, 100, 1400, 800)
        
        self.capture_worker = None
        self.capture_thread = None
        self.infer_worker = None
        self.infer_thread = None
        
        self.pc_generator = PointCloudGenerator()
        
        self.waypoints_config = None
        self.model_path = "models/weights/torch/patchcore.pt"
        self.voxel_size = 0.005  # 5mm
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Visualization panel
        visualization_panel = self.create_visualization_panel()
        main_layout.addWidget(visualization_panel, 3)
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Configuration
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
        self.rgb_label.setScaledContents(True)
        self.rgb_label.setStyleSheet("QLabel { background-color: black; }")
        self.rgb_label.setText("No feed")
        
        rgb_layout.addWidget(self.rgb_label)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QGridLayout(viz_group)
        
        self.voxel_size_spin = QSpinBox()
        self.voxel_size_spin.setRange(1, 20)
        self.voxel_size_spin.setValue(5)
        self.voxel_size_spin.setSuffix(" mm")
        
        self.pixel_threshold_slider = QSlider(Qt.Horizontal)
        self.pixel_threshold_slider.setRange(0, 100)
        self.pixel_threshold_slider.setValue(80)
        self.pixel_threshold_label = QLabel("0.80")
        
        viz_layout.addWidget(QLabel("Voxel Size:"), 0, 0)
        viz_layout.addWidget(self.voxel_size_spin, 0, 1)
        viz_layout.addWidget(QLabel("Threshold:"), 1, 0)
        viz_layout.addWidget(self.pixel_threshold_slider, 1, 1)
        viz_layout.addWidget(self.pixel_threshold_label, 1, 2)
        
        # VOI controls
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
        
        self.voi_x_min.setValue(-1.0)
        self.voi_x_max.setValue(1.0)
        self.voi_y_min.setValue(-1.0)
        self.voi_y_max.setValue(1.0)
        self.voi_z_min.setValue(0.1)
        self.voi_z_max.setValue(2.0)
        
        voi_layout.addWidget(QLabel("X:"), 0, 0)
        voi_layout.addWidget(self.voi_x_min, 0, 1)
        voi_layout.addWidget(self.voi_x_max, 0, 2)
        voi_layout.addWidget(QLabel("Y:"), 1, 0)
        voi_layout.addWidget(self.voi_y_min, 1, 1)
        voi_layout.addWidget(self.voi_y_max, 1, 2)
        voi_layout.addWidget(QLabel("Z:"), 2, 0)
        voi_layout.addWidget(self.voi_z_min, 2, 1)
        voi_layout.addWidget(self.voi_z_max, 2, 2)
        
        # Anomaly size filters
        self.min_anomaly_spin = QSpinBox()
        self.min_anomaly_spin.setRange(10, 10000)
        self.min_anomaly_spin.setValue(100)
        self.max_anomaly_spin = QSpinBox()
        self.max_anomaly_spin.setRange(100, 100000)
        self.max_anomaly_spin.setValue(50000)
        
        voi_layout.addWidget(QLabel("Min Size:"), 3, 0)
        voi_layout.addWidget(self.min_anomaly_spin, 3, 1)
        voi_layout.addWidget(QLabel("Max Size:"), 3, 2)
        voi_layout.addWidget(self.max_anomaly_spin, 3, 3)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add all groups
        layout.addWidget(file_group)
        layout.addWidget(capture_group)
        layout.addWidget(rgb_group)
        layout.addWidget(viz_group)
        layout.addWidget(voi_group)
        layout.addWidget(log_group)
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        title = QLabel("3D Point Cloud Visualization")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        self.capture_status_label = QLabel("Live Point Cloud")
        self.capture_status_label.setAlignment(Qt.AlignCenter)
        self.capture_status_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.capture_status_label)
        
        splitter = QSplitter(Qt.Horizontal)
        
        self.normal_viewer = PyVistaViewer("Normal View")
        self.anomaly_viewer = PyVistaViewer("Anomaly View")
        
        splitter.addWidget(self.normal_viewer)
        splitter.addWidget(self.anomaly_viewer)
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter)
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections"""
        self.waypoints_btn.clicked.connect(self.load_waypoints_config)
        self.model_btn.clicked.connect(self.load_model)
        self.start_capture_btn.clicked.connect(self.start_capture)
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        
        self.voxel_size_spin.valueChanged.connect(self.update_voxel_size)
        self.pixel_threshold_slider.valueChanged.connect(self.update_pixel_threshold)
        
        self.voi_x_min.valueChanged.connect(self.update_voi)
        self.voi_x_max.valueChanged.connect(self.update_voi)
        self.voi_y_min.valueChanged.connect(self.update_voi)
        self.voi_y_max.valueChanged.connect(self.update_voi)
        self.voi_z_min.valueChanged.connect(self.update_voi)
        self.voi_z_max.valueChanged.connect(self.update_voi)
        
        self.min_anomaly_spin.valueChanged.connect(self.update_anomaly_filters)
        self.max_anomaly_spin.valueChanged.connect(self.update_anomaly_filters)
        
    def load_waypoints_config(self):
        """Load waypoints configuration"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Waypoints Config", "", "JSON Files (*.json)"
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
            self, "Load Model", "", "PyTorch Files (*.pt)"
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
            self.setup_inference_worker()
            self.setup_capture_worker()
            
            self.infer_thread.start()
            self.capture_thread.start()
            
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(True)
            self.status_label.setText("Capturing...")
            self.capture_status_label.setText("Live Point Cloud")
            self.log("Started capture and inference")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start capture: {e}")
            self.log(f"Error: {e}")
    
    def stop_capture(self):
        """Stop capture"""
        if self.capture_worker:
            self.capture_worker.stop_capture()
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        self.log("Stopped capture, processing...")
        
        # Trigger final processing after delay
        QTimer.singleShot(1000, self.trigger_final_processing)
    
    def trigger_final_processing(self):
        """Trigger final processing"""
        if self.infer_worker:
            self.log("Triggering final processing...")
            self.trigger_processing.emit()
    
    def setup_capture_worker(self):
        """Setup capture worker"""
        self.capture_thread = QThread()
        self.capture_worker = CaptureWorker(self.waypoints_config)
        self.capture_worker.moveToThread(self.capture_thread)
        
        self.capture_thread.started.connect(self.capture_worker.start_capture)
        self.capture_worker.progress.connect(self.progress_bar.setValue)
        self.capture_worker.error.connect(self.log)
        self.capture_worker.live_frame.connect(self.process_live_frame)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.on_capture_finished)
        
    def setup_inference_worker(self):
        """Setup inference worker"""
        self.infer_thread = QThread()
        self.infer_worker = InferWorker(self.model_path)
        self.infer_worker.moveToThread(self.infer_thread)
        
        self.update_voi()
        self.update_anomaly_filters()
        
        self.infer_thread.started.connect(self.infer_worker.start_inference)
        self.infer_worker.error.connect(self.log)
        self.infer_worker.live_inference_result.connect(self.update_live_visualization)
        self.infer_worker.capture_completed.connect(self.process_final_data)
        self.infer_worker.finished.connect(self.infer_thread.quit)
        
        self.trigger_processing.connect(self.infer_worker.trigger_final_processing)
    
    @Slot(object)
    def process_live_frame(self, frame_data):
        """Process live frame"""
        # Update RGB feed
        bgr = frame_data["rgb"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.rgb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.rgb_label.setPixmap(scaled)
        
        # Pass to inference
        if self.infer_worker:
            self.infer_worker.process_live_frame(frame_data)
    
    @Slot(object)
    def update_live_visualization(self, result):
        """Update live point cloud visualization"""
        try:
            frame_data = result["frame_data"]
            normalized_map = result["normalized_anomaly_map"]
            
            # Generate point clouds
            normal_cloud = self.pc_generator.create_point_cloud(
                frame_data["rgb"], frame_data["depth"], 
                frame_data["camera_matrix"], frame_data["transform_matrix"]
            )
            
            anomaly_cloud = self.pc_generator.create_anomaly_point_cloud(
                frame_data["rgb"], frame_data["depth"], 
                frame_data["camera_matrix"], normalized_map, 
                frame_data["transform_matrix"]
            )
            
            # Add to viewers
            if normal_cloud.n_points > 0:
                self.normal_viewer.add_point_cloud(normal_cloud)
            if anomaly_cloud.n_points > 0:
                self.anomaly_viewer.add_point_cloud(anomaly_cloud)
                
        except Exception as e:
            self.log(f"Visualization error: {e}")
    
    @Slot(object)
    def process_final_data(self, data):
        """Process final accumulated data"""
        self.log("Processing final data...")
        self.capture_status_label.setText("Voxelizing...")
        
        # Check if we have data
        if not data or len(data.get('anomaly_maps', [])) == 0:
            self.log("No data to process!")
            self.status_label.setText("No data")
            return
        
        try:
            # Log data info
            self.log(f"Processing {len(data['anomaly_maps'])} frames")
            if len(data['anomaly_maps']) > 0:
                self.log(f"Anomaly map size: {data['anomaly_maps'][0].shape}")
                self.log(f"Depth image size: {data['depth_images'][0].shape}")
            
            # Create combined point clouds
            all_normal_points = []
            all_anomaly_points = []
            all_anomaly_scores = []
            
            # Combine anomaly maps with voting
            height, width = data["depth_images"][0].shape
            combined_anomaly_map = np.zeros((height, width), dtype=np.float32)
            
            # Vote on anomaly locations
            for i, (amap, rgb, depth, transform, cam_matrix) in enumerate(zip(
                data["anomaly_maps"], data["rgb_images"], data["depth_images"],
                data["transforms"], data["camera_matrices"]
            )):
                # Resize anomaly map to match depth image size
                try:
                    # Ensure anomaly map is 2D
                    if amap.ndim > 2:
                        amap = amap.squeeze()
                    amap_resized = cv2.resize(amap.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    self.log(f"Error resizing anomaly map: {e}, skipping frame {i}")
                    continue
                
                # Accumulate votes
                combined_anomaly_map += (amap_resized > data["pixel_threshold"]).astype(np.float32)
                
                # Create point clouds
                normal_cloud = self.pc_generator.create_point_cloud(
                    rgb, depth, cam_matrix, transform
                )
                
                if normal_cloud.n_points > 0:
                    all_normal_points.append(normal_cloud.points)
            
            # Normalize votes
            combined_anomaly_map /= len(data["anomaly_maps"])
            
            # Log voting results
            votes_above_threshold = np.sum(combined_anomaly_map > 0.3)
            total_pixels = combined_anomaly_map.size
            self.log(f"Voting complete: {votes_above_threshold:,}/{total_pixels:,} pixels "
                    f"({votes_above_threshold/total_pixels*100:.1f}%) voted as anomalous")
            
            # Now create anomaly point cloud from voted map
            vote_threshold = 0.3  # At least 30% of views should agree
            
            for i, (rgb, depth, transform, cam_matrix) in enumerate(zip(
                data["rgb_images"], data["depth_images"], 
                data["transforms"], data["camera_matrices"]
            )):
                # Use the combined voted anomaly map for all frames
                anomaly_cloud = self.pc_generator.create_anomaly_point_cloud(
                    rgb, depth, cam_matrix, combined_anomaly_map, transform
                )
                
                if anomaly_cloud.n_points > 0:
                    # Filter by vote threshold
                    mask = anomaly_cloud["Anomaly_Score"] > vote_threshold
                    if np.any(mask):
                        all_anomaly_points.append(anomaly_cloud.points[mask])
                        all_anomaly_scores.append(anomaly_cloud["Anomaly_Score"][mask])
            
            # Combine all points
            if all_normal_points:
                combined_normal = np.vstack(all_normal_points)
                self.log(f"Combined normal points shape: {combined_normal.shape}")
            else:
                combined_normal = np.array([])
                self.log("No normal points to combine")
                
            if all_anomaly_points:
                combined_anomaly = np.vstack(all_anomaly_points)
                combined_scores = np.hstack(all_anomaly_scores)
                self.log(f"Combined anomaly points shape: {combined_anomaly.shape}")
            else:
                combined_anomaly = np.array([])
                combined_scores = np.array([])
                self.log("No anomaly points to combine")
            
            self.log(f"Combined points: Normal={len(combined_normal):,}, Anomaly={len(combined_anomaly):,}")
            
            # Voxelize
            voxel_size = self.voxel_size_spin.value() / 1000.0
            self.log(f"Voxelizing with size {voxel_size*1000:.1f}mm")
            
            try:
                normal_voxels = self.voxelize_points(combined_normal, voxel_size)
                anomaly_voxels = self.voxelize_anomaly_points(combined_anomaly, combined_scores, voxel_size)
                
                self.log(f"Voxelized: Normal={normal_voxels.n_cells if normal_voxels.n_points > 0 else 0:,}, "
                        f"Anomaly={anomaly_voxels.n_cells if anomaly_voxels.n_points > 0 else 0:,}")
            except Exception as e:
                self.log(f"Voxelization error: {e}")
                import traceback
                traceback.print_exc()
                normal_voxels = pv.PolyData()
                anomaly_voxels = pv.PolyData()
            
            # Detect 3D anomaly regions
            try:
                bounding_boxes = self.detect_anomaly_regions(
                    anomaly_voxels, voxel_size, data["voi_bounds"], 
                    data["min_anomaly_volume"], data["max_anomaly_volume"]
                )
                
                self.log(f"Detected {len(bounding_boxes)} anomaly regions")
            except Exception as e:
                self.log(f"Error detecting anomaly regions: {e}")
                bounding_boxes = []
            
            # Save annotated images
            try:
                self.save_annotated_images(bounding_boxes, data)
            except Exception as e:
                self.log(f"Error saving annotated images: {e}")
            
            # Update visualization
            self.update_voxel_visualization(normal_voxels, anomaly_voxels, bounding_boxes)
            
        except Exception as e:
            self.log(f"Error processing final data: {e}")
            import traceback
            traceback.print_exc()
    
    def voxelize_points(self, points, voxel_size):
        """Voxelize point cloud"""
        if points is None or len(points) == 0 or points.size == 0:
            self.log("No points to voxelize")
            return pv.PolyData()
        
        # Ensure points is 2D array
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        
        # Get bounds
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        self.log(f"Point cloud bounds: [{min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f}] "
                f"to [{max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f}]")
        
        # Calculate voxel indices
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
        
        # Get unique voxels
        unique_voxels = np.unique(voxel_indices, axis=0)
        self.log(f"Created {len(unique_voxels)} unique voxels from {len(points)} points")
        
        # Create voxel centers
        voxel_centers = min_bound + (unique_voxels + 0.5) * voxel_size
        
        # Create voxel mesh
        voxel_cloud = pv.PolyData(voxel_centers)
        cube = pv.Cube(center=(0, 0, 0), x_length=voxel_size, 
                      y_length=voxel_size, z_length=voxel_size)
        voxel_mesh = voxel_cloud.glyph(geom=cube)
        
        return voxel_mesh
    
    def voxelize_anomaly_points(self, points, scores, voxel_size):
        """Voxelize anomaly points with scores"""
        if points is None or len(points) == 0 or points.size == 0:
            self.log("No anomaly points to voxelize")
            return pv.PolyData()
        
        # Ensure points is 2D array
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        
        # Get bounds
        min_bound = np.min(points, axis=0)
        
        # Calculate voxel indices
        voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
        
        # Aggregate scores by voxel
        voxel_dict = {}
        for idx, score in zip(voxel_indices, scores):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(score)
        
        # Create voxel centers and average scores
        voxel_centers = []
        voxel_scores = []
        
        for voxel_idx, scores_list in voxel_dict.items():
            center = min_bound + (np.array(voxel_idx) + 0.5) * voxel_size
            voxel_centers.append(center)
            voxel_scores.append(np.mean(scores_list))
        
        self.log(f"Created {len(voxel_centers)} anomaly voxels")
        
        if len(voxel_centers) == 0:
            return pv.PolyData()
        
        # Create voxel mesh
        voxel_cloud = pv.PolyData(np.array(voxel_centers))
        voxel_cloud["Anomaly_Score"] = np.array(voxel_scores)
        
        cube = pv.Cube(center=(0, 0, 0), x_length=voxel_size, 
                      y_length=voxel_size, z_length=voxel_size)
        voxel_mesh = voxel_cloud.glyph(geom=cube)
        voxel_mesh["Anomaly_Score"] = np.repeat(voxel_scores, cube.n_points)
        
        return voxel_mesh
    
    def detect_anomaly_regions(self, anomaly_voxels, voxel_size, voi_bounds, min_vol, max_vol):
        """Detect 3D anomaly regions"""
        if anomaly_voxels.n_points == 0:
            return []
        
        # Get voxel centers
        points_per_voxel = 8  # Cube has 8 vertices
        voxel_centers = anomaly_voxels.points[::points_per_voxel]
        voxel_scores = anomaly_voxels["Anomaly_Score"][::points_per_voxel]
        
        # Filter by VOI
        if voi_bounds:
            x_min, x_max, y_min, y_max, z_min, z_max = voi_bounds
            mask = (
                (voxel_centers[:, 0] >= x_min) & (voxel_centers[:, 0] <= x_max) &
                (voxel_centers[:, 1] >= y_min) & (voxel_centers[:, 1] <= y_max) &
                (voxel_centers[:, 2] >= z_min) & (voxel_centers[:, 2] <= z_max)
            )
            voxel_centers = voxel_centers[mask]
            voxel_scores = voxel_scores[mask]
        
        if len(voxel_centers) == 0:
            return []
        
        # Cluster voxels
        eps = voxel_size * np.sqrt(3) * 1.1
        
        if HAS_SKLEARN:
            clustering = DBSCAN(eps=eps, min_samples=1).fit(voxel_centers)
            labels = clustering.labels_
        else:
            # Simple clustering - all connected
            labels = np.zeros(len(voxel_centers), dtype=int)
        
        # Process clusters
        bounding_boxes = []
        unique_labels = np.unique(labels[labels >= 0])
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = voxel_centers[mask]
            cluster_scores = voxel_scores[mask]
            
            # Check size constraints
            num_voxels = len(cluster_points)
            if num_voxels < min_vol or num_voxels > max_vol:
                continue
            
            # Calculate bounding box
            bbox_min = np.min(cluster_points, axis=0) - voxel_size/2
            bbox_max = np.max(cluster_points, axis=0) + voxel_size/2
            center = (bbox_min + bbox_max) / 2
            
            bbox = {
                "id": len(bounding_boxes),
                "center": center,
                "min": bbox_min,
                "max": bbox_max,
                "num_voxels": num_voxels,
                "avg_score": np.mean(cluster_scores),
                "points": cluster_points
            }
            
            bounding_boxes.append(bbox)
        
        return bounding_boxes
    
    def save_annotated_images(self, bounding_boxes, data):
        """Save 2D annotated images for each anomaly"""
        session_dir = Path(data["session_dir"])
        session_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Saving annotated images to: {session_dir}")
        
        for bbox in bounding_boxes:
            best_img_idx = None
            best_score = 0
            bbox_center = bbox["center"]
            
            # Find best view for this anomaly
            for i, (transform, cam_matrix, depth) in enumerate(zip(
                data["transforms"], data["camera_matrices"], data["depth_images"]
            )):
                # Project bbox center to image
                bbox_homo = np.append(bbox_center, 1)
                cam_coord = np.linalg.inv(transform) @ bbox_homo
                
                if cam_coord[2] <= 0:
                    continue
                
                fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
                cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
                
                u = int(fx * cam_coord[0] / cam_coord[2] + cx)
                v = int(fy * cam_coord[1] / cam_coord[2] + cy)
                
                h, w = depth.shape
                if 0 <= u < w and 0 <= v < h:
                    # Check visibility
                    if depth[v, u] > 0 and abs(depth[v, u] - cam_coord[2]) < 0.1:
                        score = 1.0 / cam_coord[2]  # Closer is better
                        if score > best_score:
                            best_score = score
                            best_img_idx = i
            
            if best_img_idx is not None:
                # Get the best image
                img = data["rgb_images"][best_img_idx].copy()
                anomaly_map = data["anomaly_maps"][best_img_idx]
                
                # Draw anomaly overlay
                h, w = img.shape[:2]
                
                # Resize anomaly map to match image size
                # Ensure anomaly map is 2D
                if anomaly_map.ndim > 2:
                    anomaly_map = anomaly_map.squeeze()
                anomaly_map_resized = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Normalize to 0-255 range if needed
                if anomaly_map_resized.max() <= 1.0:
                    anomaly_map_uint8 = (anomaly_map_resized * 255).astype(np.uint8)
                else:
                    anomaly_map_uint8 = anomaly_map_resized.astype(np.uint8)
                
                overlay = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                
                # Save annotated image
                filename = session_dir / f"anomaly_{bbox['id']:03d}_score_{bbox['avg_score']:.3f}.png"
                cv2.imwrite(str(filename), img)
                
                self.log(f"Saved anomaly {bbox['id']} to {filename.name}")
        
        if len(bounding_boxes) > 0:
            self.log(f"All annotated images saved to: {session_dir}")
    
    def update_voxel_visualization(self, normal_voxels, anomaly_voxels, bounding_boxes):
        """Update visualization with voxels and bounding boxes"""
        self.capture_status_label.setText(f"Voxelized - {len(bounding_boxes)} Anomalies Detected")
        
        # Check if we have valid voxel meshes
        if normal_voxels is None or normal_voxels.n_points == 0:
            self.log("Warning: No normal voxels to display")
        else:
            self.log(f"Displaying {normal_voxels.n_cells} normal voxels")
            
        if anomaly_voxels is None or anomaly_voxels.n_points == 0:
            self.log("Warning: No anomaly voxels to display")
        else:
            self.log(f"Displaying {anomaly_voxels.n_cells} anomaly voxels")
        
        # Show voxelized clouds
        if normal_voxels is not None and normal_voxels.n_points > 0:
            self.normal_viewer.show_voxelized(normal_voxels)
        else:
            self.normal_viewer.clear()
            self.normal_viewer.info_label.setText("No normal voxels")
            self.normal_viewer.plotter.render()
        
        if anomaly_voxels is not None and anomaly_voxels.n_points > 0:
            self.anomaly_viewer.show_anomaly_voxels(anomaly_voxels)
        else:
            self.anomaly_viewer.clear()
            self.anomaly_viewer.info_label.setText("No anomaly voxels")
            self.anomaly_viewer.plotter.render()
        
        # Add bounding boxes
        for bbox in bounding_boxes:
            bounds = [
                bbox["min"][0], bbox["max"][0],
                bbox["min"][1], bbox["max"][1],
                bbox["min"][2], bbox["max"][2]
            ]
            
            # Add to both viewers
            self.normal_viewer.add_bounding_box(bbox["id"], bounds, color='green', opacity=0.3)
            self.anomaly_viewer.add_bounding_box(bbox["id"], bounds, color='red', 
                                                opacity=0.3 + 0.7 * bbox["avg_score"])
            
            # Add label
            label = f"A{bbox['id']}: {bbox['avg_score']:.2f}"
            self.anomaly_viewer.add_bbox_label(bbox["id"], label, bbox["center"])
        
        # Log bounding box info
        for bbox in bounding_boxes:
            self.log(f"Anomaly {bbox['id']}: {bbox['num_voxels']} voxels, score={bbox['avg_score']:.3f}")
        
        self.log("Visualization complete")
        self.status_label.setText("Complete")
        
        # Reset camera to show all content and render
        self.normal_viewer.plotter.reset_camera()
        self.anomaly_viewer.plotter.reset_camera()
        self.normal_viewer.plotter.render()
        self.anomaly_viewer.plotter.render()
    
    def on_capture_finished(self):
        """Handle capture finished"""
        self.capture_status_label.setText("Processing...")
        QTimer.singleShot(1000, self.trigger_final_processing)
    
    def update_voxel_size(self):
        self.voxel_size = self.voxel_size_spin.value() / 1000.0
        
    def update_pixel_threshold(self, value):
        threshold = value / 100.0
        self.pixel_threshold_label.setText(f"{threshold:.2f}")
        if self.infer_worker:
            self.infer_worker.update_thresholds(pixel_threshold=threshold)
            
    def update_voi(self):
        bounds = (
            self.voi_x_min.value(), self.voi_x_max.value(),
            self.voi_y_min.value(), self.voi_y_max.value(),
            self.voi_z_min.value(), self.voi_z_max.value()
        )
        if self.infer_worker:
            self.infer_worker.set_volume_of_interest(bounds)
            
    def update_anomaly_filters(self):
        if self.infer_worker:
            self.infer_worker.set_anomaly_size_filters(
                self.min_anomaly_spin.value(), 
                self.max_anomaly_spin.value()
            )
    
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
    app = QApplication(sys.argv)
    app.setApplicationName("Robot Vision 3D")
    
    try:
        window = RobotVision3DApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()