# test_split_view_3d.py
import sys
import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSplitter)
from PySide6.QtGui import QPixmap, QImage
import cv2

# Import the workers
from capture import CaptureWorker
from infer import InferWorker
from pc_rgb import RgbPCWorker
from pc_anom import AnomPCWorker


class PointCloudWidget(QLabel):
    """Widget for displaying point cloud using offscreen rendering"""
    
    def __init__(self, window_name="3D View"):
        super().__init__()
        self.window_name = window_name
        self.point_cloud = o3d.geometry.PointCloud()
        
        # Offscreen renderer
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
        self.setup_renderer()
        
        # Set initial black image
        self.setMinimumSize(640, 480)
        self.setMaximumSize(640, 480)
        self.setStyleSheet("border: 1px solid #333;")
        self.setScaledContents(True)
        
        # Camera parameters
        self.view_distance = 2.0
        self.camera_angle = 0
        self.camera_elevation = 30
        
        # Initial render
        self.render_point_cloud()
        
    def setup_renderer(self):
        """Setup the offscreen renderer"""
        # Set background color
        self.renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Setup lighting
        self.renderer.scene.scene.set_sun_light(
            [0.577, -0.577, -0.577],  # direction
            [1.0, 1.0, 1.0],  # color
            100000  # intensity
        )
        self.renderer.scene.scene.enable_sun_light(True)
        
        # Add ambient light
        self.renderer.scene.scene.set_indirect_light_intensity(30000)
        
    def update_point_cloud(self, data):
        """Update point cloud with new data"""
        points = data.get('points', np.array([]))
        colors = data.get('colors', np.array([]))
        
        if points.shape[0] > 0:
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            if colors.shape[0] == points.shape[0]:
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Default color if not provided
                default_colors = np.ones((points.shape[0], 3)) * 0.5
                self.point_cloud.colors = o3d.utility.Vector3dVector(default_colors)
            
            # Update visualization
            self.render_point_cloud()
            
    def render_point_cloud(self):
        """Render the point cloud to an image"""
        # Clear previous geometry
        self.renderer.scene.clear_geometry()
        
        if len(self.point_cloud.points) > 0:
            # Create material for point cloud
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 3.0
            
            # Add point cloud to scene
            self.renderer.scene.add_geometry("pointcloud", self.point_cloud, mat)
            
            # Calculate bounding box center
            bbox = self.point_cloud.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = max(extent)
            
            # Setup camera
            if max_extent > 0:
                # Calculate camera position
                theta = np.radians(self.camera_angle)
                phi = np.radians(self.camera_elevation)
                
                camera_distance = max_extent * self.view_distance
                
                eye = center + np.array([
                    camera_distance * np.cos(phi) * np.sin(theta),
                    camera_distance * np.cos(phi) * np.cos(theta),
                    camera_distance * np.sin(phi)
                ])
                
                # Setup camera
                self.renderer.setup_camera(
                    60,  # field of view
                    center,  # look at
                    eye,  # camera position
                    [0, 0, 1]  # up vector
                )
        
        # Render to image
        img = self.renderer.render_to_image()
        
        # Convert to QPixmap and display
        img_array = np.asarray(img)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(pixmap)
        
    def rotate_camera(self, angle_delta):
        """Rotate camera around the point cloud"""
        self.camera_angle += angle_delta
        self.render_point_cloud()
        
    def elevate_camera(self, elevation_delta):
        """Change camera elevation"""
        self.camera_elevation = np.clip(self.camera_elevation + elevation_delta, -89, 89)
        self.render_point_cloud()
        
    def zoom_camera(self, zoom_factor):
        """Zoom camera in/out"""
        self.view_distance = np.clip(self.view_distance * zoom_factor, 0.5, 5.0)
        self.render_point_cloud()
        
    def mousePressEvent(self, event):
        """Handle mouse press for camera control"""
        self.last_mouse_pos = event.pos()
        
    def mouseMoveEvent(self, event):
        """Handle mouse move for camera rotation"""
        if event.buttons() & Qt.LeftButton:
            delta = event.pos() - self.last_mouse_pos
            self.rotate_camera(delta.x() * 0.5)
            self.elevate_camera(-delta.y() * 0.5)
            self.last_mouse_pos = event.pos()
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom_camera(zoom_factor)


class SplitView3DTest(QMainWindow):
    """Main window with split view 3D visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Split View 3D Point Cloud Test")
        self.setGeometry(100, 100, 1400, 600)
        
        # Workers
        self.capture_worker = None
        self.infer_worker = None
        self.rgb_pc_worker = None
        self.anom_pc_worker = None
        
        # Threads
        self.capture_thread = QThread()
        self.infer_thread = QThread()
        self.rgb_pc_thread = QThread()
        self.anom_pc_thread = QThread()
        
        # Camera rotation timer
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.auto_rotate)
        self.auto_rotation_enabled = False
        
        # Setup UI
        self.setup_ui()
        
        # Setup workers
        self.setup_workers()
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Control panel
        control_panel = QHBoxLayout()
        
        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self.start_capture)
        control_panel.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        control_panel.addWidget(self.stop_button)
        
        self.rotate_button = QPushButton("Auto Rotate")
        self.rotate_button.setCheckable(True)
        self.rotate_button.toggled.connect(self.toggle_rotation)
        control_panel.addWidget(self.rotate_button)
        
        self.status_label = QLabel("Status: Ready")
        control_panel.addWidget(self.status_label)
        control_panel.addStretch()
        
        main_layout.addLayout(control_panel)
        
        # Split view for 3D visualizations
        splitter = QSplitter(Qt.Horizontal)
        
        # RGB point cloud view
        rgb_container = QWidget()
        rgb_layout = QVBoxLayout()
        rgb_layout.addWidget(QLabel("RGB Point Cloud"))
        self.rgb_view = PointCloudWidget("RGB Point Cloud")
        rgb_layout.addWidget(self.rgb_view)
        self.rgb_info_label = QLabel("Points: 0")
        rgb_layout.addWidget(self.rgb_info_label)
        rgb_container.setLayout(rgb_layout)
        
        # Anomaly point cloud view
        anom_container = QWidget()
        anom_layout = QVBoxLayout()
        anom_layout.addWidget(QLabel("Anomaly Point Cloud"))
        self.anom_view = PointCloudWidget("Anomaly Point Cloud")
        anom_layout.addWidget(self.anom_view)
        self.anom_info_label = QLabel("Anomalies: 0")
        anom_layout.addWidget(self.anom_info_label)
        anom_container.setLayout(anom_layout)
        
        splitter.addWidget(rgb_container)
        splitter.addWidget(anom_container)
        splitter.setSizes([700, 700])
        
        main_layout.addWidget(splitter)
        
        # Instructions
        instructions = QLabel("Use mouse to rotate view, scroll wheel to zoom")
        instructions.setStyleSheet("color: #666; padding: 5px;")
        main_layout.addWidget(instructions)
        
    def setup_workers(self):
        """Setup all workers and connections"""
        # Test waypoints configuration
        waypoints_config = {
            "home": {
                "position": [0, -90, 0, -90, -90, 0],
                "speed": 1.0,
                "acceleration": 1.0
            },
            "waypoints": {
                "wp1": {
                    "position": [10, -85, 5, -85, -90, 0],
                    "description": "Front view",
                    "speed": 0.5
                },
                "wp2": {
                    "position": [-10, -85, 5, -85, -90, 0],
                    "description": "Side view",
                    "speed": 0.5
                }
            }
        }
        
        # Create workers
        self.capture_worker = CaptureWorker(waypoints_config, robot_ip="192.168.1.10")
        self.infer_worker = InferWorker()
        self.rgb_pc_worker = RgbPCWorker(max_points=500000, voxel_size=0.005)
        self.anom_pc_worker = AnomPCWorker(spatial_threshold=0.02, min_points=50)
        
        # Move workers to threads
        self.capture_worker.moveToThread(self.capture_thread)
        self.infer_worker.moveToThread(self.infer_thread)
        self.rgb_pc_worker.moveToThread(self.rgb_pc_thread)
        self.anom_pc_worker.moveToThread(self.anom_pc_thread)
        
        # Connect signals - Capture Worker
        self.capture_worker.started.connect(lambda: self.update_status("Capture started"))
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.error.connect(self.on_error)
        self.capture_worker.frame_captured.connect(self.on_frame_captured)
        
        # Connect signals - RGB Point Cloud Worker
        self.rgb_pc_worker.point_cloud_updated.connect(self.on_rgb_pc_updated)
        self.rgb_pc_worker.error.connect(self.on_error)
        
        # Connect signals - Inference Worker
        self.infer_worker.anomaly_detected.connect(self.on_anomaly_detected)
        self.infer_worker.error.connect(self.on_error)
        
        # Connect signals - Anomaly Point Cloud Worker
        self.anom_pc_worker.point_cloud_updated.connect(self.on_anom_pc_updated)
        self.anom_pc_worker.error.connect(self.on_error)
        
        # Connect data flow between workers
        self.capture_worker.frame_captured.connect(self.rgb_pc_worker.process_frame)
        self.capture_worker.frame_captured.connect(self.anom_pc_worker.update_frame_data)
        self.infer_worker.anomaly_detected.connect(self.anom_pc_worker.process_anomaly)
        
        # Start threads
        self.capture_thread.start()
        self.infer_thread.start()
        self.rgb_pc_thread.start()
        self.anom_pc_thread.start()
        
    @Slot()
    def start_capture(self):
        """Start the capture and processing pipeline"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Start workers
        QTimer.singleShot(0, self.capture_worker.start_capture)
        QTimer.singleShot(0, self.infer_worker.start_inference)
        QTimer.singleShot(0, self.rgb_pc_worker.start)
        QTimer.singleShot(0, self.anom_pc_worker.start)
        
        self.update_status("Starting capture...")
        
    @Slot()
    def stop_capture(self):
        """Stop the capture and processing pipeline"""
        self.capture_worker.stop_capture()
        self.infer_worker.stop_inference()
        self.rgb_pc_worker.stop()
        self.anom_pc_worker.stop()
        
        self.update_status("Stopping capture...")
        
    @Slot(bool)
    def toggle_rotation(self, checked):
        """Toggle auto rotation"""
        self.auto_rotation_enabled = checked
        if checked:
            self.rotation_timer.start(50)  # 20 FPS
        else:
            self.rotation_timer.stop()
            
    @Slot()
    def auto_rotate(self):
        """Auto rotate both views"""
        self.rgb_view.rotate_camera(2)
        self.anom_view.rotate_camera(2)
        
    @Slot(object)
    def on_frame_captured(self, frame_data):
        """Handle captured frame data"""
        # Send RGB frame to inference worker
        if frame_data.get('rgb') is not None:
            self.infer_worker.process_frame(frame_data['rgb'])
            
    @Slot(object)
    def on_rgb_pc_updated(self, pc_data):
        """Handle RGB point cloud update"""
        self.rgb_view.update_point_cloud(pc_data)
        num_points = pc_data.get('num_points', 0)
        self.rgb_info_label.setText(f"Points: {num_points:,}")
        
    @Slot(object)
    def on_anomaly_detected(self, anomaly_data):
        """Handle anomaly detection"""
        # The anomaly worker will process this
        pass
        
    @Slot(object)
    def on_anom_pc_updated(self, pc_data):
        """Handle anomaly point cloud update"""
        self.anom_view.update_point_cloud(pc_data)
        num_anomalies = pc_data.get('num_anomalies', 0)
        self.anom_info_label.setText(f"Anomalies: {num_anomalies}")
        
    @Slot()
    def on_capture_finished(self):
        """Handle capture finished"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Capture finished")
        
    @Slot(str)
    def on_error(self, error_msg):
        """Handle errors from workers"""
        self.update_status(f"Error: {error_msg}")
        print(f"Error: {error_msg}")
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(f"Status: {message}")
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop all workers
        self.capture_worker.stop_capture()
        self.infer_worker.stop_inference()
        self.rgb_pc_worker.stop()
        self.anom_pc_worker.stop()
        
        # Stop auto rotation
        self.rotation_timer.stop()
        
        # Wait for threads to finish
        self.capture_thread.quit()
        self.capture_thread.wait()
        self.infer_thread.quit()
        self.infer_thread.wait()
        self.rgb_pc_thread.quit()
        self.rgb_pc_thread.wait()
        self.anom_pc_thread.quit()
        self.anom_pc_thread.wait()
        
        event.accept()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = SplitView3DTest()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()