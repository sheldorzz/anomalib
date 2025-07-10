# test_split_view_3d.py
import sys
import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSplitter)
from PySide6.QtGui import QPixmap, QImage
import threading
import queue
import cv2

# Import the workers
from workers.capture import CaptureWorker
from workers.infer import InferWorker
from workers.pc_rgb import RgbPCWorker
from workers.pc_anom import AnomPCWorker


class Open3DWidget(QWidget):
    """Widget for displaying Open3D point cloud"""
    
    def __init__(self, window_name="3D View"):
        super().__init__()
        self.window_name = window_name
        self.vis = None
        self.point_cloud = o3d.geometry.PointCloud()
        self.update_queue = queue.Queue()
        self.vis_thread = None
        self.is_running = False
        
    def start_visualization(self):
        """Start the visualization in a separate thread"""
        self.is_running = True
        self.vis_thread = threading.Thread(target=self._vis_thread_func)
        self.vis_thread.start()
        
    def _vis_thread_func(self):
        """Visualization thread function"""
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=640, height=480)
        
        # Add empty point cloud
        self.vis.add_geometry(self.point_cloud)
        
        # Set render options
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set initial view
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Main visualization loop
        while self.is_running:
            # Check for updates
            try:
                update_data = self.update_queue.get(timeout=0.01)
                if update_data is not None:
                    points = update_data.get('points', np.array([]))
                    colors = update_data.get('colors', np.array([]))
                    
                    if points.shape[0] > 0:
                        self.point_cloud.points = o3d.utility.Vector3dVector(points)
                        if colors.shape[0] == points.shape[0]:
                            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
                        self.vis.update_geometry(self.point_cloud)
            except queue.Empty:
                pass
            
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
        
        # Cleanup
        self.vis.destroy_window()
        
    def update_point_cloud(self, data):
        """Queue point cloud update"""
        self.update_queue.put(data)
        
    def stop_visualization(self):
        """Stop the visualization thread"""
        self.is_running = False
        if self.vis_thread:
            self.vis_thread.join()


class SplitView3DTest(QMainWindow):
    """Main window with split view 3D visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Split View 3D Point Cloud Test")
        self.setGeometry(100, 100, 1400, 800)
        
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
        self.rgb_view = Open3DWidget("RGB Point Cloud")
        rgb_layout.addWidget(self.rgb_view)
        self.rgb_info_label = QLabel("Points: 0")
        rgb_layout.addWidget(self.rgb_info_label)
        rgb_container.setLayout(rgb_layout)
        
        # Anomaly point cloud view
        anom_container = QWidget()
        anom_layout = QVBoxLayout()
        anom_layout.addWidget(QLabel("Anomaly Point Cloud"))
        self.anom_view = Open3DWidget("Anomaly Point Cloud")
        anom_layout.addWidget(self.anom_view)
        self.anom_info_label = QLabel("Anomalies: 0")
        anom_layout.addWidget(self.anom_info_label)
        anom_container.setLayout(anom_layout)
        
        splitter.addWidget(rgb_container)
        splitter.addWidget(anom_container)
        splitter.setSizes([700, 700])
        
        main_layout.addWidget(splitter)
        
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
        
        # Start visualization widgets
        self.rgb_view.start_visualization()
        self.anom_view.start_visualization()
        
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
        
        # Stop visualizations
        self.rgb_view.stop_visualization()
        self.anom_view.stop_visualization()
        
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