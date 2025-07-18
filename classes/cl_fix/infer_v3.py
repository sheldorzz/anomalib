import numpy as np
import torch
import cv2
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
from anomalib.deploy import TorchInferencer
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """Worker for streaming inference using trained PatchCore model"""

    # Signals
    started = Signal()
    ready = Signal()
    anomaly_detected = Signal(object)  # Emits dict with anomaly data
    live_inference_result = Signal(object)  # For real-time 3D visualization
    capture_completed = Signal(object)  # Emits accumulated data after capture
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_path="models/weights/torch/patchcore.pt", device="auto", score_thr=0.8):
        super().__init__()
        self._frame_idx = 0
        self.model_path = Path(model_path)
        self.device = device
        self.score_thr = score_thr
        self.pixel_threshold = 0.8
        self.min_anomaly_area = 200
        self.inferencer = None
        self._running = False
        
        # Accumulation data structures
        self.accumulated_anomaly_maps = []
        self.accumulated_frame_data = []
        self.accumulated_rgb_images = []
        self.accumulated_depth_images = []
        self.accumulated_transforms = []
        self.accumulated_waypoint_ids = []
        self.accumulated_camera_matrices = []
        
        # Volume of Interest bounds
        self.voi_bounds = None
        
        # Anomaly size filtering
        self.min_anomaly_volume = 100
        self.max_anomaly_volume = 50000
        
        # Session directory for saving images
        self.session_dir = None

    @Slot()
    def start_inference(self):
        self.started.emit()
        try:
            self.inferencer = TorchInferencer(path=str(self.model_path), device=self.device)
            self._running = True
            
            # Reset accumulation and create session directory
            self.reset_accumulation()
            self.create_session_directory()
            
            self.ready.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()

    def create_session_directory(self):
        """Create a directory for this session's annotated images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"anomaly_sessions/session_{timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.error.emit(f"Session directory created: {self.session_dir}")

    def reset_accumulation(self):
        """Reset accumulated data structures"""
        self.accumulated_anomaly_maps.clear()
        self.accumulated_frame_data.clear()
        self.accumulated_rgb_images.clear()
        self.accumulated_depth_images.clear()
        self.accumulated_transforms.clear()
        self.accumulated_waypoint_ids.clear()
        self.accumulated_camera_matrices.clear()

    def set_volume_of_interest(self, bounds):
        """Set volume of interest for anomaly filtering"""
        self.voi_bounds = bounds

    def set_anomaly_size_filters(self, min_volume, max_volume):
        """Set min/max anomaly volume filters"""
        self.min_anomaly_volume = min_volume
        self.max_anomaly_volume = max_volume

    @Slot(object)
    def process_live_frame(self, frame_data):
        """Process live frame from capture worker"""
        if not self._running or self.inferencer is None:
            return

        try:
            # Extract data
            bgr_frame = frame_data["rgb"]
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            preds = self.inferencer.predict(rgb_frame)
            anomaly_map = preds.anomaly_map.squeeze().cpu().numpy()
            anomaly_score = float(preds.pred_score)
            
            # Accumulate data
            self.accumulated_anomaly_maps.append(anomaly_map)
            self.accumulated_frame_data.append(frame_data)
            self.accumulated_rgb_images.append(bgr_frame)
            self.accumulated_depth_images.append(frame_data["depth"])
            self.accumulated_transforms.append(frame_data["transform_matrix"])
            self.accumulated_waypoint_ids.append(frame_data.get("waypoint_id", f"frame_{len(self.accumulated_waypoint_ids)}"))
            self.accumulated_camera_matrices.append(frame_data["camera_matrix"])
            
            # Log progress
            if len(self.accumulated_anomaly_maps) % 10 == 0:
                self.error.emit(f"Accumulated {len(self.accumulated_anomaly_maps)} frames")
            
            # Normalize for live view
            normalized_map = anomaly_map / np.max(anomaly_map) if np.max(anomaly_map) > 0 else anomaly_map
            
            # Emit for live visualization
            result = {
                "frame_data": frame_data,
                "anomaly_map": anomaly_map,
                "normalized_anomaly_map": normalized_map,
                "anomaly_score": anomaly_score,
                "has_anomaly": anomaly_score >= self.score_thr,
                "timestamp": frame_data.get("waypoint_id", "unknown")
            }
            
            self.live_inference_result.emit(result)
                
        except Exception as e:
            self.error.emit(f"Live inference error: {str(e)}")

    @Slot()
    def trigger_final_processing(self):
        """Trigger processing of accumulated data"""
        self.error.emit(f"Final processing triggered with {len(self.accumulated_anomaly_maps)} frames")
        if len(self.accumulated_anomaly_maps) > 0:
            self.process_accumulated_data()

    def process_accumulated_data(self):
        """Process accumulated data and emit for voxelization"""
        try:
            num_frames = len(self.accumulated_anomaly_maps)
            self.error.emit(f"Processing {num_frames} accumulated frames...")
            
            # Package all accumulated data
            result_data = {
                "anomaly_maps": self.accumulated_anomaly_maps,
                "rgb_images": self.accumulated_rgb_images,
                "depth_images": self.accumulated_depth_images,
                "transforms": self.accumulated_transforms,
                "waypoint_ids": self.accumulated_waypoint_ids,
                "camera_matrices": self.accumulated_camera_matrices,
                "frame_data": self.accumulated_frame_data,
                "voi_bounds": self.voi_bounds,
                "min_anomaly_volume": self.min_anomaly_volume,
                "max_anomaly_volume": self.max_anomaly_volume,
                "pixel_threshold": self.pixel_threshold,
                "session_dir": str(self.session_dir)
            }
            
            # Emit for UI to process
            self.capture_completed.emit(result_data)
            
            # Stop inference
            self._running = False
            
        except Exception as e:
            self.error.emit(f"Error processing accumulated data: {str(e)}")
            logger.error(traceback.format_exc())

    def update_thresholds(self, image_threshold=None, pixel_threshold=None):
        """Update thresholds at runtime"""
        if image_threshold is not None:
            self.score_thr = image_threshold
        if pixel_threshold is not None:
            self.pixel_threshold = pixel_threshold

    def stop_inference(self):
        """Stop inference and trigger final processing"""
        self._running = False
        if len(self.accumulated_anomaly_maps) > 0:
            self.process_accumulated_data()
        self.finished.emit()