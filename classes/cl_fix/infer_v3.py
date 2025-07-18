import numpy as np
import torch
import cv2
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
from anomalib.deploy import TorchInferencer
import torchvision.transforms as T
from PIL import Image
import logging
import traceback
from scipy import ndimage
from collections import defaultdict
from sklearn.cluster import DBSCAN
import open3d as o3d

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """Worker for streaming inference using trained PatchCore model"""

    # Signals
    started = Signal()
    ready = Signal()
    anomaly_detected = Signal(object)  # Emits dict with anomaly data
    live_inference_result = Signal(object)  # For real-time 3D visualization
    capture_completed = Signal(object)  # NEW: Emits accumulated data after capture
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_path="models/weights/torch/patchcore.pt", device="auto", score_thr=0.8):
        super().__init__()
        self._frame_idx = 0
        self.map_dir = Path("results/maps")
        self.map_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path)
        self.device = device
        self.score_thr = score_thr
        self.pixel_threshold = 0.8  # Controls region segmentation
        self.min_anomaly_area = 200  # Filter out small blobs (in pixels)
        self.inferencer = None
        self._running = False
        
        # Real-time processing mode
        self.real_time_mode = False
        
        # NEW: Anomaly accumulation data structures
        self.accumulated_anomaly_maps = []
        self.accumulated_frame_data = []
        self.accumulated_rgb_images = []
        self.accumulated_depth_images = []
        self.accumulated_transforms = []
        self.accumulated_waypoint_ids = []
        
        # NEW: Volume of Interest (VOI) bounds
        self.voi_bounds = None  # Will be set as (x_min, x_max, y_min, y_max, z_min, z_max)
        
        # NEW: Anomaly size filtering
        self.min_anomaly_volume = 100  # Minimum voxel count for anomaly
        self.max_anomaly_volume = 50000  # Maximum voxel count for anomaly

    def _save_anomaly_map(self, anomaly_map: np.ndarray, bgr_frame: np.ndarray, anomaly_regions=None):
        # Optional: smooth map (helps reduce noise)
        # anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)

        # Normalize to 0–255
        raw_uint8 = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(raw_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (bgr_frame.shape[1], bgr_frame.shape[0]))
        overlay = cv2.addWeighted(bgr_frame, 0.6, heatmap, 0.4, 0)

        # Draw bounding boxes if available
        if anomaly_regions:
            for region in anomaly_regions:
                x, y, w, h = region["bbox"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save output only if not in real-time mode
        if not self.real_time_mode:
            out = self.map_dir / f"frame_{self._frame_idx:06d}"
            cv2.imwrite(str(out.with_name(out.name + "_raw.png")), raw_uint8)
            cv2.imwrite(str(out.with_name(out.name + "_heat.png")), heatmap)
            cv2.imwrite(str(out.with_name(out.name + "_overlay.png")), overlay)

        self._frame_idx += 1
        return raw_uint8, heatmap, overlay

    @Slot()
    def start_inference(self):
        self.started.emit()
        try:
            self.inferencer = TorchInferencer(path=str(self.model_path), device=self.device)
            self._running = True
            
            # NEW: Reset accumulation on start
            self.reset_accumulation()
            
            self.ready.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()

    def stop_inference(self):
        self._running = False
        
        # NEW: Process accumulated data before finishing
        if len(self.accumulated_anomaly_maps) > 0:
            self.process_accumulated_data()
            
        self.finished.emit()

    def reset_accumulation(self):
        """NEW: Reset accumulated data structures"""
        self.accumulated_anomaly_maps.clear()
        self.accumulated_frame_data.clear()
        self.accumulated_rgb_images.clear()
        self.accumulated_depth_images.clear()
        self.accumulated_transforms.clear()
        self.accumulated_waypoint_ids.clear()

    def enable_real_time_mode(self, enabled=True):
        """Enable/disable real-time processing mode"""
        self.real_time_mode = enabled
        if enabled:
            logger.info("Real-time inference mode enabled")
        else:
            logger.info("Real-time inference mode disabled")

    def set_volume_of_interest(self, bounds):
        """NEW: Set volume of interest for anomaly filtering
        
        Args:
            bounds: tuple of (x_min, x_max, y_min, y_max, z_min, z_max) in meters
        """
        self.voi_bounds = bounds
        logger.info(f"Volume of interest set: {bounds}")

    def set_anomaly_size_filters(self, min_volume, max_volume):
        """NEW: Set min/max anomaly volume filters
        
        Args:
            min_volume: Minimum voxel count for valid anomaly
            max_volume: Maximum voxel count for valid anomaly
        """
        self.min_anomaly_volume = min_volume
        self.max_anomaly_volume = max_volume
        logger.info(f"Anomaly size filters: min={min_volume}, max={max_volume}")

    def process_frame(self, bgr_frame: np.ndarray):
        """Process frame for anomaly detection (original method)"""
        if not self._running or self.inferencer is None:
            return

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        preds = self.inferencer.predict(rgb)
        amap = preds.anomaly_map.squeeze().cpu().numpy()
        score = float(preds.pred_score)

        # Detect regions based on pixel threshold
        anomaly_regions = self.detect_anomaly_regions(amap)

        # Save visual output
        raw_uint8, heatmap, overlay = self._save_anomaly_map(amap, bgr_frame, anomaly_regions)

        if score >= self.score_thr:
            payload = {
                "image_score": score,
                "anomaly_regions": anomaly_regions,
            }
            self.anomaly_detected.emit(payload)

    @Slot(object)
    def process_live_frame(self, frame_data):
        """Process live frame from capture worker for real-time 3D visualization"""
        if not self._running or self.inferencer is None:
            return

        try:
            # Extract RGB frame (BGR format from RealSense)
            bgr_frame = frame_data["rgb"]
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            preds = self.inferencer.predict(rgb_frame)
            anomaly_map = preds.anomaly_map.squeeze().cpu().numpy()
            anomaly_score = float(preds.pred_score)
            
            # NEW: Accumulate data for post-processing
            self.accumulated_anomaly_maps.append(anomaly_map.copy())
            self.accumulated_frame_data.append(frame_data.copy())
            self.accumulated_rgb_images.append(bgr_frame.copy())
            self.accumulated_depth_images.append(frame_data["depth"].copy())
            self.accumulated_transforms.append(frame_data["transform_matrix"].copy())
            self.accumulated_waypoint_ids.append(frame_data.get("waypoint_id", f"frame_{len(self.accumulated_waypoint_ids)}"))
            
            # Detect anomaly regions
            anomaly_regions = self.detect_anomaly_regions(anomaly_map)
            
            # Create normalized anomaly map for live visualization
            # Use current accumulated maps for normalization
            if len(self.accumulated_anomaly_maps) > 0:
                all_maps = np.array(self.accumulated_anomaly_maps)
                global_min = np.min(all_maps)
                global_max = np.max(all_maps)
                
                if global_max > global_min:
                    normalized_map = (anomaly_map - global_min) / (global_max - global_min)
                else:
                    normalized_map = anomaly_map
            else:
                normalized_map = anomaly_map
            
            # Prepare result for 3D visualization
            result = {
                "frame_data": frame_data,  # Original frame data with pose, depth, etc.
                "anomaly_map": anomaly_map,
                "normalized_anomaly_map": normalized_map,  # NEW: Globally normalized
                "anomaly_score": anomaly_score,
                "anomaly_regions": anomaly_regions,
                "has_anomaly": anomaly_score >= self.score_thr,
                "timestamp": frame_data.get("waypoint_id", "unknown")
            }
            
            # Emit for 3D visualization
            self.live_inference_result.emit(result)
            
            # Also emit traditional anomaly detection if threshold exceeded
            if anomaly_score >= self.score_thr:
                payload = {
                    "image_score": anomaly_score,
                    "anomaly_regions": anomaly_regions,
                    "waypoint_id": frame_data.get("waypoint_id", "unknown")
                }
                self.anomaly_detected.emit(payload)
                
        except Exception as e:
            self.error.emit(f"Live inference error: {str(e)}")

    def process_accumulated_data(self):
        """NEW: Process all accumulated data to compute 3D bounding boxes"""
        try:
            logger.info("Processing accumulated anomaly data...")
            
            if len(self.accumulated_anomaly_maps) == 0:
                logger.warning("No accumulated data to process")
                return
            
            # Compute global normalization
            all_maps = np.array(self.accumulated_anomaly_maps)
            global_min = np.min(all_maps)
            global_max = np.max(all_maps)
            
            # Normalize all anomaly maps globally
            normalized_maps = []
            for amap in self.accumulated_anomaly_maps:
                if global_max > global_min:
                    normalized = (amap - global_min) / (global_max - global_min)
                else:
                    normalized = amap
                normalized_maps.append(normalized)
            
            # Prepare data for 3D bounding box computation
            result_data = {
                "anomaly_maps": self.accumulated_anomaly_maps,
                "normalized_maps": normalized_maps,
                "rgb_images": self.accumulated_rgb_images,
                "depth_images": self.accumulated_depth_images,
                "transforms": self.accumulated_transforms,
                "waypoint_ids": self.accumulated_waypoint_ids,
                "frame_data": self.accumulated_frame_data,
                "global_min": global_min,
                "global_max": global_max,
                "voi_bounds": self.voi_bounds,
                "min_anomaly_volume": self.min_anomaly_volume,
                "max_anomaly_volume": self.max_anomaly_volume,
                "pixel_threshold": self.pixel_threshold
            }
            
            # Emit completed capture data
            self.capture_completed.emit(result_data)
            
            logger.info(f"Processed {len(self.accumulated_anomaly_maps)} frames for 3D analysis")
            
        except Exception as e:
            self.error.emit(f"Error processing accumulated data: {str(e)}")
            logger.error(traceback.format_exc())

    def process_frame_with_result(self, bgr_frame: np.ndarray):
        """Process frame and return full results for external use"""
        if not self._running or self.inferencer is None:
            return None

        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            preds = self.inferencer.predict(rgb)
            amap = preds.anomaly_map.squeeze().cpu().numpy()
            score = float(preds.pred_score)

            # Detect regions based on pixel threshold
            anomaly_regions = self.detect_anomaly_regions(amap)
            
            # Create binary mask
            anomaly_mask = (amap > self.pixel_threshold).astype(np.uint8)

            # Generate visualization maps
            raw_uint8, heatmap, overlay = self._save_anomaly_map(amap, bgr_frame, anomaly_regions)

            return {
                "anomaly_map": amap,
                "anomaly_mask": anomaly_mask,
                "anomaly_score": score,
                "anomaly_regions": anomaly_regions,
                "has_anomaly": score >= self.score_thr,
                "visualization": {
                    "raw_uint8": raw_uint8,
                    "heatmap": heatmap,
                    "overlay": overlay
                }
            }
            
        except Exception as e:
            self.error.emit(f"Frame processing error: {str(e)}")
            return None

    def detect_anomaly_regions(self, anomaly_map):
        """Detect individual anomaly regions in the anomaly map"""
        # Optional: smooth map
        # anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)

        # Threshold map
        binary_map = (anomaly_map > self.pixel_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anomaly_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_anomaly_area:
                x, y, w, h = cv2.boundingRect(contour)
                region_mask = np.zeros_like(anomaly_map)
                cv2.drawContours(region_mask, [contour], -1, 1, -1)
                region_score = np.mean(anomaly_map[region_mask > 0])
                anomaly_regions.append(
                    {"bbox": (x, y, w, h), "contour": contour, "area": area, "score": float(region_score), "center": (x + w // 2, y + h // 2)}
                )

        anomaly_regions.sort(key=lambda x: x["score"], reverse=True)
        return anomaly_regions

    def update_thresholds(self, image_threshold=None, pixel_threshold=None):
        """Update thresholds at runtime"""
        if image_threshold is not None:
            self.score_thr = image_threshold
            logger.info(f"Updated image score threshold: {self.score_thr}")
        if pixel_threshold is not None:
            self.pixel_threshold = pixel_threshold
            logger.info(f"Updated pixel threshold: {self.pixel_threshold}")
            
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.inferencer is None:
            return None
            
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "score_threshold": self.score_thr,
            "pixel_threshold": self.pixel_threshold,
            "min_anomaly_area": self.min_anomaly_area,
            "running": self._running
        }