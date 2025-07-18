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
from skimage import measure

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """Worker for streaming inference using trained PatchCore model"""

    # Signals
    started = Signal()
    ready = Signal()
    anomaly_detected = Signal(object)  # Emits dict with anomaly data
    live_inference_result = Signal(object)  # For real-time 3D visualization
    anomaly_accumulation_complete = Signal(object)  # NEW: Emits accumulated anomaly data
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
        
        # NEW: Anomaly accumulation
        self.accumulated_anomaly_maps = []
        self.accumulated_frame_data = []
        self.global_min_anomaly = float('inf')
        self.global_max_anomaly = float('-inf')
        
        # NEW: Volume of interest (meters)
        self.voi = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.1, 'z_max': 2.0
        }
        
        # NEW: 3D bounding box parameters
        self.min_box_volume = 0.001  # m³ (1 liter)
        self.max_box_volume = 0.1    # m³ (100 liters)
        self.box_confidence_threshold = 0.85

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
            
            # Reset accumulation
            self.accumulated_anomaly_maps.clear()
            self.accumulated_frame_data.clear()
            self.global_min_anomaly = float('inf')
            self.global_max_anomaly = float('-inf')
            
            logger.info("Inference worker ready and accumulation reset")
            self.ready.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()

    def stop_inference(self):
        self._running = False
        logger.info(f"Stopping inference. Accumulated maps: {len(self.accumulated_anomaly_maps)}")
        
        # Process accumulated data before finishing
        if self.accumulated_anomaly_maps:
            self.process_accumulated_anomalies()
        else:
            logger.warning("No accumulated anomaly maps to process")
            # Still emit result to trigger voxelization
            result = {
                'bounding_boxes': [],
                'anomaly_points': np.array([]),
                'anomaly_scores': np.array([]),
                'global_min': self.global_min_anomaly,
                'global_max': self.global_max_anomaly
            }
            self.anomaly_accumulation_complete.emit(result)
            
        self.finished.emit()

    def enable_real_time_mode(self, enabled=True):
        """Enable/disable real-time processing mode"""
        self.real_time_mode = enabled
        if enabled:
            logger.info("Real-time inference mode enabled")
        else:
            logger.info("Real-time inference mode disabled")

    def set_volume_of_interest(self, voi_dict):
        """Set volume of interest for anomaly filtering"""
        self.voi.update(voi_dict)
        logger.info(f"Volume of interest updated: {self.voi}")

    def set_box_size_limits(self, min_volume_m3, max_volume_m3):
        """Set min/max volume for 3D bounding boxes"""
        self.min_box_volume = min_volume_m3
        self.max_box_volume = max_volume_m3
        logger.info(f"Box volume limits: {min_volume_m3:.3f} - {max_volume_m3:.3f} m³")

    def normalize_accumulated_anomaly_map(self, anomaly_map):
        """Normalize anomaly map using global min/max"""
        if self.global_max_anomaly > self.global_min_anomaly:
            normalized = (anomaly_map - self.global_min_anomaly) / (self.global_max_anomaly - self.global_min_anomaly)
        else:
            normalized = np.zeros_like(anomaly_map)
        return normalized

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
            
            # Update global min/max for normalization
            self.global_min_anomaly = min(self.global_min_anomaly, anomaly_map.min())
            self.global_max_anomaly = max(self.global_max_anomaly, anomaly_map.max())
            
            # Store for accumulation
            self.accumulated_anomaly_maps.append({
                'map': anomaly_map.copy(),
                'score': anomaly_score,
                'waypoint_id': frame_data.get("waypoint_id", "unknown"),
                'transform_matrix': frame_data.get("transform_matrix"),
                'camera_matrix': frame_data.get("camera_matrix"),
                'depth': frame_data.get("depth"),
                'rgb': bgr_frame.copy(),
                'intrinsics': frame_data.get("intrinsics")
            })
            self.accumulated_frame_data.append(frame_data.copy())
            
            # Log accumulation progress every 10 frames
            if len(self.accumulated_anomaly_maps) % 10 == 0:
                logger.info(f"Accumulated {len(self.accumulated_anomaly_maps)} anomaly maps")
            
            # Normalize using current global values
            normalized_map = self.normalize_accumulated_anomaly_map(anomaly_map)
            
            # Detect anomaly regions
            anomaly_regions = self.detect_anomaly_regions(anomaly_map)
            
            # Create binary anomaly mask for compatibility
            anomaly_mask = (anomaly_map > self.pixel_threshold).astype(np.uint8)
            
            # Prepare result for 3D visualization
            result = {
                "frame_data": frame_data,
                "anomaly_map": anomaly_map,
                "normalized_anomaly_map": normalized_map,  # NEW
                "anomaly_mask": anomaly_mask,
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

    def process_accumulated_anomalies(self):
        """Process all accumulated anomaly data to compute 3D bounding boxes"""
        if not self.accumulated_anomaly_maps:
            logger.info("No accumulated anomaly maps to process")
            # Still emit empty result to trigger voxelization
            result = {
                'bounding_boxes': [],
                'anomaly_points': np.array([]),
                'anomaly_scores': np.array([]),
                'global_min': self.global_min_anomaly,
                'global_max': self.global_max_anomaly
            }
            self.anomaly_accumulation_complete.emit(result)
            return
            
        try:
            logger.info(f"Processing {len(self.accumulated_anomaly_maps)} accumulated anomaly maps...")
            
            # Prepare 3D anomaly volume
            all_anomaly_points = []
            all_anomaly_scores = []
            associated_images = {}  # Map 3D points to source images
            
            for idx, data in enumerate(self.accumulated_anomaly_maps):
                anomaly_map = data['map']
                normalized_map = self.normalize_accumulated_anomaly_map(anomaly_map)
                transform = data['transform_matrix']
                camera_matrix = data['camera_matrix']
                depth = data['depth']
                rgb = data['rgb']
                waypoint_id = data['waypoint_id']
                
                # Generate 3D points for anomaly regions
                height, width = depth.shape
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                
                # Find high-confidence anomaly pixels
                anomaly_pixels = np.where(normalized_map > self.pixel_threshold)
                
                for i in range(len(anomaly_pixels[0])):
                    v, u = anomaly_pixels[0][i], anomaly_pixels[1][i]
                    z = depth[v, u]
                    
                    if 0.1 < z < 3.0:  # Valid depth range
                        # Convert to 3D
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        
                        # Apply transformation
                        point_cam = np.array([x, y, z, 1.0])
                        point_world = (transform @ point_cam)[:3]
                        
                        # Check if within VOI
                        if (self.voi['x_min'] <= point_world[0] <= self.voi['x_max'] and
                            self.voi['y_min'] <= point_world[1] <= self.voi['y_max'] and
                            self.voi['z_min'] <= point_world[2] <= self.voi['z_max']):
                            
                            all_anomaly_points.append(point_world)
                            all_anomaly_scores.append(normalized_map[v, u])
                            
                            # Track which image this point came from
                            point_key = tuple(np.round(point_world * 1000).astype(int))  # mm precision
                            if point_key not in associated_images:
                                associated_images[point_key] = {
                                    'waypoint_id': waypoint_id,
                                    'image_idx': idx,
                                    'rgb': rgb,
                                    'score': normalized_map[v, u]
                                }
            
            if not all_anomaly_points:
                logger.info("No anomaly points found within VOI")
                # Still emit result to trigger voxelization
                result = {
                    'bounding_boxes': [],
                    'anomaly_points': np.array([]),
                    'anomaly_scores': np.array([]),
                    'global_min': self.global_min_anomaly,
                    'global_max': self.global_max_anomaly
                }
                self.anomaly_accumulation_complete.emit(result)
                return
                
            # Convert to numpy array
            anomaly_points = np.array(all_anomaly_points)
            anomaly_scores = np.array(all_anomaly_scores)
            
            # Compute 3D bounding boxes using clustering
            bounding_boxes = self.compute_3d_bounding_boxes(
                anomaly_points, anomaly_scores, associated_images
            )
            
            # Emit results
            result = {
                'bounding_boxes': bounding_boxes,
                'anomaly_points': anomaly_points,
                'anomaly_scores': anomaly_scores,
                'global_min': self.global_min_anomaly,
                'global_max': self.global_max_anomaly
            }
            
            self.anomaly_accumulation_complete.emit(result)
            logger.info(f"Found {len(bounding_boxes)} anomaly regions")
            
        except Exception as e:
            self.error.emit(f"Error processing accumulated anomalies: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Still emit empty result to trigger voxelization
            result = {
                'bounding_boxes': [],
                'anomaly_points': np.array([]),
                'anomaly_scores': np.array([]),
                'global_min': self.global_min_anomaly,
                'global_max': self.global_max_anomaly
            }
            self.anomaly_accumulation_complete.emit(result)

    def compute_3d_bounding_boxes(self, points, scores, associated_images):
        """Compute 3D bounding boxes from anomaly points"""
        if len(points) == 0:
            return []
            
        # Voxelize points for clustering (5mm resolution)
        voxel_size = 0.005
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Create binary volume for connected components
        min_voxel = voxel_indices.min(axis=0)
        max_voxel = voxel_indices.max(axis=0)
        volume_shape = max_voxel - min_voxel + 1
        
        # Create binary volume
        binary_volume = np.zeros(volume_shape, dtype=bool)
        for i, voxel in enumerate(voxel_indices):
            if scores[i] > self.box_confidence_threshold:
                idx = tuple(voxel - min_voxel)
                binary_volume[idx] = True
        
        # Find connected components
        labeled_volume, num_features = ndimage.label(binary_volume)
        
        bounding_boxes = []
        
        for label_id in range(1, num_features + 1):
            # Get voxels for this component
            component_voxels = np.argwhere(labeled_volume == label_id) + min_voxel
            
            # Convert back to world coordinates
            component_min = component_voxels.min(axis=0) * voxel_size
            component_max = (component_voxels.max(axis=0) + 1) * voxel_size
            
            # Calculate volume
            box_volume = np.prod(component_max - component_min)
            
            # Filter by size
            if self.min_box_volume <= box_volume <= self.max_box_volume:
                # Find points in this box
                mask = np.all((points >= component_min) & (points <= component_max), axis=1)
                box_points = points[mask]
                box_scores = scores[mask]
                
                if len(box_points) > 0:
                    # Find best viewing angle (image with highest anomaly score in this region)
                    best_image_data = None
                    best_score = 0
                    
                    for point in box_points:
                        point_key = tuple(np.round(point * 1000).astype(int))
                        if point_key in associated_images:
                            img_data = associated_images[point_key]
                            if img_data['score'] > best_score:
                                best_score = img_data['score']
                                best_image_data = img_data
                    
                    bbox = {
                        'min': component_min.tolist(),
                        'max': component_max.tolist(),
                        'center': ((component_min + component_max) / 2).tolist(),
                        'volume': float(box_volume),
                        'mean_score': float(box_scores.mean()),
                        'max_score': float(box_scores.max()),
                        'num_points': len(box_points),
                        'label_id': label_id,
                        'best_view': best_image_data['waypoint_id'] if best_image_data else None,
                        'best_view_idx': best_image_data['image_idx'] if best_image_data else None
                    }
                    
                    bounding_boxes.append(bbox)
        
        # Sort by mean anomaly score
        bounding_boxes.sort(key=lambda x: x['mean_score'], reverse=True)
        
        return bounding_boxes

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