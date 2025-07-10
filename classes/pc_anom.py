# workers/pc_anom.py
import numpy as np
import open3d as o3d
from PySide6.QtCore import QObject, Signal, QThread,Slot
import cv2
import uuid
import logging
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class AnomPCWorker(QObject):
    """Worker for anomaly point cloud processing and bounding box generation"""
    
    # Signals
    started = Signal()
    point_cloud_updated = Signal(object)  # Emits dict with anomaly points
    anomaly_cropped = Signal(object)  # Emits dict with crop, bbox, pose
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, spatial_threshold=0.02, min_points=50):
        super().__init__()
        self.is_running = False
        self.spatial_threshold = spatial_threshold  # 2cm for anomaly grouping
        self.min_points = min_points  # Minimum points for valid anomaly
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Anomaly tracking
        self.anomalies = {}  # UUID -> anomaly data
        self.anomaly_point_cloud = o3d.geometry.PointCloud()
        
        # RGB point cloud reference (shared from RGB worker)
        self.rgb_point_cloud = None
        self.rgb_points = None
        self.rgb_colors = None
        
        # Camera intrinsics
        self.setup_camera_intrinsics()
        
        # Frame data cache
        self.current_frame_data = None
        
    def setup_camera_intrinsics(self):
        """Setup camera intrinsic parameters"""
        width = 1280
        height = 720
        fx = 643.884
        fy = 643.884
        cx = 634.661
        cy = 360.029
        
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Intrinsic matrix for projection
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def pose_to_matrix(self, pose):
        """Convert robot pose to transformation matrix"""
        x, y, z, rx, ry, rz = pose
        
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
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    def project_anomaly_to_3d(self, anomaly_map, depth, pose, threshold=0.5):
        """Project 2D anomaly map to 3D points
        
        Args:
            anomaly_map: 2D anomaly score map (H, W)
            depth: Depth image in meters (H, W)
            pose: Robot TCP pose
            threshold: Anomaly score threshold
            
        Returns:
            Tuple of (points, scores)
        """
        # Get anomalous pixel indices
        anomaly_mask = anomaly_map > threshold
        y_idx, x_idx = np.where(anomaly_mask)
        
        if len(y_idx) == 0:
            return np.array([]), np.array([])
        
        # Get depth values at anomalous pixels
        z_values = depth[y_idx, x_idx]
        
        # Filter out invalid depth
        valid_mask = (z_values > 0) & (z_values < 3.0)  # Max 3m range
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        z_values = z_values[valid_mask]
        
        if len(x_idx) == 0:
            return np.array([]), np.array([])
        
        # Back-project to 3D (camera frame)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        x_cam = (x_idx - cx) * z_values / fx
        y_cam = (y_idx - cy) * z_values / fy
        z_cam = z_values
        
        # Stack into points
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # Transform to robot base frame
        T = self.pose_to_matrix(pose)
        points_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_robot = (T @ points_h.T).T[:, :3]
        
        # Get anomaly scores for these points
        scores = anomaly_map[y_idx, x_idx]
        
        return points_robot, scores
    
    def find_or_create_anomaly(self, points, center):
        """Find existing anomaly or create new one based on spatial proximity"""
        with self.lock:
            # Check distance to existing anomalies
            for anomaly_id, anomaly_data in self.anomalies.items():
                existing_center = anomaly_data['center']
                distance = np.linalg.norm(center - existing_center)
                
                if distance < self.spatial_threshold:
                    return anomaly_id, False  # Found existing
            
            # Create new anomaly
            new_id = str(uuid.uuid4())
            self.anomalies[new_id] = {
                'id': new_id,
                'points': [],
                'scores': [],
                'center': center,
                'frames_seen': 0,
                'first_seen': np.datetime64('now'),
                'last_seen': np.datetime64('now'),
                'bboxes': {'aabb': None, 'obb': None}
            }
            
            return new_id, True  # Created new
    
    def update_anomaly(self, anomaly_id, points, scores):
        """Update anomaly with new points"""
        with self.lock:
            if anomaly_id not in self.anomalies:
                return
            
            anomaly = self.anomalies[anomaly_id]
            
            # Add new points
            anomaly['points'].append(points)
            anomaly['scores'].append(scores)
            anomaly['frames_seen'] += 1
            anomaly['last_seen'] = np.datetime64('now')
            
            # Update center
            all_points = np.vstack(anomaly['points'])
            anomaly['center'] = np.mean(all_points, axis=0)
            
            # Update bounding boxes if enough points
            if all_points.shape[0] >= self.min_points:
                self.update_bounding_boxes(anomaly_id, all_points)
    
    def update_bounding_boxes(self, anomaly_id, points):
        """Update AABB and OBB for anomaly"""
        # Create temporary point cloud
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Compute AABB
        aabb = temp_pcd.get_axis_aligned_bounding_box()
        aabb_min = np.asarray(aabb.min_bound)
        aabb_max = np.asarray(aabb.max_bound)
        
        # Compute OBB
        obb = temp_pcd.get_oriented_bounding_box()
        obb_center = np.asarray(obb.center)
        obb_extent = np.asarray(obb.extent)
        obb_rotation = np.asarray(obb.R)
        
        # Store bounding box info
        self.anomalies[anomaly_id]['bboxes'] = {
            'aabb': {
                'min': aabb_min,
                'max': aabb_max,
                'center': (aabb_min + aabb_max) / 2,
                'size': aabb_max - aabb_min
            },
            'obb': {
                'center': obb_center,
                'extent': obb_extent,
                'rotation': obb_rotation,
                'volume': np.prod(obb_extent)
            }
        }
    
    def crop_rgb_region(self, anomaly_id):
        """Crop RGB region around anomaly from point cloud"""
        if self.rgb_points is None or anomaly_id not in self.anomalies:
            return None
        
        anomaly = self.anomalies[anomaly_id]
        bbox = anomaly['bboxes']['aabb']
        
        if bbox is None:
            return None
        
        # Add margin to bounding box
        margin = 0.05  # 5cm margin
        bbox_min = bbox['min'] - margin
        bbox_max = bbox['max'] + margin
        
        # Find points within bounding box
        mask = np.all(self.rgb_points >= bbox_min, axis=1)
        mask &= np.all(self.rgb_points <= bbox_max, axis=1)
        
        if not np.any(mask):
            return None
        
        # Get cropped points and colors
        cropped_points = self.rgb_points[mask]
        cropped_colors = self.rgb_colors[mask]
        
        # Project to 2D for image crop
        if self.current_frame_data is not None:
            # Get current frame and pose
            frame = self.current_frame_data['frame']
            pose = self.current_frame_data['pose']
            
            # Transform points to camera frame
            T = self.pose_to_matrix(pose)
            T_inv = np.linalg.inv(T)
            points_h = np.hstack([cropped_points, np.ones((cropped_points.shape[0], 1))])
            points_cam = (T_inv @ points_h.T).T[:, :3]
            
            # Project to image
            x_img = (points_cam[:, 0] * self.K[0, 0] / points_cam[:, 2] + self.K[0, 2]).astype(int)
            y_img = (points_cam[:, 1] * self.K[1, 1] / points_cam[:, 2] + self.K[1, 2]).astype(int)
            
            # Get image crop bounds
            valid_mask = (x_img >= 0) & (x_img < frame.shape[1]) & (y_img >= 0) & (y_img < frame.shape[0])
            if np.any(valid_mask):
                x_min, x_max = x_img[valid_mask].min(), x_img[valid_mask].max()
                y_min, y_max = y_img[valid_mask].min(), y_img[valid_mask].max()
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Crop image
                image_crop = frame[y_min:y_max, x_min:x_max]
                
                return {
                    'image': image_crop,
                    'bounds_2d': (x_min, y_min, x_max, y_max),
                    'points_3d': cropped_points,
                    'colors': cropped_colors
                }
        
        return None
    
    @Slot()
    def start(self):
        """Start the worker"""
        self.started.emit()
        self.is_running = True
        logger.info("Anomaly point cloud worker started")
    
    @Slot(object)
    def update_rgb_reference(self, rgb_data):
        """Update reference to RGB point cloud data"""
        with self.lock:
            self.rgb_points = rgb_data.get('points')
            self.rgb_colors = rgb_data.get('colors')
    
    @Slot(object)
    def process_anomaly(self, anomaly_data):
        """Process anomaly detection results
        
        Args:
            anomaly_data: Dict with 'frame', 'anomaly_map', 'anomaly_regions', etc.
        """
        if not self.is_running:
            return
        
        try:
            # Extract data
            frame = anomaly_data['frame']
            anomaly_map = anomaly_data['anomaly_map']
            anomaly_regions = anomaly_data.get('anomaly_regions', [])
            
            # Get depth and pose from current frame data
            if self.current_frame_data is None:
                return
            
            depth = self.current_frame_data.get('depth')
            pose = self.current_frame_data.get('pose')
            
            if depth is None or pose is None:
                return
            
            # Cache current frame data for cropping
            self.current_frame_data['frame'] = frame
            
            # Process each anomaly region
            all_anomaly_points = []
            all_anomaly_scores = []
            
            for region in anomaly_regions:
                # Create region mask
                region_mask = np.zeros_like(anomaly_map)
                cv2.drawContours(region_mask, [region['contour']], -1, 1, -1)
                
                # Project region to 3D
                region_anomaly_map = anomaly_map * region_mask
                points, scores = self.project_anomaly_to_3d(
                    region_anomaly_map, depth, pose, threshold=0.3
                )
                
                if points.shape[0] >= self.min_points:
                    # Find or create anomaly
                    center = np.mean(points, axis=0)
                    anomaly_id, is_new = self.find_or_create_anomaly(points, center)
                    
                    # Update anomaly
                    self.update_anomaly(anomaly_id, points, scores)
                    
                    # Accumulate points for visualization
                    all_anomaly_points.append(points)
                    all_anomaly_scores.append(scores)
                    
                    # If new anomaly, emit crop for captioning
                    if is_new:
                        crop_data = self.crop_rgb_region(anomaly_id)
                        if crop_data is not None:
                            self.anomaly_cropped.emit({
                                'anomaly_id': anomaly_id,
                                'crop': crop_data['image'],
                                'bbox_3d': self.anomalies[anomaly_id]['bboxes'],
                                'pose': pose,
                                'score': region['score']
                            })
            
            # Update visualization
            if all_anomaly_points:
                combined_points = np.vstack(all_anomaly_points)
                combined_scores = np.hstack(all_anomaly_scores)
                
                # Color by anomaly score (red = high anomaly)
                colors = np.zeros((combined_points.shape[0], 3))
                colors[:, 0] = np.clip(combined_scores, 0, 1)  # Red channel
                
                # Emit point cloud update
                self.point_cloud_updated.emit({
                    'points': combined_points,
                    'colors': colors,
                    'num_anomalies': len(self.anomalies),
                    'anomaly_ids': list(self.anomalies.keys())
                })
            
        except Exception as e:
            logger.error(f"Error processing anomaly: {str(e)}")
            self.error.emit(f"Anomaly processing error: {str(e)}")
    
    Slot(object)
    def update_frame_data(self, frame_data):
        """Update current frame data from capture worker"""
        self.current_frame_data = frame_data
    
    def get_anomaly_summary(self):
        """Get summary of all detected anomalies"""
        with self.lock:
            summary = []
            for anomaly_id, data in self.anomalies.items():
                if data['bboxes']['aabb'] is not None:
                    summary.append({
                        'id': anomaly_id,
                        'center': data['center'].tolist(),
                        'num_points': sum(len(p) for p in data['points']),
                        'avg_score': np.mean([np.mean(s) for s in data['scores']]),
                        'volume': data['bboxes']['obb']['volume'] if data['bboxes']['obb'] else 0,
                        'first_seen': str(data['first_seen']),
                        'frames_seen': data['frames_seen']
                    })
            return summary
    
    @Slot()
    def stop(self):
        """Stop the worker"""
        self.is_running = False
        
        # Log anomaly summary
        summary = self.get_anomaly_summary()
        logger.info(f"Detected {len(summary)} anomalies")
        for anomaly in summary:
            logger.info(f"Anomaly {anomaly['id'][:8]}: {anomaly['num_points']} points, score={anomaly['avg_score']:.3f}")
        
        self.finished.emit()