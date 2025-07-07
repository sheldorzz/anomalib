# workers/capture.py
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QObject, Signal, Slot, QThread
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import cv2


class CaptureWorker(QObject):
    """Worker for synchronized RealSense capture and robot movement"""
    
    # Signals
    started = Signal()
    progress = Signal(int)
    frame_captured = Signal(object)  # Emits dict with rgb, depth, pose
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, waypoints_config, robot_ip="192.168.1.10"):
        super().__init__()
        self.waypoints = waypoints_config
        self.robot_ip = robot_ip
        self.is_running = False
        
        # RealSense configuration
        self.pipeline = None
        self.config = None
        self.align = None
        
        # Robot interfaces
        self.rtde_control = None
        self.rtde_receive = None
        
        # Output directory
        self.output_dir = Path("data/train")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_realsense(self):
        """Initialize RealSense D455 camera"""
        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            # RGB stream at 1280x720 @ 30 FPS
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            # Depth stream at 1280x720 @ 30 FPS
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get device
            device = profile.get_device()
            
            # Get depth scale
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create align object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            # Wait for auto-exposure to settle
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
            return True
            
        except Exception as e:
            self.error.emit(f"Failed to setup RealSense: {str(e)}")
            return False
    
    def setup_robot(self):
        """Initialize robot RTDE interfaces"""
        try:
            self.rtde_control = RTDEControlInterface(self.robot_ip)
            self.rtde_receive = RTDEReceiveInterface(self.robot_ip)
            
            # Check connection
            if not self.rtde_control.isConnected():
                raise ConnectionError("Failed to connect to robot controller")
                
            return True
            
        except Exception as e:
            self.error.emit(f"Failed to setup robot: {str(e)}")
            return False
    
    def get_transform_matrix(self, pose):
        """Convert robot pose to 4x4 transformation matrix
        
        Args:
            pose: [x, y, z, rx, ry, rz] in meters and radians
            
        Returns:
            4x4 transformation matrix
        """
        x, y, z, rx, ry, rz = pose
        
        # Create rotation matrix from axis-angle representation
        angle = np.sqrt(rx**2 + ry**2 + rz**2)
        if angle > 0:
            k = np.array([rx, ry, rz]) / angle
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        else:
            R = np.eye(3)
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    def capture_frame(self):
        """Capture aligned RGB-D frame from RealSense"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert depth to meters
            depth_in_meters = depth_image * self.depth_scale
            
            return color_image, depth_in_meters
            
        except Exception as e:
            self.error.emit(f"Frame capture error: {str(e)}")
            return None, None
    
    def save_capture(self, rgb, depth, pose, waypoint_id, frame_count):
        """Save captured data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"wp{waypoint_id}_frame{frame_count:04d}_{timestamp}"
        
        # Save RGB image
        rgb_path = self.output_dir / f"{base_name}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb)
        
        # Save depth map as numpy array
        depth_path = self.output_dir / f"{base_name}_depth.npy"
        np.save(str(depth_path), depth)
        
        # Save metadata
        metadata = {
            "waypoint_id": waypoint_id,
            "frame_count": frame_count,
            "timestamp": timestamp,
            "pose": pose.tolist() if isinstance(pose, np.ndarray) else pose,
            "transform_matrix": self.get_transform_matrix(pose).tolist(),
            "depth_scale": self.depth_scale,
            "rgb_path": str(rgb_path.name),
            "depth_path": str(depth_path.name)
        }
        
        metadata_path = self.output_dir / f"{base_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def move_to_waypoint(self, waypoint_data):
        """Move robot to waypoint position"""
        position = waypoint_data["position"]
        speed = waypoint_data.get("speed", 1.0)
        acceleration = waypoint_data.get("acceleration", 1.0)
        async_move = waypoint_data.get("async", False)
        
        # Execute movement
        success = self.rtde_control.moveJ(position, speed, acceleration, async_move)
        
        if not success:
            self.error.emit(f"Failed to move to position: {position}")
            return False
        
        # Wait for movement to complete if not async
        if not async_move:
            # Small delay to ensure movement is complete
            time.sleep(0.1)
            
        return True
    
    @Slot()
    def start_capture(self):
        """Main capture routine"""
        self.started.emit()
        self.is_running = True
        
        # Setup devices
        if not self.setup_realsense() or not self.setup_robot():
            self.finished.emit()
            return
        
        try:
            # Move to home position first
            if "home" in self.waypoints:
                self.progress.emit(0)
                home_wp = self.waypoints["home"]
                if not self.move_to_waypoint(home_wp):
                    return
            
            # Get waypoint list
            waypoints = self.waypoints.get("waypoints", {})
            total_waypoints = len(waypoints)
            
            if total_waypoints == 0:
                self.error.emit("No waypoints defined")
                return
            
            # Capture at each waypoint
            for idx, (wp_id, wp_data) in enumerate(waypoints.items()):
                if not self.is_running:
                    break
                
                # Update progress
                progress = int((idx / total_waypoints) * 100)
                self.progress.emit(progress)
                
                # Move to waypoint
                if not self.move_to_waypoint(wp_data):
                    continue
                
                # Get current pose after movement
                current_pose = self.rtde_receive.getActualTCPPose()
                
                # Capture multiple frames at this position
                frames_per_waypoint = 10
                for frame_idx in range(frames_per_waypoint):
                    if not self.is_running:
                        break
                    
                    # Capture frame
                    rgb, depth = self.capture_frame()
                    if rgb is None or depth is None:
                        continue
                    
                    # Save to disk
                    self.save_capture(rgb, depth, current_pose, wp_id, frame_idx)
                    
                    # Emit frame data
                    frame_data = {
                        "rgb": rgb,
                        "depth": depth,
                        "pose": current_pose,
                        "waypoint_id": wp_id,
                        "waypoint_description": wp_data.get("description", "")
                    }
                    self.frame_captured.emit(frame_data)
                    
                    # Small delay between captures
                    time.sleep(0.1)
            
            # Return to home
            if "home" in self.waypoints:
                home_wp = self.waypoints["home"]
                self.move_to_waypoint(home_wp)
            
            self.progress.emit(100)
            
        except Exception as e:
            self.error.emit(f"Capture error: {str(e)}")
            
        finally:
            self.cleanup()
            self.finished.emit()
    
    def stop_capture(self):
        """Stop the capture process"""
        self.is_running = False
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            
        if self.rtde_control:
            self.rtde_control.disconnect()
            self.rtde_control = None
            
        if self.rtde_receive:
            self.rtde_receive.disconnect()
            self.rtde_receive = None