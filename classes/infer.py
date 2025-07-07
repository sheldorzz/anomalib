# workers/infer.py
import numpy as np
import torch
import cv2
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThread, Slot
from anomalib.models import Patchcore
from anomalib.engine import Engine
#from anomalib.data.utils import InputNormalizationMethod
import torchvision.transforms as T
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """Worker for streaming inference using trained PatchCore model"""
    
    # Signals
    started = Signal()
    anomaly_detected = Signal(object)  # Emits dict with anomaly data
    finished = Signal()
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.engine = None
        self.is_running = False
        self.device = None
        self.transform = None
        
        # Model paths
        self.model_dir = Path("models")
        self.checkpoint_path = self.model_dir / "patchcore_checkpoint.ckpt"
        self.model_state_path = self.model_dir / "patchcore.pt"
        
        # Inference parameters
        self.anomaly_threshold = 0.5  # Will be updated from model metadata
        self.pixel_threshold = 0.5
        self.min_anomaly_area = 100  # Minimum pixels for valid anomaly
        
        # Frame queue for async processing
        self.frame_queue = []
        self.max_queue_size = 5
        
    def load_model(self):
        """Load trained PatchCore model"""
        try:
            # Check if checkpoint exists
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model state for configuration
            if self.model_state_path.exists():
                model_state = torch.load(self.model_state_path, map_location=self.device)
                backbone = model_state.get('backbone', 'wide_resnet50_2')
                layers = model_state.get('layers', ['layer2', 'layer3'])
                coreset_sampling_ratio = model_state.get('coreset_sampling_ratio', 0.1)
                num_neighbors = model_state.get('num_neighbors', 9)
                normalization_stats = model_state.get('normalization_stats', {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                })
            else:
                # Default configuration
                backbone = 'wide_resnet50_2'
                layers = ['layer2', 'layer3']
                coreset_sampling_ratio = 0.1
                num_neighbors = 9
                normalization_stats = {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            
            # Create model
            self.model = Patchcore(
                backbone=backbone,
                layers=layers,
                pre_trained=True,
                coreset_sampling_ratio=coreset_sampling_ratio,
                num_neighbors=num_neighbors
            )
            
            # Create engine for inference
            self.engine = Engine(
                accelerator="auto",
                devices=1,
                enable_progress_bar=False
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract thresholds if available
            if 'callbacks' in checkpoint:
                for callback in checkpoint.get('callbacks', {}).values():
                    if isinstance(callback, dict) and 'image_threshold' in callback:
                        self.anomaly_threshold = float(callback['image_threshold'])
                        self.pixel_threshold = float(callback.get('pixel_threshold', self.pixel_threshold))
                        logger.info(f"Loaded thresholds - Image: {self.anomaly_threshold}, Pixel: {self.pixel_threshold}")
                        break
            
            # Setup transforms
            self.setup_transforms(normalization_stats)
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.error.emit(f"Failed to load model: {str(e)}")
            return False
    
    def setup_transforms(self, normalization_stats):
        """Setup image preprocessing transforms"""
        mean = normalization_stats.get('mean', [0.485, 0.456, 0.406])
        std = normalization_stats.get('std', [0.229, 0.224, 0.225])
        
        self.transform = T.Compose([
            T.Resize((720, 1280)),  # Match training resolution
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    
    def preprocess_frame(self, frame):
        """Preprocess frame for inference"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def run_inference(self, frame):
        """Run inference on a single frame"""
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Use engine.predict for proper inference pipeline
                predictions = self.engine.predict(
                    model=self.model,
                    dataloaders=[input_tensor],
                    return_predictions=True
                )
                
                # Extract results from predictions
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    
                    # Get anomaly map and score
                    anomaly_map = pred.anomaly_maps[0].cpu().numpy() if hasattr(pred, 'anomaly_maps') else None
                    image_score = float(pred.image_scores[0]) if hasattr(pred, 'image_scores') else 0.0
                    
                    # Get prediction label
                    image_labels = pred.image_labels[0].item() if hasattr(pred, 'image_labels') else 0
                    is_anomalous = image_labels == 1 or image_score > self.anomaly_threshold
                    
                    return {
                        'anomaly_map': anomaly_map,
                        'image_score': image_score,
                        'is_anomalous': is_anomalous,
                        'pixel_scores': anomaly_map if anomaly_map is not None else np.zeros((720, 1280))
                    }
            
            # Fallback direct model inference if engine predict fails
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Extract anomaly map and score
                if hasattr(output, 'anomaly_maps'):
                    anomaly_map = output.anomaly_maps[0].cpu().numpy()
                else:
                    anomaly_map = np.zeros((720, 1280))
                
                if hasattr(output, 'pred_scores'):
                    image_score = float(output.pred_scores[0])
                else:
                    image_score = 0.0
                
                is_anomalous = image_score > self.anomaly_threshold
                
                return {
                    'anomaly_map': anomaly_map,
                    'image_score': image_score,
                    'is_anomalous': is_anomalous,
                    'pixel_scores': anomaly_map
                }
                
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return None
    
    def post_process_anomaly_map(self, anomaly_map, original_shape):
        """Post-process anomaly map to match original image shape"""
        # Resize anomaly map to original shape
        if anomaly_map.shape != original_shape[:2]:
            anomaly_map = cv2.resize(anomaly_map, (original_shape[1], original_shape[0]))
        
        # Normalize to 0-1 range if needed
        if anomaly_map.max() > 1.0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        
        return anomaly_map
    
    def detect_anomaly_regions(self, anomaly_map):
        """Detect individual anomaly regions in the anomaly map"""
        # Threshold the anomaly map
        binary_map = (anomaly_map > self.pixel_threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomaly_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_anomaly_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region score
                region_mask = np.zeros_like(anomaly_map)
                cv2.drawContours(region_mask, [contour], -1, 1, -1)
                region_score = np.mean(anomaly_map[region_mask > 0])
                
                anomaly_regions.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area,
                    'score': float(region_score),
                    'center': (x + w // 2, y + h // 2)
                })
        
        # Sort by score
        anomaly_regions.sort(key=lambda x: x['score'], reverse=True)
        
        return anomaly_regions
    
    @Slot()
    def start_inference(self):
        """Start inference streaming"""
        self.started.emit()
        self.is_running = True
        
        # Load model
        if not self.load_model():
            self.finished.emit()
            return
        
        logger.info("Inference worker started")
    
    @Slot(object)
    def process_frame(self, frame):
        """Process a single frame for anomaly detection"""
        if not self.is_running or self.model is None:
            return
        
        try:
            # Add frame to queue
            if len(self.frame_queue) >= self.max_queue_size:
                self.frame_queue.pop(0)  # Remove oldest frame
            
            # Store original frame shape
            original_shape = frame.shape
            
            # Run inference
            result = self.run_inference(frame)
            
            if result and result['is_anomalous']:
                # Post-process anomaly map
                anomaly_map = self.post_process_anomaly_map(
                    result['anomaly_map'], 
                    original_shape
                )
                
                # Detect anomaly regions
                anomaly_regions = self.detect_anomaly_regions(anomaly_map)
                
                # Emit anomaly data
                anomaly_data = {
                    'frame': frame,
                    'anomaly_map': anomaly_map,
                    'image_score': result['image_score'],
                    'anomaly_regions': anomaly_regions,
                    'timestamp': np.datetime64('now')
                }
                
                self.anomaly_detected.emit(anomaly_data)
                
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
    
    def stop_inference(self):
        """Stop inference streaming"""
        self.is_running = False
        self.frame_queue.clear()
        
        if self.model:
            self.model = None
            
        self.finished.emit()
        logger.info("Inference worker stopped")
    
    def update_thresholds(self, image_threshold=None, pixel_threshold=None):
        """Update anomaly detection thresholds"""
        if image_threshold is not None:
            self.anomaly_threshold = image_threshold
            logger.info(f"Updated image threshold: {self.anomaly_threshold}")
            
        if pixel_threshold is not None:
            self.pixel_threshold = pixel_threshold
            logger.info(f"Updated pixel threshold: {self.pixel_threshold}")