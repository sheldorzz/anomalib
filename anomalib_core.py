"""
Core anomaly detection module using Anomalib.

Supports training and inference with EfficientAD and PatchCore models.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import logging
import time

try:
    from anomalib.data import Folder
    from anomalib.models import Patchcore, EfficientAd
    from anomalib.engine import Engine
    from anomalib.data.utils import TestSplitMode
    from anomalib.callbacks import ModelCheckpoint
    import torch
    from pytorch_lightning.callbacks import Callback, ProgressBar
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    logging.warning("Anomalib not installed. Install with: pip install anomalib")

logger = logging.getLogger(__name__)


class TrainingProgressCallback(Callback):
    """Lightning callback to track training progress."""
    
    def __init__(self, progress_callback=None):
        """
        Initialize progress callback.
        
        Args:
            progress_callback: Function to call with progress updates
        """
        super().__init__()
        self.progress_callback = progress_callback
        self.current_epoch = 0
        self.total_epochs = 0
        self.metrics = {}
        
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        self.total_epochs = trainer.max_epochs
        if self.progress_callback:
            self.progress_callback({
                'stage': 'start',
                'total_epochs': self.total_epochs
            })
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        self.current_epoch = trainer.current_epoch
        if self.progress_callback:
            self.progress_callback({
                'stage': 'epoch_start',
                'current_epoch': self.current_epoch + 1,
                'total_epochs': self.total_epochs
            })
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after each training batch."""
        if self.progress_callback and batch_idx % 10 == 0:  # Update every 10 batches
            total_batches = len(trainer.train_dataloader)
            self.progress_callback({
                'stage': 'batch',
                'current_epoch': self.current_epoch + 1,
                'total_epochs': self.total_epochs,
                'batch': batch_idx + 1,
                'total_batches': total_batches,
                'loss': outputs.get('loss', 0.0) if isinstance(outputs, dict) else 0.0
            })
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        metrics = trainer.callback_metrics
        self.metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))}
        
        if self.progress_callback:
            self.progress_callback({
                'stage': 'validation',
                'current_epoch': self.current_epoch + 1,
                'total_epochs': self.total_epochs,
                'metrics': self.metrics
            })
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        if self.progress_callback:
            self.progress_callback({
                'stage': 'end',
                'final_metrics': self.metrics
            })


class AnomalyDetector:
    """
    Core class for anomaly detection using Anomalib.
    
    Supports training on good images only and inference to detect anomalies.
    """
    
    def __init__(self, 
                 model_type: str = "patchcore",
                 device: str = "auto"):
        """
        Initialize anomaly detector.
        
        Args:
            model_type: Model to use ("patchcore" or "efficientad")
            device: Device to use ("auto", "cpu", or "cuda")
        """
        if not ANOMALIB_AVAILABLE:
            raise ImportError("Anomalib is not installed. Install with: pip install anomalib")
            
        self.model_type = model_type.lower()
        self.device = self._setup_device(device)
        self.model = None
        self.engine = None
        self.is_trained = False
        
        # Model configuration
        self.image_size = (256, 256)  # Default image size
        
        logger.info(f"Initialized AnomalyDetector with {self.model_type} on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _create_model(self) -> Union[Patchcore, EfficientAd]:
        """Create the anomaly detection model."""
        if self.model_type == "patchcore":
            return Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                pre_trained=True,
                coreset_sampling_ratio=0.1,
                num_neighbors=9
            )
        elif self.model_type == "efficientad":
            return EfficientAd(
                model_size="s",  # small model for faster training
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self,
              data_path: Union[str, Path],
              max_epochs: int = 1,
              batch_size: int = 32,
              num_workers: int = 8,
              val_split_ratio: float = 0.1,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train the model on good images only.
        
        Args:
            data_path: Path to folder containing only good/normal images
            max_epochs: Maximum training epochs
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            val_split_ratio: Ratio of data to use for validation
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict with training results
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        logger.info(f"Starting training on data from: {data_path}")
        
        try:
            # Create datamodule for unsupervised learning
            datamodule = Folder(
                name="custom_dataset",
                root=data_path.parent,
                normal_dir=data_path.name,
                abnormal_dir=None,  # No abnormal images for training
                test_split_mode=TestSplitMode.FROM_DIR,
                val_split_ratio=val_split_ratio,
                image_size=self.image_size,
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=num_workers,
                task="segmentation"  # For pixel-level anomaly maps
            )
            
            # Create model
            self.model = self._create_model()
            
            # Setup callbacks
            callbacks = []
            
            # Add progress tracking callback
            if progress_callback:
                progress_cb = TrainingProgressCallback(progress_callback)
                callbacks.append(progress_cb)
            
            # Add checkpoint callback
            checkpoint_cb = ModelCheckpoint(
                mode="max",
                monitor="pixel_F1Score",
            )
            callbacks.append(checkpoint_cb)
            
            # Create engine with callbacks
            self.engine = Engine(
                max_epochs=max_epochs,
                accelerator=self.device,
                devices=1,
                check_val_every_n_epoch=1,
                callbacks=callbacks,
                enable_progress_bar=True
            )
            
            # Train the model
            logger.info("Starting model training...")
            start_time = time.time()
            
            self.engine.fit(
                model=self.model,
                datamodule=datamodule
            )
            
            training_time = time.time() - start_time
            self.is_trained = True
            logger.info("Training completed successfully")
            
            # Get final metrics
            metrics = self.engine.trainer.callback_metrics
            results = {
                "status": "success",
                "model_type": self.model_type,
                "epochs_trained": self.engine.trainer.current_epoch + 1,
                "training_time": training_time,
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))}
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def predict(self, 
                image: np.ndarray,
                return_visualization: bool = False) -> Dict[str, Any]:
        """
        Perform inference on a single image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            return_visualization: Whether to return visualization images
            
        Returns:
            Dict containing:
                - anomaly_map: Pixel-level anomaly heatmap
                - anomaly_score: Image-level anomaly score
                - is_anomalous: Boolean prediction
                - visualization: Optional visualization dict
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Ensure image is RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Extract results
        anomaly_map = outputs.anomaly_map[0].cpu().numpy()
        anomaly_score = float(outputs.pred_score[0].cpu())
        
        # Determine threshold (this should ideally be calibrated on validation data)
        threshold = 0.5  # Default threshold
        is_anomalous = anomaly_score > threshold
        
        result = {
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "threshold": threshold
        }
        
        if return_visualization:
            result["visualization"] = self._create_visualization(image, anomaly_map, anomaly_score)
        
        return result
    
    def _create_visualization(self, 
                            image: np.ndarray, 
                            anomaly_map: np.ndarray,
                            score: float) -> Dict[str, np.ndarray]:
        """Create visualization images."""
        import cv2
        
        # Normalize anomaly map to 0-255
        anomaly_map_norm = (anomaly_map * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        # Add score text
        text = f"Score: {score:.3f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "anomaly_map": anomaly_map_norm
        }
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model checkpoint
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using the trainer's checkpoint callback
        checkpoint_path = self.engine.trainer.checkpoint_callback.best_model_path
        
        # Copy checkpoint to desired location
        import shutil
        shutil.copy2(checkpoint_path, save_path)
        
        logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, checkpoint_path: Union[str, Path]):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model and engine if not exists
        if self.model is None:
            self.model = self._create_model()
        if self.engine is None:
            self.engine = Engine()
        
        # Load checkpoint
        self.model = self.model.__class__.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.is_trained = True
        logger.info(f"Model loaded from: {checkpoint_path}")