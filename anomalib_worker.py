"""
Qt Worker thread for anomaly detection operations.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import time
from anomalib_core import AnomalyDetector

logger = logging.getLogger(__name__)


class AnomalyDetectionWorker(QThread):
    """
    Worker thread for anomaly detection training and inference.
    
    Handles both model training on good images and inference
    for anomaly detection in a separate thread.
    """
    
    # Signals
    training_started = pyqtSignal()
    training_progress = pyqtSignal(dict)  # Detailed progress info
    training_completed = pyqtSignal(dict)  # training results
    training_error = pyqtSignal(str)
    
    inference_started = pyqtSignal()
    inference_result = pyqtSignal(dict)  # inference results
    batch_inference_progress = pyqtSignal(int, int)  # current, total
    batch_inference_completed = pyqtSignal(list)  # list of results
    inference_error = pyqtSignal(str)
    
    model_loaded = pyqtSignal(str)  # model type
    model_saved = pyqtSignal(str)  # save path
    
    def __init__(self, 
                 model_type: str = "patchcore",
                 device: str = "auto"):
        """
        Initialize anomaly detection worker.
        
        Args:
            model_type: Model to use ("patchcore" or "efficientad")
            device: Device to use ("auto", "cpu", or "cuda")
        """
        super().__init__()
        self.detector = AnomalyDetector(model_type, device)
        
        # Worker state
        self.is_running = False
        self.current_task = None
        self.task_data = {}
        
    def run(self):
        """Main worker thread loop."""
        self.is_running = True
        
        while self.is_running:
            if self.current_task is None:
                self.msleep(100)  # Sleep if no task
                continue
            
            try:
                if self.current_task == "train":
                    self._perform_training()
                elif self.current_task == "infer":
                    self._perform_inference()
                elif self.current_task == "batch_infer":
                    self._perform_batch_inference()
                elif self.current_task == "save":
                    self._save_model()
                elif self.current_task == "load":
                    self._load_model()
                else:
                    logger.warning(f"Unknown task: {self.current_task}")
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                if "train" in self.current_task:
                    self.training_error.emit(str(e))
                else:
                    self.inference_error.emit(str(e))
            
            finally:
                self.current_task = None
                self.task_data = {}
    
    def train_model(self,
                   data_path: Union[str, Path],
                   max_epochs: int = 1,
                   batch_size: int = 32,
                   num_workers: int = 8,
                   val_split_ratio: float = 0.1):
        """
        Queue a training task.
        
        Args:
            data_path: Path to folder with good images
            max_epochs: Maximum training epochs
            batch_size: Batch size
            num_workers: Data loading workers
            val_split_ratio: Validation split ratio
        """
        if self.current_task is not None:
            self.training_error.emit("Worker is busy with another task")
            return
        
        self.task_data = {
            'data_path': data_path,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'val_split_ratio': val_split_ratio
        }
        self.current_task = "train"
    
    def _perform_training(self):
        """Execute training task."""
        self.training_started.emit()
        
        # Create progress callback that emits signals
        def progress_callback(info):
            """Callback to emit progress signals."""
            self.training_progress.emit(info)
        
        # Add progress callback to task data
        self.task_data['progress_callback'] = progress_callback
        
        # Perform actual training
        result = self.detector.train(**self.task_data)
        
        self.training_completed.emit(result)
    
    def infer_image(self, 
                   image: np.ndarray,
                   return_visualization: bool = False):
        """
        Queue an inference task for a single image.
        
        Args:
            image: RGB image array
            return_visualization: Whether to return visualization
        """
        if self.current_task is not None:
            self.inference_error.emit("Worker is busy with another task")
            return
        
        if not self.detector.is_trained:
            self.inference_error.emit("Model not trained. Train the model first.")
            return
        
        self.task_data = {
            'image': image.copy(),
            'return_visualization': return_visualization
        }
        self.current_task = "infer"
    
    def _perform_inference(self):
        """Execute inference task."""
        self.inference_started.emit()
        
        result = self.detector.predict(
            self.task_data['image'],
            self.task_data['return_visualization']
        )
        
        self.inference_result.emit(result)
    
    def infer_batch(self,
                   images: List[np.ndarray],
                   return_visualization: bool = False):
        """
        Queue a batch inference task.
        
        Args:
            images: List of RGB image arrays
            return_visualization: Whether to return visualizations
        """
        if self.current_task is not None:
            self.inference_error.emit("Worker is busy with another task")
            return
        
        if not self.detector.is_trained:
            self.inference_error.emit("Model not trained. Train the model first.")
            return
        
        self.task_data = {
            'images': [img.copy() for img in images],
            'return_visualization': return_visualization
        }
        self.current_task = "batch_infer"
    
    def _perform_batch_inference(self):
        """Execute batch inference task."""
        self.inference_started.emit()
        
        images = self.task_data['images']
        return_viz = self.task_data['return_visualization']
        results = []
        
        for i, image in enumerate(images):
            # Update progress
            self.batch_inference_progress.emit(i + 1, len(images))
            
            # Perform inference
            result = self.detector.predict(image, return_viz)
            results.append(result)
            
            # Small delay to prevent UI freezing
            if i % 10 == 0:
                self.msleep(1)
        
        self.batch_inference_completed.emit(results)
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Queue a model save task.
        
        Args:
            save_path: Path to save the model
        """
        if self.current_task is not None:
            self.inference_error.emit("Worker is busy with another task")
            return
        
        self.task_data = {'save_path': save_path}
        self.current_task = "save"
    
    def _save_model(self):
        """Execute model save task."""
        save_path = self.task_data['save_path']
        self.detector.save_model(save_path)
        self.model_saved.emit(str(save_path))
    
    def load_model(self, checkpoint_path: Union[str, Path]):
        """
        Queue a model load task.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        if self.current_task is not None:
            self.inference_error.emit("Worker is busy with another task")
            return
        
        self.task_data = {'checkpoint_path': checkpoint_path}
        self.current_task = "load"
    
    def _load_model(self):
        """Execute model load task."""
        checkpoint_path = self.task_data['checkpoint_path']
        self.detector.load_model(checkpoint_path)
        self.model_loaded.emit(self.detector.model_type)
    
    def set_model_type(self, model_type: str):
        """
        Change the model type.
        
        Args:
            model_type: "patchcore" or "efficientad"
        """
        if self.current_task is not None:
            logger.warning("Cannot change model type while worker is busy")
            return
        
        if model_type != self.detector.model_type:
            self.detector = AnomalyDetector(model_type, self.detector.device)
            logger.info(f"Model type changed to: {model_type}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict with model information
        """
        return {
            'model_type': self.detector.model_type,
            'is_trained': self.detector.is_trained,
            'device': self.detector.device,
            'image_size': self.detector.image_size
        }
    
    def stop(self):
        """Stop the worker thread."""
        self.is_running = False
        if self.isRunning():
            self.wait(5000)  # Wait up to 5 seconds
            if self.isRunning():
                logger.warning("Anomaly detection worker did not stop gracefully")
                self.terminate()