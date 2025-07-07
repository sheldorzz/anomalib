# workers/train.py
import os
import glob
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThread, Slot
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainWorker(QObject):
    """Worker for training Anomalib PatchCore model"""
    
    # Signals
    started = Signal()
    progress = Signal(int)
    result = Signal(object)
    finished = Signal()
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.engine = None
        self.datamodule = None
        self.is_running = False
        
        # Paths
        self.data_dir = Path("data/train")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "patchcore.pt"
        
        # Training parameters
        self.backbone = "wide_resnet50_2"
        self.layers = ["layer2", "layer3"]
        self.coreset_sampling_ratio = 0.1
        self.num_neighbors = 9
        
    def prepare_dataset(self):
        """Prepare dataset for training"""
        try:
            # Check if training data exists
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Training data directory not found: {self.data_dir}")
            
            # Count available images
            image_files = list(self.data_dir.glob("*_rgb.png"))
            if len(image_files) == 0:
                raise ValueError("No training images found")
            
            logger.info(f"Found {len(image_files)} training images")
            
            # Create anomalib Folder datamodule
            # For training, we only have normal images
            self.datamodule = Folder(
                name="robot_inspection",
                root=str(self.data_dir.parent),  # Parent of train folder
                normal_dir=str(self.data_dir.name),  # train folder contains normal images
                abnormal_dir=None,  # No abnormal images for training
                normal_test_dir=None,  # Will use synthetic anomalies for validation
                mask_dir=None,
                extensions=[".png"],
                task="segmentation",  # We want pixel-level anomaly detection
                test_split_mode=TestSplitMode.SYNTHETIC,  # Generate synthetic anomalies for validation
                test_split_ratio=0.2,  # Use 20% for validation
                seed=42,
                num_workers=4
            )
            
            # Setup the datamodule
            self.datamodule.setup()
            
            return True
            
        except Exception as e:
            self.error.emit(f"Dataset preparation failed: {str(e)}")
            return False
    
    def create_model(self):
        """Create PatchCore model"""
        try:
            # Initialize PatchCore model
            self.model = Patchcore(
                backbone=self.backbone,
                layers=self.layers,
                pre_trained=True,
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                num_neighbors=self.num_neighbors
            )
            
            logger.info(f"Created PatchCore model with backbone: {self.backbone}")
            return True
            
        except Exception as e:
            self.error.emit(f"Model creation failed: {str(e)}")
            return False
    
    def setup_engine(self):
        """Setup training engine"""
        try:
            # Create engine with appropriate settings
            self.engine = Engine(
                accelerator="auto",  # Automatically choose GPU if available
                devices=1,
                max_epochs=1,  # PatchCore doesn't need multiple epochs
                check_val_every_n_epoch=1,
                log_every_n_steps=10,
                enable_checkpointing=True,
                default_root_dir=str(self.model_dir),
                enable_progress_bar=False  # We'll use our own progress tracking
            )
            
            # Set up callbacks for progress tracking
            self.setup_callbacks()
            
            return True
            
        except Exception as e:
            self.error.emit(f"Engine setup failed: {str(e)}")
            return False
    
    def setup_callbacks(self):
        """Setup training callbacks for progress tracking"""
        # In a real implementation, we would add custom callbacks
        # to track training progress and emit progress signals
        pass
    
    @Slot()
    def start_training(self):
        """Main training routine"""
        self.started.emit()
        self.is_running = True
        
        try:
            # Step 1: Prepare dataset
            self.progress.emit(10)
            if not self.prepare_dataset():
                self.finished.emit()
                return
            
            # Step 2: Create model
            self.progress.emit(20)
            if not self.create_model():
                self.finished.emit()
                return
            
            # Step 3: Setup engine
            self.progress.emit(30)
            if not self.setup_engine():
                self.finished.emit()
                return
            
            # Step 4: Train model
            self.progress.emit(40)
            logger.info("Starting PatchCore training...")
            
            # For PatchCore, "training" is actually just feature extraction
            # from normal images and building the memory bank
            self.engine.fit(
                model=self.model,
                datamodule=self.datamodule
            )
            
            self.progress.emit(80)
            
            # Step 5: Save model
            logger.info(f"Saving model to {self.model_path}")
            
            # Export the trained model
            # Get the best checkpoint path
            checkpoint_path = self.engine.trainer.checkpoint_callback.best_model_path
            
            if checkpoint_path and Path(checkpoint_path).exists():
                # Load the best checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Save model state
                model_state = {
                    'model_state_dict': self.model.model.state_dict() if hasattr(self.model, 'model') else self.model.state_dict(),
                    'memory_bank': self.model.model.memory_bank if hasattr(self.model.model, 'memory_bank') else None,
                    'backbone': self.backbone,
                    'layers': self.layers,
                    'coreset_sampling_ratio': self.coreset_sampling_ratio,
                    'num_neighbors': self.num_neighbors,
                    'image_size': (720, 1280),  # Based on RealSense config
                    'normalization_stats': self.get_normalization_stats()
                }
                
                torch.save(model_state, self.model_path)
                logger.info(f"Model saved successfully to {self.model_path}")
                
                # Also save the full checkpoint for inference
                full_checkpoint_path = self.model_dir / "patchcore_checkpoint.ckpt"
                torch.save(checkpoint, full_checkpoint_path)
                
            else:
                # If no checkpoint, save current model state
                logger.warning("No checkpoint found, saving current model state")
                self.save_current_model()
            
            self.progress.emit(100)
            
            # Emit training results
            results = {
                'model_path': str(self.model_path),
                'checkpoint_path': str(full_checkpoint_path) if 'full_checkpoint_path' in locals() else None,
                'training_images': len(list(self.data_dir.glob("*_rgb.png"))),
                'backbone': self.backbone,
                'success': True
            }
            self.result.emit(results)
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.error.emit(f"Training failed: {str(e)}")
            
        finally:
            self.is_running = False
            self.finished.emit()
    
    def save_current_model(self):
        """Save current model state (fallback method)"""
        try:
            model_state = {
                'model': self.model,
                'backbone': self.backbone,
                'layers': self.layers,
                'coreset_sampling_ratio': self.coreset_sampling_ratio,
                'num_neighbors': self.num_neighbors,
                'image_size': (720, 1280),
                'normalization_stats': self.get_normalization_stats()
            }
            torch.save(model_state, self.model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    def get_normalization_stats(self):
        """Get normalization statistics for preprocessing"""
        # ImageNet statistics (commonly used for pretrained models)
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    
    def stop_training(self):
        """Stop the training process"""
        self.is_running = False
        if self.engine and hasattr(self.engine, 'trainer'):
            self.engine.trainer.should_stop = True