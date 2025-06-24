"""
Example usage of the anomaly detection core and worker.

Demonstrates training on good images and performing inference.
"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSlot
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.anomaly_detection_worker import AnomalyDetectionWorker

logging.basicConfig(level=logging.INFO)


class AnomalyDetectionDemo(QObject):
    """Demo application for anomaly detection."""
    
    def __init__(self, model_type="patchcore"):
        super().__init__()
        
        # Initialize worker
        self.worker = AnomalyDetectionWorker(model_type=model_type)
        
        # Connect signals
        self._connect_signals()
        
        # Start worker
        self.worker.start()
        
    def _connect_signals(self):
        """Connect worker signals."""
        # Training signals
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_progress.connect(self.on_training_progress)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_error)
        
        # Inference signals
        self.worker.inference_started.connect(self.on_inference_started)
        self.worker.inference_result.connect(self.on_inference_result)
        self.worker.batch_inference_completed.connect(self.on_batch_inference_completed)
        self.worker.inference_error.connect(self.on_error)
        
        # Model signals
        self.worker.model_saved.connect(self.on_model_saved)
        self.worker.model_loaded.connect(self.on_model_loaded)
    
    @pyqtSlot()
    def on_training_started(self):
        """Handle training start."""
        logging.info("Training started...")
    
    @pyqtSlot(dict)
    def on_training_progress(self, info):
        """Handle training progress updates."""
        stage = info.get('stage', '')
        
        if stage == 'start':
            logging.info(f"Training starting with {info['total_epochs']} epochs")
        
        elif stage == 'epoch_start':
            logging.info(f"Epoch {info['current_epoch']}/{info['total_epochs']} starting")
        
        elif stage == 'batch':
            if 'loss' in info:
                logging.info(f"Epoch {info['current_epoch']}/{info['total_epochs']} - "
                           f"Batch {info['batch']}/{info['total_batches']} - "
                           f"Loss: {info['loss']:.4f}")
        
        elif stage == 'validation':
            logging.info(f"Epoch {info['current_epoch']}/{info['total_epochs']} - Validation")
            if 'metrics' in info:
                for metric, value in info['metrics'].items():
                    logging.info(f"  {metric}: {value:.4f}")
        
        elif stage == 'end':
            logging.info("Training finished!")
            if 'final_metrics' in info:
                logging.info("Final metrics:")
                for metric, value in info['final_metrics'].items():
                    logging.info(f"  {metric}: {value:.4f}")
    
    @pyqtSlot(dict)
    def on_training_completed(self, results):
        """Handle training completion."""
        logging.info(f"Training completed: {results}")
        
        if results['status'] == 'success':
            logging.info(f"Model trained successfully in {results['training_time']:.2f} seconds")
            if 'metrics' in results:
                for metric, value in results['metrics'].items():
                    logging.info(f"  {metric}: {value:.4f}")
    
    @pyqtSlot()
    def on_inference_started(self):
        """Handle inference start."""
        logging.info("Inference started...")
    
    @pyqtSlot(dict)
    def on_inference_result(self, result):
        """Handle inference result."""
        logging.info(f"Inference result:")
        logging.info(f"  Anomaly score: {result['anomaly_score']:.4f}")
        logging.info(f"  Is anomalous: {result['is_anomalous']}")
        logging.info(f"  Anomaly map shape: {result['anomaly_map'].shape}")
        
        if 'visualization' in result:
            logging.info("  Visualization images generated")
    
    @pyqtSlot(list)
    def on_batch_inference_completed(self, results):
        """Handle batch inference completion."""
        logging.info(f"Batch inference completed: {len(results)} images")
        
        # Analyze results
        anomalous_count = sum(1 for r in results if r['is_anomalous'])
        avg_score = np.mean([r['anomaly_score'] for r in results])
        
        logging.info(f"  Anomalous images: {anomalous_count}/{len(results)}")
        logging.info(f"  Average anomaly score: {avg_score:.4f}")
    
    @pyqtSlot(str)
    def on_model_saved(self, path):
        """Handle model save."""
        logging.info(f"Model saved to: {path}")
    
    @pyqtSlot(str)
    def on_model_loaded(self, model_type):
        """Handle model load."""
        logging.info(f"Model loaded: {model_type}")
    
    @pyqtSlot(str)
    def on_error(self, error_msg):
        """Handle errors."""
        logging.error(f"Error: {error_msg}")
    
    def train_on_good_images(self, data_path):
        """
        Train the model on good images.
        
        Args:
            data_path: Path to folder containing only good images
        """
        logging.info(f"Starting training on images from: {data_path}")
        
        self.worker.train_model(
            data_path=data_path,
            max_epochs=1,  # Quick training for demo
            batch_size=32,
            num_workers=4,
            val_split_ratio=0.1
        )
    
    def infer_single_image(self, image_path):
        """
        Perform inference on a single image.
        
        Args:
            image_path: Path to test image
        """
        # Load image
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logging.info(f"Running inference on: {image_path}")
        self.worker.infer_image(image, return_visualization=True)
    
    def infer_batch(self, image_folder):
        """
        Perform batch inference on a folder of images.
        
        Args:
            image_folder: Path to folder with test images
        """
        import cv2
        
        # Load all images
        image_paths = list(Path(image_folder).glob("*.jpg")) + \
                     list(Path(image_folder).glob("*.png"))
        
        if not image_paths:
            logging.error(f"No images found in: {image_folder}")
            return
        
        images = []
        for path in image_paths[:10]:  # Limit to 10 images for demo
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        logging.info(f"Running batch inference on {len(images)} images")
        self.worker.infer_batch(images)
    
    def save_model(self, save_path):
        """Save the trained model."""
        self.worker.save_model(save_path)
    
    def load_model(self, checkpoint_path):
        """Load a pre-trained model."""
        self.worker.load_model(checkpoint_path)


def create_synthetic_data(output_dir, num_images=50):
    """
    Create synthetic good images for testing.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
    """
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Creating {num_images} synthetic images in: {output_dir}")
    
    for i in range(num_images):
        # Create a simple synthetic image (e.g., a grid pattern)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        # Add grid pattern
        for y in range(0, 256, 32):
            cv2.line(img, (0, y), (255, y), (200, 200, 200), 1)
        for x in range(0, 256, 32):
            cv2.line(img, (x, 0), (x, 255), (200, 200, 200), 1)
        
        # Add some circles (normal pattern)
        for _ in range(5):
            x = np.random.randint(32, 224)
            y = np.random.randint(32, 224)
            cv2.circle(img, (x, y), 15, (100, 150, 100), -1)
        
        # Save image
        cv2.imwrite(str(output_dir / f"good_{i:04d}.png"), img)
    
    logging.info("Synthetic data created")


def create_anomalous_test_image(output_path):
    """Create a synthetic anomalous test image."""
    import cv2
    
    # Create similar to good images but with anomaly
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Add grid pattern
    for y in range(0, 256, 32):
        cv2.line(img, (0, y), (255, y), (200, 200, 200), 1)
    for x in range(0, 256, 32):
        cv2.line(img, (x, 0), (x, 255), (200, 200, 200), 1)
    
    # Add normal circles
    for _ in range(3):
        x = np.random.randint(32, 224)
        y = np.random.randint(32, 224)
        cv2.circle(img, (x, y), 15, (100, 150, 100), -1)
    
    # Add anomaly (red square)
    cv2.rectangle(img, (100, 100), (150, 150), (255, 50, 50), -1)
    
    cv2.imwrite(str(output_path), img)
    logging.info(f"Created anomalous test image: {output_path}")


def main():
    """Main demo function."""
    app = QApplication(sys.argv)
    
    # Create demo
    demo = AnomalyDetectionDemo(model_type="patchcore")
    
    # Create synthetic data
    good_images_dir = Path("synthetic_data/good")
    create_synthetic_data(good_images_dir, num_images=20)
    
    # Create test image
    test_image_path = Path("synthetic_data/test_anomaly.png")
    test_image_path.parent.mkdir(exist_ok=True)
    create_anomalous_test_image(test_image_path)
    
    # Run training
    demo.train_on_good_images(good_images_dir)
    
    # Wait for training to complete
    app.processEvents()
    import time
    time.sleep(5)  # Give time for training
    
    # Run inference
    demo.infer_single_image(test_image_path)
    
    # Save model
    demo.save_model("anomaly_model.ckpt")
    
    # Run application
    try:
        app.exec()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        demo.worker.stop()


if __name__ == "__main__":
    main()