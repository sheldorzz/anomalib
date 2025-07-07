# run_infer.py
import sys
import cv2
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QThread, QObject, QTimer
from workers.infer import InferWorker


def main():
    # 1) Create a headless Qt application
    app = QCoreApplication(sys.argv)

    # 2) Instantiate the inference worker
    worker = InferWorker()

    # 3) Connect signals to console output
    worker.started.connect(lambda: print("[+] Inference started"))
    worker.anomaly_detected.connect(
        lambda data: print(
            f"[!] Anomaly detected: score={data['image_score']:.3f}, regions={len(data['anomaly_regions'])}"
        )
    )
    worker.error.connect(lambda err: print(f"[!] ERROR: {err}"))
    worker.finished.connect(lambda: (print("[+] Inference finished"), app.quit()))

    # 4) Move worker to a separate thread and start inference
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start_inference)
    thread.start()

    # 5) After model is loaded, stream test frames
    class FrameFeeder(QObject):
        def __init__(self, images, target_worker, interval=100):  # interval in ms
            super().__init__()
            self.images = images
            self.worker = target_worker
            self.index = 0
            self.timer = QTimer(self)
            self.timer.setInterval(interval)
            self.timer.timeout.connect(self.feed_next)

        def start(self):
            self.timer.start()

        def feed_next(self):
            if self.index >= len(self.images):
                # No more frames: stop feeding and shutdown
                self.timer.stop()
                self.worker.stop_inference()
                return
            img_path = self.images[self.index]
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[!] Failed to load frame: {img_path}")
            else:
                # Send frame to inference
                self.worker.process_frame(frame)
            self.index += 1

    # Gather test images
    test_dir = Path("data/test")
    image_paths = sorted(test_dir.glob("*.png"))
    if not image_paths:
        print(f"[!] No test images found in: {test_dir}")
        # wait until engine loads and then quit
        worker.finished.connect(app.quit)
    else:
        feeder = FrameFeeder(image_paths, worker, interval=200)
        # Start feeding once inference is ready
        worker.started.connect(feeder.start)

    # 6) Enter the Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
