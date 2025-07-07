# run_train.py
import sys
from PySide6.QtCore import QCoreApplication, QThread
from workers.train import TrainWorker

def main():
    # 1) Create a headless Qt application
    app = QCoreApplication(sys.argv)

    # 2) Instantiate the training worker
    worker = TrainWorker()

    # 3) Connect signals to console output
    worker.started.connect(lambda: print("[+] Training started"))
    worker.progress.connect(lambda p: print(f"[+] Progress: {p}%"))
    worker.result.connect(lambda res: print(f"[+] Training result: {res}"))
    worker.error.connect(lambda err: print(f"[!] ERROR: {err}"))
    worker.finished.connect(lambda: (print("[+] Training finished"), app.quit()))

    # 4) Move worker to a separate thread and kick off training
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start_training)
    thread.start()

    # 5) Enter the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
