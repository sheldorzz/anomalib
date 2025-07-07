# run_capture.py
import sys
import json
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QThread
from workers.capture import CaptureWorker

def main():
    # 1) Create a headless Qt app
    app = QCoreApplication(sys.argv)

    # 2) Define a minimal waypoint config.
    #    You can also load this from a JSON file if you prefer.
    waypoints_config = {
        "home": {
            "position": [0.0, 0.0, 0.2, 0, 3.14, 0],
            "speed": 0.5,
            "acceleration": 0.5,
            "async": False
        },
        "waypoints": {
            "wp1": {
                "position": [0.1, 0.2, 0.3, 0, 3.14, 0],
                "description": "Point 1"
            },
            "wp2": {
                "position": [0.2, 0.1, 0.3, 0, 3.14, 0],
                "description": "Point 2"
            }
        }
    }
    # (Or: waypoints_config = json.load(open("my_waypoints.json")))

    # 3) Instantiate your worker
    worker = CaptureWorker(waypoints_config, robot_ip="192.168.1.10")

    # 4) Hook up signals for basic console feedback
    worker.started.connect(lambda: print("[+] Capture started"))
    worker.progress.connect(lambda p: print(f"[+] Progress: {p}%"))
    worker.frame_captured.connect(
        lambda frame: print(f"[+] Frame @ wp{frame['waypoint_id']}: pose={frame['pose']}")
    )
    worker.error.connect(lambda err: print(f"[!] ERROR: {err}"))
    worker.finished.connect(lambda: (print("[+] Capture finished"), app.quit()))

    # 5) Run it in its own thread
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start_capture)
    thread.start()

    # 6) Enter the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
