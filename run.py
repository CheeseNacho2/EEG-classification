import sys
import os
import time
import subprocess
import atexit
import multiprocessing
import requests
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

# ── Critical for PyInstaller on Windows ──────────────────────────
multiprocessing.freeze_support()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desktop.app import MainWindow, SplashScreen

# ── Configuration ─────────────────────────────────────────────────
API_URL = "http://localhost:8000"
API_RETRIES = 40  # Accounts for backend boot & model loading time


# ── API Launcher Thread ───────────────────────────────────────────
class APILaunchWorker(QThread):
    ready = pyqtSignal()
    failed = pyqtSignal()
    status = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, api_process):
        super().__init__()
        self.api_process = api_process

    def run(self):
        for attempt in range(API_RETRIES):
            # Scale progress from 0% to 90% during polling
            progress_value = int((attempt / API_RETRIES) * 90)
            self.progress.emit(progress_value)

            try:
                response = requests.get(f"{API_URL}/health", timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('model_loaded'):
                        self.status.emit("Ready!")
                        self.progress.emit(100)
                        self.ready.emit()
                        return
                    else:
                        self.status.emit("Loading model...")
            except Exception:
                if attempt < 5:
                    self.status.emit("Initialising...")
                elif attempt < 15:
                    self.status.emit("Loading components...")
                else:
                    self.status.emit("Almost ready...")

            time.sleep(0.5)

        self.failed.emit()


# ── Subprocess Management ─────────────────────────────────────────
def start_api():
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
        process = subprocess.Popen(
            [exe_path, '--api-mode'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")
        process = subprocess.Popen(
            [sys.executable, api_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    return process


def cleanup_api(process):
    """Guarantees process shutdown across Windows and UNIX platforms."""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()


# ── Main Application Entry ────────────────────────────────────────
def main():
    app = QApplication(sys.argv)

    # Show Splash Screen
    splash = SplashScreen()
    splash.show()

    # Center splash on screen
    screen = app.primaryScreen().geometry()
    splash.move(
        (screen.width() - splash.width()) // 2,
        (screen.height() - splash.height()) // 2
    )

    # Start API Subprocess & register guaranteed exit cleanup
    api_process = start_api()
    atexit.register(cleanup_api, api_process)

    # Launch Worker Thread
    worker = APILaunchWorker(api_process)
    worker.status.connect(splash.set_loading_text)
    worker.progress.connect(splash.set_progress)

    def on_ready():
        window = MainWindow(skip_api_check=True)
        QTimer.singleShot(400, lambda: (
            splash.close(),
            window.show()
        ))
        # Retain reference on app to prevent garbage collection
        app._window = window

    def on_failed():
        splash.set_loading_text("Failed to start API. Exiting...")
        QTimer.singleShot(2000, lambda: (
            cleanup_api(api_process),
            sys.exit(1)
        ))

    # Connect signals & thread lifecycle handlers
    worker.ready.connect(on_ready)
    worker.failed.connect(on_failed)
    worker.finished.connect(worker.deleteLater)
    worker.start()

    # Run Event Loop
    exit_code = app.exec()

    # Shutdown Subprocess
    cleanup_api(api_process)
    sys.exit(exit_code)


if __name__ == "__main__":
    if '--api-mode' in sys.argv:
        import uvicorn
        import api
        uvicorn.run(api.app, host="0.0.0.0", port=8000)
        sys.exit(0)

    main()