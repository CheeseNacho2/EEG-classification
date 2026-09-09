import sys
import os
import requests
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QFrame, QMessageBox, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QMovie
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def get_base_dir():
    """Returns the base project root directory in both dev and PyInstaller modes."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    # In dev mode, app.py is inside desktop/, so we move one directory up to get project root
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

BASE_DIR = get_base_dir()

# Correctly resolves project_root/desktop/assets across both environments
ASSETS_DIR = os.path.join(BASE_DIR, 'desktop', 'assets')

API_URL = "http://localhost:8000"
HTTP_TIMEOUT = (3.0, 60.0)


# ── Worker Threads ────────────────────────────────────────────────
class PredictWorker(QThread):
    finished = pyqtSignal(dict, dict)
    error    = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            with open(self.filepath, 'rb') as f:
                filename = os.path.basename(self.filepath)
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (filename, f, "application/octet-stream")},
                    timeout=HTTP_TIMEOUT
                )

            if response.status_code != 200:
                self.error.emit(response.json().get("detail", f"Prediction failed (Status {response.status_code})"))
                return

            pred_data = response.json()

            imp_data = {}
            try:
                imp_resp = requests.get(f"{API_URL}/feature-importance", timeout=HTTP_TIMEOUT)
                if imp_resp.status_code == 200:
                    imp_data = imp_resp.json()
            except Exception:
                pass

            self.finished.emit(pred_data, imp_data)

        except requests.exceptions.Timeout:
            self.error.emit("Connection timed out while attempting to reach the server.")
        except requests.exceptions.ConnectionError:
            self.error.emit("Could not connect to the API server. Ensure the server is running.")
        except Exception as e:
            self.error.emit(str(e))


class RetrainWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, filepath, label):
        super().__init__()
        self.filepath = filepath
        self.label    = label

    def run(self):
        try:
            with open(self.filepath, 'rb') as f:
                filename = os.path.basename(self.filepath)
                response = requests.post(
                    f"{API_URL}/retrain",
                    files={"file": (filename, f, "application/octet-stream")},
                    data={"label": str(self.label)},
                    timeout=HTTP_TIMEOUT
                )
            if response.status_code == 200:
                self.finished.emit(response.json())
            else:
                self.error.emit(response.json().get("detail", f"Retraining failed (Status {response.status_code})"))
        except requests.exceptions.Timeout:
            self.error.emit("Connection timed out during model retraining.")
        except requests.exceptions.ConnectionError:
            self.error.emit("Could not connect to the API server. Ensure the server is running.")
        except Exception as e:
            self.error.emit(str(e))


# ── UI Components ─────────────────────────────────────────────────
class BrainBackground(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_original = None
        
        brain_path = os.path.join(ASSETS_DIR, "brain.png")
        if os.path.exists(brain_path):
            self.pixmap_original = QPixmap(brain_path)
            if self.pixmap_original.isNull():
                self.pixmap_original = None
            else:
                self.setPixmap(self.pixmap_original.scaled(
                    900, 800,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.12) 
        self.setGraphicsEffect(effect)
        self.lower()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.x_fine = None
        self.y_fine = None
        self.ax     = None
        self.annot  = None
        self.mpl_connect('motion_notify_event', self.on_hover)

    def plot_epochs(self, epoch_probs):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        epoch_probs  = np.array(epoch_probs)
        x            = np.arange(len(epoch_probs))
        self.x_fine  = np.linspace(0, len(epoch_probs) - 1, len(epoch_probs) * 10)
        self.y_fine  = np.interp(self.x_fine, x, epoch_probs)

        self.ax.plot(self.x_fine, self.y_fine, color='gray', linewidth=1, zorder=1)
        self.ax.axhline(0.5, color='black', linestyle='--', linewidth=1, zorder=2)

        self.ax.fill_between(self.x_fine, self.y_fine, 0.5,
                             where=(self.y_fine < 0.5),
                             color='green', alpha=0.4, label='Calm')

        self.ax.fill_between(self.x_fine, self.y_fine, 0.5,
                             where=(self.y_fine >= 0.5),
                             color='red', alpha=0.4, label='Stress')

        self.ax.set_title("Stress Probability Per Epoch")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 1)
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
        self.fig.tight_layout(rect=[0, 0, 0.85, 1])

        self.annot = self.ax.annotate(
            "", xy=(0, 0),
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
            fontsize=9
        )
        self.annot.set_visible(False)
        self.draw()

    def on_hover(self, event):
        if event.inaxes != self.ax or self.x_fine is None or len(self.x_fine) == 0:
            if self.annot:
                self.annot.set_visible(False)
                self.draw_idle()
            return

        idx       = np.searchsorted(self.x_fine, event.xdata)
        idx       = np.clip(idx, 0, len(self.x_fine) - 1)
        prob      = self.y_fine[idx]
        epoch_num = int(self.x_fine[idx])
        label     = "Stress" if prob >= 0.5 else "Calm"
        time_sec  = epoch_num

        self.annot.xy = (self.x_fine[idx], prob)
        self.annot.set_text(
            f"Epoch: {epoch_num}\n"
            f"Time:  {time_sec}s\n"
            f"State: {label}\n"
            f"Prob:  {prob:.2f}"
        )
        color = "red" if label == "Stress" else "green"
        self.annot.get_bbox_patch().set_edgecolor(color)
        self.annot.set_visible(True)
        self.draw_idle()

    def plot_importance(self, names, scores):
        names  = names[:10]
        scores = scores[:10]

        band_symbols = {'theta': 'θ', 'alpha': 'α', 'beta': 'β'}
        short_names  = []
        for name in names:
            name = name.replace('EEG ', '')
            for band, symbol in band_symbols.items():
                name = name.replace(f'_{band}', f' {symbol}')
            short_names.append(name)

        colors = ['tomato' if i < 3 else 'steelblue' for i in range(len(scores))]

        self.fig.clear()
        ax    = self.fig.add_subplot(111)
        y_pos = np.arange(len(short_names))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_title("Top 10 Feature Importance")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        self.draw()


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Stress Classifier")
        self.setFixedSize(900, 1000)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame = QFrame(self)
        frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 20px;
            }
        """)
        frame.setFixedSize(self.width() - 60, self.height() - 60)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(30, 30, 30, 30)
        frame_layout.setSpacing(12)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("EEG Stress Classifier")
        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #1e1e2e;")
        frame_layout.addWidget(title)

        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setFixedSize(400, 400)

        gif_path = os.path.join(ASSETS_DIR, "splash.gif")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(self.gif_label.size())
            self.gif_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.gif_label.setText("EEG")
            self.gif_label.setStyleSheet("font-size: 48px; color: #4f46e5;")

        gif_row = QHBoxLayout()
        gif_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_row.addWidget(self.gif_label)
        frame_layout.addLayout(gif_row)

        version = QLabel("Version 1.0.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #6b7280; font-size: 13px;")
        frame_layout.addWidget(version)

        license = QLabel("MIT License — © 2026")
        license.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license.setStyleSheet("color: #9ca3af; font-size: 11px;")
        frame_layout.addWidget(license)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet("""
            QProgressBar {
                background: #e0e7ff;
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background: #4f46e5;
                border-radius: 3px;
            }
        """)
        frame_layout.addWidget(self.progress)

        self.loading_label = QLabel("Initialising...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        frame_layout.addWidget(self.loading_label)

        layout.addWidget(frame)

    def set_loading_text(self, text):
        self.loading_label.setText(text)

    def set_progress(self, value):
        self.progress.setValue(value)


# ── Main Window ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, skip_api_check=False):
        super().__init__()
        self.setWindowTitle("EEG Stress Classifier")
        self.setMinimumSize(900, 800)
        self.resize(900, 800)
        self.filepath = None
        self.result   = None
        self.worker   = None
        self.setup_ui()

        if not skip_api_check:
            self.check_api()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'brain_bg'):
            central = self.centralWidget()
            self.brain_bg.setGeometry(0, 0, central.width(), central.height())
            if self.brain_bg.pixmap_original is not None:
                self.brain_bg.setPixmap(self.brain_bg.pixmap_original.scaled(
                    central.width(),
                    central.height(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setSpacing(12)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.brain_bg = BrainBackground(central)
        self.brain_bg.setGeometry(0, 0, central.width(), central.height())

        self.setStyleSheet("""
            QPushButton {
                background-color: #0d9488;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
                font-family: Arial;
            }
            QPushButton:hover {
                background-color: #0f766e;
            }
            QPushButton:pressed {
                background-color: #115e59;
            }
            QPushButton:disabled {
                background-color: #99f6e4;
                color: #ffffff;
            }
            QProgressBar {
                background: #e0f2f1;
                border-radius: 4px;
                border: none;
                height: 6px;
            }
            QProgressBar::chunk {
                background: #0d9488;
                border-radius: 4px;
            }
        """)

        self.banner = QLabel("")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.main_layout.addWidget(self.banner)

        title = QLabel("EEG Stress Classifier")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title)

        self.upload_btn = QPushButton("Upload EEG File (.edf)")
        self.upload_btn.clicked.connect(self.upload_file)
        self.main_layout.addWidget(self.upload_btn)

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.file_label)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        self.main_layout.addWidget(self.predict_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.main_layout.addWidget(self.progress)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        self.epoch_canvas      = PlotCanvas()
        self.importance_canvas = PlotCanvas()
        self.main_layout.addWidget(self.epoch_canvas)
        self.main_layout.addWidget(self.importance_canvas)
        self.epoch_canvas.setVisible(False)
        self.importance_canvas.setVisible(False)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        self.main_layout.addWidget(line)

        self.retrain_question = QLabel("Was this prediction correct?")
        self.retrain_question.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.retrain_question.setVisible(False)
        self.main_layout.addWidget(self.retrain_question)

        self.retrain_subtitle = QLabel("Your feedback helps improve the model")
        self.retrain_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.retrain_subtitle.setStyleSheet("color: gray; font-size: 12px;")
        self.retrain_subtitle.setVisible(False)
        self.main_layout.addWidget(self.retrain_subtitle)

        retrain_row = QHBoxLayout()
        self.yes_btn = QPushButton("✓ Yes")
        self.no_btn  = QPushButton("✗ No")
        self.yes_btn.clicked.connect(self.on_correct)
        self.no_btn.clicked.connect(self.on_incorrect)
        self.yes_btn.setVisible(False)
        self.no_btn.setVisible(False)
        retrain_row.addWidget(self.yes_btn)
        retrain_row.addWidget(self.no_btn)
        self.main_layout.addLayout(retrain_row)

        self.retrain_label = QLabel("")
        self.retrain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.retrain_label.setVisible(False)
        self.main_layout.addWidget(self.retrain_label)

    def check_api(self):
        try:
            response = requests.get(f"{API_URL}/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if not data.get('model_loaded'):
                    self.show_banner(
                        "Warning: Model is not loaded. Please run train.py first.",
                        color="orange"
                    )
            else:
                self.show_banner("Warning: API returned unexpected status.", color="orange")
        except requests.exceptions.ConnectionError:
            self.show_banner("Cannot connect to API. Please start api.py before using the app.", color="red")
        except requests.exceptions.Timeout:
            self.show_banner("API connection timed out. Please check that api.py is running.", color="red")
        except Exception as e:
            self.show_banner(f"Unexpected error checking API: {e}", color="red")

    def show_banner(self, message, color="red"):
        bg = "#fee2e2" if color == "red" else "#fef9c3" if color == "orange" else "#dcfce7"
        fg = "#dc2626" if color == "red" else "#92400e" if color == "orange" else "#16a34a"
        self.banner.setText(message)
        self.banner.setStyleSheet(f"""
            padding: 10px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            background: {bg};
            color: {fg};
        """)
        self.banner.setVisible(True)

    def hide_banner(self):
        self.banner.setVisible(False)

    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select EEG File", "", "EDF Files (*.edf)"
        )
        if not path:
            return

        if not path.lower().endswith('.edf'):
            self.show_banner("Please select a valid .edf file.", color="red")
            return

        if os.path.getsize(path) == 0:
            self.show_banner("The selected file is empty.", color="red")
            return

        self.filepath = path
        self.file_label.setText(f"Selected: {os.path.basename(path)}")
        self.predict_btn.setEnabled(True)
        self.hide_banner()

        self.result_label.setText("")
        self.result_label.setStyleSheet("")
        self.epoch_canvas.setVisible(False)
        self.importance_canvas.setVisible(False)
        self.yes_btn.setVisible(False)
        self.no_btn.setVisible(False)
        self.retrain_question.setVisible(False)
        self.retrain_subtitle.setVisible(False)
        self.retrain_label.setText("")
        self.retrain_label.setStyleSheet("")

    def run_prediction(self):
        self.predict_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.result_label.setText("Analyzing...")

        self.worker = PredictWorker(self.filepath)
        self.worker.finished.connect(self.on_prediction_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_prediction_done(self, result, importance_data):
        self.result = result
        self.progress.setVisible(False)
        self.predict_btn.setEnabled(True)

        label = result.get('label', 'Unknown')
        prob  = result.get('probability', 0.0) * 100
        color = "red" if label == "Stress" else "green"
        self.result_label.setText(f"{label} — {prob:.1f}%")
        self.result_label.setStyleSheet(f"color: {color};")

        if 'per_epoch' in result:
            self.epoch_canvas.plot_epochs(result['per_epoch'])
            self.epoch_canvas.setVisible(True)

        if 'features' in importance_data:
            data   = importance_data['features']
            names  = [d['name'] for d in data]
            scores = [d['importance'] for d in data]
            self.importance_canvas.plot_importance(names, scores)
            self.importance_canvas.setVisible(True)

        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        self.yes_btn.setVisible(True)
        self.no_btn.setVisible(True)
        self.retrain_question.setVisible(True)
        self.retrain_subtitle.setVisible(True)
        self.retrain_label.setVisible(True)
        self.retrain_label.setText("")
        self.retrain_label.setStyleSheet("")

    def on_correct(self):
        self.yes_btn.setVisible(False)
        self.no_btn.setVisible(False)
        self.retrain_question.setVisible(False)
        self.retrain_subtitle.setVisible(False)
        self.retrain_label.setText("✓ Thank you for your feedback!")
        self.retrain_label.setStyleSheet("color: green;")

    def on_incorrect(self):
        current_label = self.result['label']
        correct_label = 0 if current_label == 'Stress' else 1
        self.run_retrain(correct_label)

    def run_retrain(self, label):
        self.yes_btn.setEnabled(False)
        self.no_btn.setEnabled(False)
        self.retrain_label.setText("Retraining...")
        self.progress.setVisible(True)

        self.retrain_worker = RetrainWorker(self.filepath, label)
        self.retrain_worker.finished.connect(self.on_retrain_done)
        self.retrain_worker.error.connect(self.on_error)
        self.retrain_worker.start()

    def on_retrain_done(self, result):
        self.progress.setVisible(False)
        self.yes_btn.setVisible(False)
        self.no_btn.setVisible(False)
        self.retrain_question.setVisible(False)
        self.retrain_subtitle.setVisible(False)
        self.retrain_label.setText("✓ Thank you for your feedback!")
        self.retrain_label.setStyleSheet("color: green;")

    def on_error(self, message):
        self.progress.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.retrain_label.setText("")
        self.result_label.setText("Request Failed")
        self.result_label.setStyleSheet("color: red;")

        QMessageBox.critical(
            self,
            "Network Error",
            f"An error occurred while communicating with the server:\n\n{message}",
            QMessageBox.StandardButton.Ok
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()

    screen = app.primaryScreen().geometry()
    splash.move(
        (screen.width()  - splash.width())  // 2,
        (screen.height() - splash.height()) // 2
    )

    progress_value = [0]

    def tick_progress():
        if progress_value[0] < 90:
            progress_value[0] += 1
            splash.set_progress(progress_value[0])

            # Show friendly messages at specific progress points
            if progress_value[0] == 20:
                splash.set_loading_text("Initialising...")
            elif progress_value[0] == 40:
                splash.set_loading_text("Loading components...")
            elif progress_value[0] == 60:
                splash.set_loading_text("Loading model...")
            elif progress_value[0] == 80:
                splash.set_loading_text("Almost ready...")

    progress_timer = QTimer()
    progress_timer.timeout.connect(tick_progress)
    progress_timer.start(50)

    
    def launch():
        splash.set_loading_text("Ready!")
        QTimer.singleShot(500, lambda: (
            splash.close(),
            window.show()
        ))

    window = MainWindow()
    QTimer.singleShot(3000, launch)

    sys.exit(app.exec())