from typing import Callable, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2

from modules.processing import extract_play_area, detect_players
from modules.solver import solve_path

class ProcessTab(QtWidgets.QWidget):
    """
    Button-triggered processing (no background runs).
    """
    def __init__(self, get_frame_func: Callable[[], Optional[np.ndarray]],
                 set_output_func: Callable[[str], None],
                 parent=None):
        super().__init__(parent)
        self.get_frame = get_frame_func
        self.set_output = set_output_func

        layout = QtWidgets.QVBoxLayout(self)

        self.process_btn = QtWidgets.QPushButton("Process Latest Frame")
        self.process_btn.clicked.connect(self.run_processing)
        layout.addWidget(self.process_btn, 0)

        # Preview panels
        panels = QtWidgets.QHBoxLayout()
        self.orig_label = QtWidgets.QLabel("(original frame)")
        self.orig_label.setAlignment(QtCore.Qt.AlignCenter)
        self.orig_label.setMinimumHeight(240)
        panels.addWidget(self.orig_label, 1)

        self.play_label = QtWidgets.QLabel("(play area)")
        self.play_label.setAlignment(QtCore.Qt.AlignCenter)
        self.play_label.setMinimumHeight(240)
        panels.addWidget(self.play_label, 1)
        layout.addLayout(panels, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#666")
        layout.addWidget(self.status)

    def _show_np_on_label(self, img: np.ndarray, label: QtWidgets.QLabel):
        if img is None:
            label.setText("(none)")
            return
        h, w = img.shape[:2]
        pw, ph = max(200, label.width()), max(200, label.height())
        scale = min(pw / w, ph / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def run_processing(self):
        frame = self.get_frame()
        if frame is None:
            QtWidgets.QMessageBox.information(self, "No frame", "No frame available. Start capture first.")
            return

        # Show original
        self._show_np_on_label(frame, self.orig_label)

        # Extract play area
        play, rect = extract_play_area(frame)
        self._show_np_on_label(play, self.play_label)

        # Detect players (very simple placeholder)
        players = detect_players(play)

        # Solve (placeholder A*/grid to “function” text)
        command_text = solve_path(play, players)

        # Send to shot output tab
        self.set_output(command_text)

        # Status
        self.status.setText(f"Play area rect: {rect} | left={len(players['left_team'])} right={len(players['right_team'])}")
