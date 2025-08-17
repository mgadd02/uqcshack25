from typing import Callable, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2

from modules.solver_engine import solve_from_bgr

class ProcessTab(QtWidgets.QWidget):
    """
    Button-triggered processing (no background runs).
    """
    def __init__(self, get_frame_func: Callable[[], Optional[np.ndarray]],
                 set_output_func: Callable[[str], None],
                 get_save_dir_func: Callable[[], str],
                 parent=None):
        super().__init__(parent)
        self.get_frame = get_frame_func
        self.set_output = set_output_func
        self.get_save_dir = get_save_dir_func

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.process_btn = QtWidgets.QPushButton("Process Latest Frame")
        self.process_btn.clicked.connect(self.run_processing)
        controls.addWidget(self.process_btn)

        controls.addSpacing(12)
        self.xmin = QtWidgets.QDoubleSpinBox(); self.xmin.setRange(-1000, 1000); self.xmin.setValue(-25.0)
        self.xmax = QtWidgets.QDoubleSpinBox(); self.xmax.setRange(-1000, 1000); self.xmax.setValue(25.0)
        self.ymin = QtWidgets.QDoubleSpinBox(); self.ymin.setRange(-1000, 1000); self.ymin.setValue(-15.0)
        self.ymax = QtWidgets.QDoubleSpinBox(); self.ymax.setRange(-1000, 1000); self.ymax.setValue(15.0)
        self.min_area = QtWidgets.QSpinBox(); self.min_area.setRange(1, 10000); self.min_area.setValue(60)

        controls.addWidget(QtWidgets.QLabel("X Range:"))
        controls.addWidget(self.xmin); controls.addWidget(self.xmax)
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Y Range:"))
        controls.addWidget(self.ymin); controls.addWidget(self.ymax)
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Min Actor Area:"))
        controls.addWidget(self.min_area)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Previews
        panels = QtWidgets.QHBoxLayout()
        self.orig_label = QtWidgets.QLabel("(latest frame)")
        self.orig_label.setAlignment(QtCore.Qt.AlignCenter)
        self.orig_label.setMinimumHeight(280)
        panels.addWidget(self.orig_label, 1)

        self.overlay_label = QtWidgets.QLabel("(overlay)")
        self.overlay_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_label.setMinimumHeight(280)
        panels.addWidget(self.overlay_label, 1)
        layout.addLayout(panels, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#666")
        layout.addWidget(self.status)

    def _show_np_rgb(self, img: np.ndarray, label: QtWidgets.QLabel):
        if img is None:
            label.setText("(none)")
            return
        # img is RGB here
        h, w = img.shape[:2]
        pw, ph = max(200, label.width()), max(200, label.height())
        scale = min(pw / w, ph / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def run_processing(self):
        rgb = self.get_frame()
        if rgb is None:
            QtWidgets.QMessageBox.information(self, "No frame", "No frame available. Start capture first.")
            return

        # Show original
        self._show_np_rgb(rgb, self.orig_label)

        # Convert to BGR for OpenCV pipeline
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Build overlay path in working dir
        save_dir = self.get_save_dir() or "."
        overlay_path = f"{save_dir}/graphwar_overlay.png"

        try:
            result = solve_from_bgr(
                bgr,
                x_range=(float(self.xmin.value()), float(self.xmax.value())),
                y_range=(float(self.ymin.value()), float(self.ymax.value())),
                min_area=int(self.min_area.value()),
                overlay_path=overlay_path  # also returns overlay as ndarray
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Solver error", str(e))
            return

        expr = result.get("expr", "0")
        overlay_bgr = result.get("overlay_bgr")
        if overlay_bgr is not None:
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            self._show_np_rgb(overlay_rgb, self.overlay_label)

        # Push expression to Shot Output tab
        self.set_output(expr)

        self.status.setText(f"Solved. Expression length: {len(expr)} | Overlay saved: {overlay_path}")
