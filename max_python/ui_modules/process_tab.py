# ui_modules/process_tab.py
from typing import Callable, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2

from modules.solver_engine import solve_from_bgr, DEFAULT_BORDER_PX, DEFAULT_INFLATE_MAIN, DEFAULT_INFLATE_TOL, DEFAULT_MIN_OBS_AREA, DEFAULT_MIN_THICK_PX, DEFAULT_SHRINK_STEPS

class ProcessTab(QtWidgets.QWidget):
    """
    Cancellable processing with full controls.
    Shows the overlay and the critical grid (no raw screenshot).
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

        # --- Controls row 1: ranges, area, grid
        row1 = QtWidgets.QHBoxLayout()
        self.process_btn = QtWidgets.QPushButton("Process Latest Frame")
        self.process_btn.clicked.connect(self.run_processing)
        row1.addWidget(self.process_btn)

        row1.addSpacing(12)
        self.xmin = QtWidgets.QDoubleSpinBox(); self.xmin.setRange(-1000, 1000); self.xmin.setValue(-25.0)
        self.xmax = QtWidgets.QDoubleSpinBox(); self.xmax.setRange(-1000, 1000); self.xmax.setValue(25.0)
        self.ymin = QtWidgets.QDoubleSpinBox(); self.ymin.setRange(-1000, 1000); self.ymin.setValue(-15.0)
        self.ymax = QtWidgets.QDoubleSpinBox(); self.ymax.setRange(-1000, 1000); self.ymax.setValue(15.0)
        self.min_area = QtWidgets.QSpinBox(); self.min_area.setRange(1, 10000); self.min_area.setValue(60)
        self.grid_dx  = QtWidgets.QDoubleSpinBox(); self.grid_dx.setRange(0.05, 2.0); self.grid_dx.setSingleStep(0.05); self.grid_dx.setValue(0.25)
        self.grid_dy  = QtWidgets.QDoubleSpinBox(); self.grid_dy.setRange(0.05, 2.0); self.grid_dy.setSingleStep(0.05); self.grid_dy.setValue(0.25)

        row1.addWidget(QtWidgets.QLabel("X Range:")); row1.addWidget(self.xmin); row1.addWidget(self.xmax)
        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Y Range:")); row1.addWidget(self.ymin); row1.addWidget(self.ymax)
        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Min Actor Area:")); row1.addWidget(self.min_area)
        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Grid Δx/Δy:")); row1.addWidget(self.grid_dx); row1.addWidget(self.grid_dy)
        row1.addStretch(1)
        layout.addLayout(row1)

        # --- Controls row 2: padding / tolerance / filters
        row2 = QtWidgets.QHBoxLayout()
        self.border_px  = QtWidgets.QSpinBox(); self.border_px.setRange(0, 60); self.border_px.setValue(DEFAULT_BORDER_PX)
        self.pad_main   = QtWidgets.QSpinBox(); self.pad_main.setRange(0, 80); self.pad_main.setValue(DEFAULT_INFLATE_MAIN)
        self.pad_tol    = QtWidgets.QSpinBox(); self.pad_tol.setRange(0, 120); self.pad_tol.setValue(DEFAULT_INFLATE_TOL)
        self.min_obs_area = QtWidgets.QSpinBox(); self.min_obs_area.setRange(0, 4000); self.min_obs_area.setValue(DEFAULT_MIN_OBS_AREA)
        self.min_thick  = QtWidgets.QDoubleSpinBox(); self.min_thick.setRange(0.0, 50.0); self.min_thick.setDecimals(1); self.min_thick.setValue(DEFAULT_MIN_THICK_PX)
        self.shrink_steps = QtWidgets.QSpinBox(); self.shrink_steps.setRange(0, 30); self.shrink_steps.setValue(DEFAULT_SHRINK_STEPS)

        row2.addWidget(QtWidgets.QLabel("Border px:")); row2.addWidget(self.border_px)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Obstacle pad px:")); row2.addWidget(self.pad_main)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Tolerance px:")); row2.addWidget(self.pad_tol)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Min obs area:")); row2.addWidget(self.min_obs_area)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Min thickness px:")); row2.addWidget(self.min_thick)
        row2.addSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Shrink steps:")); row2.addWidget(self.shrink_steps)
        row2.addStretch(1)
        layout.addLayout(row2)

        # --- Controls row 3: algo, targeting, fallback, timeout
        row3 = QtWidgets.QHBoxLayout()
        self.algo = QtWidgets.QComboBox()
        self.algo.addItems(["A*", "Dijkstra", "Greedy", "BFS", "WorstA*", "ZigZag", "RandomGreedy", "WallHugger"])
        self.algo.setCurrentText("A*")

        self.targeting = QtWidgets.QComboBox()
        self.targeting.addItems(["X-first (columns)", "Nearest"])
        self.targeting.setCurrentIndex(0)

        self.allow_soft = QtWidgets.QCheckBox("Allow soft fallback (only if no enemy hard-reachable)")
        self.allow_soft.setChecked(True)

        self.timeout_s = QtWidgets.QDoubleSpinBox(); self.timeout_s.setRange(0.2, 5.0); self.timeout_s.setDecimals(1); self.timeout_s.setSingleStep(0.1); self.timeout_s.setValue(1.6)

        row3.addWidget(QtWidgets.QLabel("Algo:")); row3.addWidget(self.algo)
        row3.addSpacing(8)
        row3.addWidget(QtWidgets.QLabel("Targeting:")); row3.addWidget(self.targeting)
        row3.addSpacing(8)
        row3.addWidget(self.allow_soft)
        row3.addSpacing(8)
        row3.addWidget(QtWidgets.QLabel("Timeout (s):")); row3.addWidget(self.timeout_s)
        row3.addStretch(1)
        layout.addLayout(row3)

        # --- Previews: overlay + critical grid, large & autoscaled
        panels = QtWidgets.QHBoxLayout()
        self.overlay_label = QtWidgets.QLabel("(overlay)")
        self.overlay_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_label.setMinimumHeight(520)
        self.overlay_label.setMinimumWidth(720)
        panels.addWidget(self.overlay_label, 1)

        self.crit_label = QtWidgets.QLabel("(critical grid)")
        self.crit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.crit_label.setMinimumHeight(520)
        self.crit_label.setMinimumWidth(720)
        panels.addWidget(self.crit_label, 1)

        layout.addLayout(panels, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#666")
        layout.addWidget(self.status)

    def _show_np_rgb(self, img: np.ndarray, label: QtWidgets.QLabel):
        if img is None or img.size == 0:
            label.setText("(none)")
            return
        h, w = img.shape[:2]
        pw, ph = max(240, label.width()), max(240, label.height())
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

        save_dir = self.get_save_dir() or "."
        overlay_path = f"{save_dir}/graphwar_overlay.png"

        # Progress dialog
        prog = QtWidgets.QProgressDialog("Solving...", "Cancel", 0, 100, self)
        prog.setWindowTitle("Pathfinding")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def progress_cb(phase: str, percent: int, note: str) -> bool:
            prog.setLabelText(f"{phase} — {note}")
            prog.setValue(max(0, min(100, percent)))
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            return not prog.wasCanceled()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        try:
            result = solve_from_bgr(
                bgr,
                x_range=(float(self.xmin.value()), float(self.xmax.value())),
                y_range=(float(self.ymin.value()), float(self.ymax.value())),
                min_area=int(self.min_area.value()),
                overlay_path=overlay_path,
                # moves-solver controls
                grid_dx=float(self.grid_dx.value()),
                grid_dy=float(self.grid_dy.value()),
                algo=self.algo.currentText(),
                targeting_mode=("x-first" if self.targeting.currentIndex()==0 else "nearest"),
                allow_soft_fallback=self.allow_soft.isChecked(),
                # padding / tolerance
                inflate_main_px=int(self.pad_main.value()),
                inflate_tol_px=int(self.pad_tol.value()),
                border_px=int(self.border_px.value()),
                min_obstacle_area=int(self.min_obs_area.value()),
                min_thickness_px=float(self.min_thick.value()),
                shrink_steps=int(self.shrink_steps.value()),
                # progress
                progress_cb=progress_cb
            )
        except Exception as e:
            prog.cancel()
            QtWidgets.QMessageBox.critical(self, "Solver error", str(e))
            return
        finally:
            prog.setValue(100)

        expr = result.get("expr", "0")
        overlay_bgr = result.get("overlay_bgr")
        crit_bgr    = result.get("crit_overlay_bgr")

        if overlay_bgr is not None:
            self._show_np_rgb(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), self.overlay_label)
        else:
            self.overlay_label.setText("(no overlay)")

        if crit_bgr is not None:
            self._show_np_rgb(cv2.cvtColor(crit_bgr, cv2.COLOR_BGR2RGB), self.crit_label)
        else:
            self.crit_label.setText("(no critical grid)")

        self.set_output(expr)
        self.status.setText(
            f"Solved. Expression length: {len(expr)} | Overlay saved: {overlay_path} | "
            f"Algo: {self.algo.currentText()} | Targeting: {self.targeting.currentText()}"
        )
