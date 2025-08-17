import sys
from PyQt5 import QtWidgets
from ui_modules.live_view_tab import LiveViewTab
from ui_modules.process_tab import ProcessTab
from ui_modules.shot_output_tab import ShotOutputTab

class GraphwarBotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphwar Bot — Capture + Solver")
        self.resize(1280, 860)

        self.shot_output = ShotOutputTab()
        self.live_view = LiveViewTab()

        self.process_tab = ProcessTab(
            get_frame_func=self.live_view.get_latest_frame,
            set_output_func=self.shot_output.set_text,
            get_save_dir_func=self._get_default_save_dir
        )

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.live_view, "Live View")
        tabs.addTab(self.process_tab, "Process")
        tabs.addTab(self.shot_output, "Shot Output")
        self.setCentralWidget(tabs)

    def _get_default_save_dir(self) -> str:
        # Use the current working directory as a default folder for overlays
        return "."

    def closeEvent(self, e):
        try:
            self.live_view.shutdown()
        except Exception:
            pass
        super().closeEvent(e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = GraphwarBotUI()
    win.show()
    sys.exit(app.exec_())
