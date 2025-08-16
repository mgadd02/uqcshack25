from PyQt5 import QtWidgets
from ui_modules.live_view_tab import LiveViewTab


class MainTab(QtWidgets.QTabWidget):
    def __init__(self, shot_output_tab=None):
        super().__init__()

        # Keep reference to the output tab (if needed for sending results)
        self.shot_output_tab = shot_output_tab

        # Sample View tab
        self.sample_tab = LiveViewTab()
        self.addTab(self.sample_tab, "Sample View")

        # Processing tab (placeholder for algorithm processing)
        self.processing_tab = QtWidgets.QWidget()
        self.addTab(self.processing_tab, "Processing")
