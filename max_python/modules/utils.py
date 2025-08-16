from PyQt5 import QtWidgets, QtCore, QtGui

class GraphwarBotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphwar Bot")
        self.resize(1000, 700)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.main_tab, "Main")

        self._init_main_tab()

    def _init_main_tab(self):
        layout = QtWidgets.QVBoxLayout()

        # Live capture display
        self.live_label = QtWidgets.QLabel("No image yet")
        self.live_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.live_label)

        # Run processing button
        self.process_btn = QtWidgets.QPushButton("Process Latest Frame")
        self.process_btn.clicked.connect(self.run_processing)
        layout.addWidget(self.process_btn)

        # Output area + copy button
        output_layout = QtWidgets.QHBoxLayout()
        self.output_box = QtWidgets.QPlainTextEdit()
        self.output_box.setReadOnly(False)  # allow select all
        output_layout.addWidget(self.output_box)

        copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_output_to_clipboard)
        output_layout.addWidget(copy_btn)

        layout.addLayout(output_layout)

        self.main_tab.setLayout(layout)

    def run_processing(self):
        # TODO: Hook this into your algorithm
        self.output_box.setPlainText("y = x^2 + ...")  # Example

    def copy_output_to_clipboard(self):
        text = self.output_box.toPlainText()
        QtWidgets.QApplication.clipboard().setText(text)
