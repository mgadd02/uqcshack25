from PyQt5 import QtWidgets

class SolverTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Solver visualisation will go here"))
        self.setLayout(layout)
