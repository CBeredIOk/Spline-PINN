import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import simulation_gui
import training_gui


class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home window")
        self.resize(350, 100)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        btn_simulation = QPushButton("Flow simulation")
        btn_training = QPushButton("Model training")

        btn_simulation.clicked.connect(self.launch_simulation)
        btn_training.clicked.connect(self.launch_training)

        layout.addWidget(btn_simulation)
        layout.addWidget(btn_training)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def launch_simulation(self):
        self.simulation_window = simulation_gui.SimulationGUI()
        self.simulation_window.show()

    def launch_training(self):
        self.training_window = training_gui.TrainingGUI()
        self.training_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainGUI()
    main_window.show()
    sys.exit(app.exec_())
