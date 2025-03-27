from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFormLayout, QLineEdit, QComboBox
)
import subprocess
import os


def load_image_names(folder="imgs"):
    """
    Scans the 'imgs' folder and returns a list of file names with no extension,
    if the file name ends in one of the standard image extensions.
    """
    image_names = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            name = os.path.splitext(filename)[0]
            image_names.append(name)
    return image_names


class SimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modeling")
        self.simulation_process = None
        self.setFixedWidth(350)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Form for specifying modeling parameters
        form_layout = QFormLayout()
        self.input_net = QLineEdit("Fluid_model")
        self.input_hidden_size = QLineEdit("50")
        self.input_mu = QLineEdit("0.1")
        self.input_rho = QLineEdit("10")

        # Drop-down list for selecting the simulation type
        self.input_type = QComboBox()
        simulation_types = ["DFG_benchmark", "box", "magnus", "image", "ecmo", "poiseuille", "paint"]
        self.input_type.addItems(simulation_types)

        # Drop-down list for selecting an image, load the list from the imgs folder
        self.input_image = QComboBox()
        self.image_names = load_image_names("imgs")
        self.input_image.addItems(self.image_names)
        # The drop-down list for an image is active only when the “image” type is selected
        self.input_image.setEnabled(self.input_type.currentText() == "image")

        # When changing the simulation type, update the state of the image selection widget
        self.input_type.currentTextChanged.connect(self.on_type_changed)

        form_layout.addRow("net:", self.input_net)
        form_layout.addRow("hidden_size:", self.input_hidden_size)
        form_layout.addRow("mu:", self.input_mu)
        form_layout.addRow("rho:", self.input_rho)
        form_layout.addRow("Simulation type:", self.input_type)
        form_layout.addRow("Image:", self.input_image)

        layout.addLayout(form_layout)

        self.btn_run = QPushButton("Run simulation")
        self.btn_run.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop simulation")
        self.btn_stop.clicked.connect(self.stop_simulation)
        layout.addWidget(self.btn_stop)

        self.status_label = QLabel("Status: pending launch")
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_type_changed(self, new_type):
        # If “image” type is selected, allow selection of a specific image
        self.input_image.setEnabled(new_type == "image")

    def run_simulation(self):
        self.status_label.setText("Running a simulation...")
        # Form a start command with parameter passing
        cmd = [
            "python", "fluid_test.py",
            "--net=" + self.input_net.text(),
            "--hidden_size=" + self.input_hidden_size.text(),
            "--mu=" + self.input_mu.text(),
            "--rho=" + self.input_rho.text(),
            "--type=" + self.input_type.currentText()
        ]
        # If the “image” type is selected, add the --image parameter with the selected name
        if self.input_type.currentText() == "image":
            cmd.append("--image=" + self.input_image.currentText())
        self.simulation_process = subprocess.Popen(cmd)

    def stop_simulation(self):
        if self.simulation_process:
            self.simulation_process.terminate()
            self.status_label.setText("Simulation stopped")
            self.simulation_process = None
