from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFormLayout, QLineEdit
import subprocess


class TrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model training")
        self.training_process = None
        self.setFixedWidth(350)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Form for setting training parameters
        form_layout = QFormLayout()
        self.input_net = QLineEdit("Fluid_model")
        self.input_hidden_size = QLineEdit("50")
        self.input_mu = QLineEdit("0.5")
        self.input_rho = QLineEdit("4")
        self.input_loss_domain_res = QLineEdit("10")
        self.input_loss_bound = QLineEdit("20")
        form_layout.addRow("net:", self.input_net)
        form_layout.addRow("hidden_size:", self.input_hidden_size)
        form_layout.addRow("mu:", self.input_mu)
        form_layout.addRow("rho:", self.input_rho)
        form_layout.addRow("loss_domain_res:", self.input_loss_domain_res)
        form_layout.addRow("loss_bound:", self.input_loss_bound)

        layout.addLayout(form_layout)

        self.btn_run = QPushButton("Start training")
        self.btn_run.clicked.connect(self.run_training)
        layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop training")
        self.btn_stop.clicked.connect(self.stop_training)
        layout.addWidget(self.btn_stop)

        self.status_label = QLabel("Status: pending")
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def run_training(self):
        self.status_label.setText("Training Launch...")
        cmd = [
            "python", "fluid_train.py",
            "--net=" + self.input_net.text(),
            "--hidden_size=" + self.input_hidden_size.text(),
            "--mu=" + self.input_mu.text(),
            "--rho=" + self.input_rho.text(),
            "--loss_domain_res=" + self.input_loss_domain_res.text(),
            "--loss_bound=" + self.input_loss_bound.text()
        ]
        self.training_process = subprocess.Popen(cmd)

    def stop_training(self):
        if self.training_process:
            self.training_process.terminate()
            self.status_label.setText("Training has been stopped")
            self.training_process = None
