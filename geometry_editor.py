from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QApplication, QVBoxLayout, QWidget, QPushButton
)
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt, QRectF
import sys
import numpy as np

class GeometryCell(QGraphicsRectItem):
    def __init__(self, x, y, size):
        super().__init__(QRectF(x, y, size, size))
        self.size = size
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.black))
        self.masked = False

    def set_masked(self, value):
        self.masked = value
        self.setBrush(QBrush(Qt.black) if self.masked else QBrush(Qt.white))

class GeometryScene(QGraphicsScene):
    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if isinstance(item, GeometryCell):
            if event.button() == Qt.LeftButton:
                item.set_masked(True)
            elif event.button() == Qt.RightButton:
                item.set_masked(False)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton or event.buttons() & Qt.RightButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, GeometryCell):
                if event.buttons() & Qt.LeftButton:
                    item.set_masked(True)
                elif event.buttons() & Qt.RightButton:
                    item.set_masked(False)
        super().mouseMoveEvent(event)

class GeometryEditor(QGraphicsView):
    def __init__(self, rows=50, cols=50, cell_size=10):
        scene = GeometryScene()
        super().__init__(scene)
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.cells = []
        self.init_grid()

    def init_grid(self):
        scene = self.scene()
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                cell = GeometryCell(j * self.cell_size, i * self.cell_size, self.cell_size)
                scene.addItem(cell)
                row.append(cell)
            self.cells.append(row)
        self.setSceneRect(0, 0, self.cols * self.cell_size, self.rows * self.cell_size)

    def get_mask_array(self):
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                mask[i, j] = 1 if cell.masked else 0
        return mask

class GeometryEditorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Редактор геометрии")
        self.editor = GeometryEditor(rows=220, cols=41, cell_size=4)
        self.save_btn = QPushButton("Сохранить геометрию")
        self.save_btn.clicked.connect(self.save_geometry)
        layout = QVBoxLayout()
        layout.addWidget(self.editor)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def save_geometry(self):
        mask = self.editor.get_mask_array()
        np.save("custom_geometry.npy", mask)
        print("Геометрия сохранена как custom_geometry.npy")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeometryEditorWindow()
    window.show()
    sys.exit(app.exec_())
