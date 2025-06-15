import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableView, QFileDialog, QDialog, QLabel, QComboBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data  # 2D list

    def data(self, index, role):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return str(self._data[index.row()][index.column()])

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            try:
                self._data[index.row()][index.column()] = float(value)
            except ValueError:
                self._data[index.row()][index.column()] = value
            return True

    def rowCount(self, _=None):
        return len(self._data)

    def columnCount(self, _=None):
        return len(self._data[0]) if self._data else 0

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled

    def add_row(self):
        self.beginInsertRows(QModelIndex(), len(self._data), len(self._data))
        self._data.append([0.0] * self.columnCount())
        self.endInsertRows()

    def add_column(self):
        for row in self._data:
            row.append(0.0)
        self.layoutChanged.emit()


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def plot_data(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y, marker='o')
        self.ax.set_title("绘图结果")
        self.draw()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出选项")
        self.format_choice = QComboBox()
        self.format_choice.addItems(["CSV", "PNG"])
        layout = QVBoxLayout()
        layout.addWidget(QLabel("选择导出格式："))
        layout.addWidget(self.format_choice)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_choice(self):
        return self.format_choice.currentText()


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 表格 + 图形")

        self.model = TableModel([[0.0 for _ in range(3)] for _ in range(5)])
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionMode(QTableView.ContiguousSelection)

        self.canvas = MatplotlibCanvas()

        layout = QHBoxLayout()
        layout.addWidget(self.table, 2)
        layout.addWidget(self.canvas, 3)

        buttons = QHBoxLayout()
        btn_plot = QPushButton("绘图")
        btn_export = QPushButton("导出")
        btn_row = QPushButton("添加行")
        btn_col = QPushButton("添加列")
        btn_plot.clicked.connect(self.plot_selected)
        btn_export.clicked.connect(self.export_data)
        btn_row.clicked.connect(self.model.add_row)
        btn_col.clicked.connect(self.model.add_column)
        for b in [btn_plot, btn_export, btn_row, btn_col]:
            buttons.addWidget(b)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(buttons)

        self.setLayout(main_layout)

    def plot_selected(self):
        indexes = self.table.selectionModel().selectedIndexes()
        if not indexes:
            return
        rows = sorted(set(i.row() for i in indexes))
        cols = sorted(set(i.column() for i in indexes))
        if len(cols) < 2:
            return

        x = [float(self.model._data[r][cols[0]]) for r in rows]
        y = [float(self.model._data[r][cols[1]]) for r in rows]
        self.canvas.plot_data(x, y)

    def export_data(self):
        dialog = ExportDialog(self)
        if dialog.exec_():
            fmt = dialog.get_choice()
            if fmt == "CSV":
                path, _ = QFileDialog.getSaveFileName(self, "保存为 CSV", "", "CSV文件 (*.csv)")
                if path:
                    with open(path, "w") as f:
                        for row in self.model._data:
                            f.write(",".join(map(str, row)) + "\n")
            elif fmt == "PNG":
                path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG图片 (*.png)")
                if path:
                    self.canvas.fig.savefig(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())
