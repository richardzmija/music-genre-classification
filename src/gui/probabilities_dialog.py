import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
                             QLabel, QPushButton, QLineEdit, QVBoxLayout, QWidget, 
                             QStatusBar, QDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox)

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from typing import Dict


class ProbabilitiesDialog(QDialog):
    def __init__(self, probabilities: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Genre Prediction Probabilities")
        self.setGeometry(300, 300, 400, 300)
        
        self.createTable(probabilities)
        self.setUpLayout()
    
    def createTable(self, probabilities: Dict[str, float]):
        self.table = QTableWidget()
        self.table.setRowCount(len(probabilities))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Genre", "Probability"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        for i, (genre, prob) in enumerate(probabilities.items()):
            self.table.setItem(i, 0, QTableWidgetItem(genre))
            self.table.setItem(i, 1, QTableWidgetItem(f"{prob:.4f}"))
    
    def setUpLayout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        closeButton = QPushButton("Close")
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton)
        self.setLayout(layout)
