import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
                             QLabel, QPushButton, QLineEdit, QVBoxLayout, QWidget, 
                             QStatusBar, QDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox)

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import joblib
import librosa
import pandas as pd
from typing import Dict
from datetime import datetime


class HistoryDialog(QDialog):
    def __init__(self, history_file, parent=None):
        super().__init__(parent)
        self.history_file = history_file
        
        self.setWindowTitle("Prediction History")
        self.setGeometry(200, 200, 600, 400)
        
        self.createTable()
        self.createButtons()
        self.setUpLayout()
        
    def createTable(self):
        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File Name",
                                              "Predicted Genre",
                                              "Date/Time"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Load the history of predictions from persistent storage (CSV file)
        self.loadHistory()
        
    def createButtons(self):
        self.deleteButton = QPushButton("Delete History")
        self.deleteButton.clicked.connect(self.deleteHistory)
        
        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        
    def setUpLayout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.deleteButton)
        layout.addWidget(self.closeButton)
        self.setLayout(layout)
    
    def loadHistory(self):
        try:
            with open(self.history_file, mode="r") as file:
                for line in file:
                    file_name, predicted_genre, date_time = line.strip().split(",")
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QTableWidgetItem(file_name))
                    self.table.setItem(row_position, 1, QTableWidgetItem(predicted_genre))
                    self.table.setItem(row_position, 2, QTableWidgetItem(date_time))
        except FileNotFoundError:
            print("Error: history.csv not found")
    
    def deleteHistory(self):
        self.table.setRowCount(0)
        with open(self.history_file, "w") as file:
            pass
