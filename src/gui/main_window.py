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
import src.core.genre_prediction as prediction
import src.core.audio_feature_extractor as extractor
from src.gui.probabilities_dialog import ProbabilitiesDialog
from src.gui.history_dialog import HistoryDialog


MODEL_PATH = r"models\xgb_model.pkl"
ENCODER_PATH = r"models\xgb_encoder.pkl"
HISTORY_FILE = r"src\gui\resources\history.csv"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._initUI()
        self.classifier = prediction.MusicGenreClassifier(MODEL_PATH, ENCODER_PATH)
        self.historyFile = HISTORY_FILE
    
    def _initUI(self):
        self.setWindowTitle("Music Genre Classifier")
        self.setGeometry(100, 100, 800, 600)
        
        self.createMenu()
        self.createWidgets()
        self.createStatusBar()
        
        self.setUpLayout()
    
    def createMenu(self):
        menubar = self.menuBar()
        
        # File menu
        fileMenu = menubar.addMenu("File")
        
        openFile = QAction("Open Audio File", self)
        openFile.triggered.connect(self.openFileDialog)
        fileMenu.addAction(openFile)
        
        # History menu
        historyMenu = menubar.addMenu("History")
        
        viewHistory = QAction("View History", self)
        viewHistory.triggered.connect(self.viewHistory)
        historyMenu.addAction(viewHistory)
        
        # Help menu
        helpMenu = menubar.addMenu("Help")
        
        aboutApp = QAction("About", self)
        aboutApp.triggered.connect(self.about)
        helpMenu.addAction(aboutApp)
        
        # Exit
        exitApp = QAction("Exit", self)
        exitApp.triggered.connect(self.close)
        fileMenu.addAction(exitApp)
        
    def createWidgets(self):
        self.fileLabel = QLabel("Browse File:")
        self.filePathEdit = QLineEdit()
        self.browseButton = QPushButton("Browse...")
        self.browseButton.clicked.connect(self.openFileDialog)
        
        self.playButton = QPushButton("Play")
        self.pauseButton = QPushButton("Pause")
        self.stopButton = QPushButton("Stop")
        
        self.playButton.clicked.connect(self.playAudio)
        self.pauseButton.clicked.connect(self.pauseAudio)
        self.stopButton.clicked.connect(self.stopAudio)
        
        self.predictButton = QPushButton("Predict Genre")
        self.predictButton.clicked.connect(self.predictGenre)
        
        self.predictedGenreLabel = QLabel("Predicted Genre:")
        self.predictedGenreEdit = QLineEdit()
        self.predictedGenreEdit.setReadOnly(True)
        
        self.showProbabilitiesButton = QPushButton("Show Probabilities")
        self.showProbabilitiesButton.clicked.connect(self.showProbabilities)
        
        self.player = QMediaPlayer()
    
    def createStatusBar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
    
    def setUpLayout(self):
        mainLayout = QVBoxLayout()
        
        mainLayout.addWidget(self.fileLabel)
        mainLayout.addWidget(self.filePathEdit)
        mainLayout.addWidget(self.browseButton)
        
        mainLayout.addWidget(self.playButton)
        mainLayout.addWidget(self.pauseButton)
        mainLayout.addWidget(self.stopButton)
        
        mainLayout.addWidget(self.predictButton)
        
        mainLayout.addWidget(self.predictedGenreLabel)
        mainLayout.addWidget(self.predictedGenreEdit)
        
        mainLayout.addWidget(self.showProbabilitiesButton)
        
        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)
        
    def openFileDialog(self):
        fileDialog = QFileDialog()
        filePath, _ = fileDialog.getOpenFileName(self, "Open Audio File",
                                                 "", "Audio Files (*.wav *.mp3 *.flac)")
        
        if filePath:
            self.filePathEdit.setText(filePath)
            self.statusBar.showMessage("File loaded successfully", 5000)
    
    def playAudio(self):
        filePath = self.filePathEdit.text()
        if filePath:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(filePath)))
            self.player.play()
        else:
            self.statusBar.showMessage("Please load an audio file first", 5000)
    
    def pauseAudio(self):
        self.player.pause()
    
    def stopAudio(self):
        self.player.stop()
    
    def predictGenre(self):
        filePath = self.filePathEdit.text()
        
        if filePath:
            try:
                audio_data, _ = extractor.load_audio(filePath)
                predicted_genre = self.classifier.classify_genre(audio_data)
                self.predictedGenreEdit.setText(predicted_genre)
                self.statusBar.showMessage("Genre predicted successfully", 5000)
                
                # Save to history
                with open(self.historyFile, mode="a") as file:
                    file.write(f"{filePath},{predicted_genre},{datetime.now()}\n")
            except Exception as e:
                self.statusBar.showMessage(f"Error: {e}", 5000)
        else:
            self.statusBar.showMessage("Please load an audio file first", 5000)

    def showProbabilities(self):
        filePath = self.filePathEdit.text()
        if filePath:
            try:
                audio_data, _ = extractor.load_audio(filePath)
                probabilities = self.classifier.predict_probabilities(audio_data)
                
                probsDialog = ProbabilitiesDialog(probabilities, self)
                probsDialog.exec_()
            except Exception as e:
                self.statusBar.showMessage(f"Error: {e}", 5000)
        else:
            self.statusBar.showMessage("Please load an audio file first", 5000)
    
    def viewHistory(self):
        historyDialog = HistoryDialog(HISTORY_FILE, self)
        historyDialog.exec_()
        
    def about(self):
        QMessageBox.about(self, "About Music Genre Classifier",
                          "This application classifies the genre of an audio"\
                          " file using a trained XGBoost model.")
    