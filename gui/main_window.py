"""
COS30018 - Main Window
PyQt5 main application window with 3 tabs:
1. Recognition - Draw/upload digits and get predictions
2. Evaluation - Compare model performance
3. Models - Pre-trained model info
"""
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QAction, QMessageBox, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from gui.theme import STYLESHEET
from gui.training_tab import TrainingTab
from gui.recognition_tab import RecognitionTab
from gui.evaluation_tab import EvaluationTab


class MainWindow(QMainWindow):
    """Main application window for the HNRS system."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HNRS - Handwritten Number Recognition System")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(STYLESHEET)

        self._init_menu()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 11))

        self.recognition_tab = RecognitionTab()
        self.tabs.addTab(self.recognition_tab, "  Recognition  ")

        self.evaluation_tab = EvaluationTab()
        self.tabs.addTab(self.evaluation_tab, "  Evaluation  ")

        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "  Models  ")

        self.setCentralWidget(self.tabs)

    def _init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("HNRS Ready  |  COS30018 - Intelligent Systems")

    def _show_about(self):
        QMessageBox.about(
            self, "About HNRS",
            "Handwritten Number Recognition System\n\n"
            "COS30018 - Intelligent Systems\n"
            "Swinburne University of Technology\n\n"
            "Models: CNN (PyTorch), SVM, KNN\n"
            "Extension: Arithmetic Expression Recognition"
        )
