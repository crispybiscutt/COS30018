"""
COS30018 - Main Window
PyQt5 main application window with 3 tabs:
1. Training - Train and configure ML models
2. Recognition - Draw/upload digits and get predictions
3. Evaluation - Compare model performance
"""
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMenuBar, QAction, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from gui.training_tab import TrainingTab
from gui.recognition_tab import RecognitionTab
from gui.evaluation_tab import EvaluationTab


class MainWindow(QMainWindow):
    """Main application window for the HNRS system."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("COS30018 - Handwritten Number Recognition System (HNRS)")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: white;
                border-radius: 3px;
            }
            QTabBar::tab {
                padding: 8px 20px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #2196F3;
                font-weight: bold;
            }
        """)

        self._init_menu()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_tabs(self):
        """Create the tab widget with 3 tabs."""
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 11))

        # Tab 1: Training
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Training")

        # Tab 2: Recognition
        self.recognition_tab = RecognitionTab()
        self.tabs.addTab(self.recognition_tab, "Recognition")

        # Tab 3: Evaluation
        self.evaluation_tab = EvaluationTab()
        self.tabs.addTab(self.evaluation_tab, "Evaluation")

        self.setCentralWidget(self.tabs)

    def _init_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - COS30018 HNRS")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About HNRS",
            "Handwritten Number Recognition System\n\n"
            "COS30018 - Intelligent Systems\n"
            "Swinburne University of Technology\n\n"
            "Features:\n"
            "- Multiple ML models (CNN, SVM, KNN)\n"
            "- Image preprocessing & segmentation\n"
            "- Drawing canvas for real-time recognition\n"
            "- Comprehensive model evaluation"
        )
