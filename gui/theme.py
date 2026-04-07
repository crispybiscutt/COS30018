"""
COS30018 - Clean Light Theme
Simple, professional stylesheet for the HNRS GUI.
"""

# Colors
PRIMARY = "#2563eb"
PRIMARY_HOVER = "#1d4ed8"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
WARNING = "#f59e0b"
TEXT = "#1e293b"
TEXT_SECONDARY = "#64748b"
BG = "#f8fafc"
CARD_BG = "#ffffff"
BORDER = "#e2e8f0"
INPUT_BG = "#f1f5f9"

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}

QTabWidget::pane {
    border: 1px solid #e2e8f0;
    background: #ffffff;
    border-radius: 4px;
}
QTabBar::tab {
    background: #f1f5f9;
    color: #64748b;
    padding: 10px 24px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}
QTabBar::tab:hover:!selected {
    background: #e2e8f0;
    color: #1e293b;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
    font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #2563eb;
}

QPushButton {
    background-color: #f1f5f9;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #e2e8f0;
    border-color: #cbd5e1;
}
QPushButton:pressed {
    background-color: #cbd5e1;
}
QPushButton:disabled {
    background-color: #f1f5f9;
    color: #94a3b8;
}

QComboBox {
    background-color: #f1f5f9;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QComboBox:hover { border-color: #2563eb; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #64748b;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    selection-background-color: #eff6ff;
    selection-color: #2563eb;
}

QSpinBox, QDoubleSpinBox {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 5px 10px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #2563eb; }

QTextEdit {
    background: #f8fafc;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 11px;
}

QProgressBar {
    background: #e2e8f0;
    border: none;
    border-radius: 6px;
    text-align: center;
    color: #1e293b;
    font-weight: 600;
    height: 20px;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 6px;
}

QCheckBox { color: #1e293b; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 2px solid #cbd5e1;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}

QLabel { color: #1e293b; }

QMenuBar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
QMenuBar::item:selected { background: #eff6ff; }
QMenu {
    background: #ffffff;
    border: 1px solid #e2e8f0;
}
QMenu::item:selected { background: #eff6ff; color: #2563eb; }

QStatusBar {
    background: #ffffff;
    color: #64748b;
    border-top: 1px solid #e2e8f0;
    font-size: 11px;
}

QScrollBar:vertical {
    background: #f1f5f9;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 30px;
}
"""


def btn_primary():
    return (
        "QPushButton { background: #2563eb; color: #fff; border: none; "
        "border-radius: 6px; padding: 8px 20px; font-weight: 600; }"
        "QPushButton:hover { background: #1d4ed8; }"
        "QPushButton:disabled { background: #94a3b8; }"
    )

def btn_success():
    return (
        "QPushButton { background: #16a34a; color: #fff; border: none; "
        "border-radius: 6px; padding: 8px 20px; font-weight: 600; }"
        "QPushButton:hover { background: #15803d; }"
        "QPushButton:disabled { background: #94a3b8; }"
    )

def btn_danger():
    return (
        "QPushButton { background: #dc2626; color: #fff; border: none; "
        "border-radius: 6px; padding: 8px 20px; font-weight: 600; }"
        "QPushButton:hover { background: #b91c1c; }"
        "QPushButton:disabled { background: #94a3b8; }"
    )

def btn_warning():
    return (
        "QPushButton { background: #f59e0b; color: #fff; border: none; "
        "border-radius: 6px; padding: 8px 20px; font-weight: 600; }"
        "QPushButton:hover { background: #d97706; }"
        "QPushButton:disabled { background: #94a3b8; }"
    )
