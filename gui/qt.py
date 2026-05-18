"""Qt compatibility helpers."""

try:
    from PyQt5.QtCore import QPointF, QThread, Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QColor, QBrush, QPen, QPainterPath, QPolygonF, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsEllipseItem,
        QGraphicsItemGroup,
        QGraphicsPathItem,
        QGraphicsPolygonItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QSizePolicy,
        QHeaderView,
    )
    QT_LIB = "PyQt5"
except ImportError:  # pragma: no cover - import fallback
    try:
        from PySide6.QtCore import QPointF, QThread, Qt, Signal
        from PySide6.QtGui import QColor, QBrush, QPen, QPainterPath, QPolygonF, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGraphicsEllipseItem,
            QGraphicsItemGroup,
            QGraphicsPathItem,
            QGraphicsPolygonItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsSimpleTextItem,
            QGraphicsView,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSplitter,
            QStackedWidget,
            QStatusBar,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
            QSizePolicy,
            QHeaderView,
        )
        QT_LIB = "PySide6"
    except ImportError as exc:  # pragma: no cover - no Qt dependency installed
        raise ImportError(
            "未检测到 PyQt5 或 PySide6，请先安装其中一个，例如：pip install PyQt5"
        ) from exc
