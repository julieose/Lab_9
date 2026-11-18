"""PySide6 по умолчанию. PyQt5 подходит для имеющихся установок Qt 5."""
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
    BINDING = "PySide6"
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
    BINDING = "PyQt5"
