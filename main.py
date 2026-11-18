"""Запуск приложения: python main.py."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))


def main():
    try:
        from lab6.qt import QtWidgets
    except ImportError as err:
        print(f"Не удалось загрузить зависимости: {err}\npython -m pip install -r requirements.txt",file=sys.stderr)
        return 1
    from lab6.ui import MainWindow
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Лабораторная №9")
    window = MainWindow()
    window.show()
    run = app.exec if hasattr(app,"exec") else app.exec_
    return run()


if __name__=="__main__":
    raise SystemExit(main())
