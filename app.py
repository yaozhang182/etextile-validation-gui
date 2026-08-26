#!/usr/bin/env python3
"""
E-Textile Validation GUI — Main entry point.

Usage:
    python app.py
    python app.py --version     # print version and exit (used by the build smoke test)
"""

import sys

__version__ = '0.2.0'


def main():
    # Handled before Qt starts so the packaged .exe can be checked without a display.
    if '--version' in sys.argv:
        print(f"E-Textile Validation GUI {__version__}")
        return 0

    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("E-Textile Validation GUI")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
