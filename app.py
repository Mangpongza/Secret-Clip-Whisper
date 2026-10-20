"""Entry point ของโปรแกรม Audio Steganography."""
from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication

from whispersteg.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("เสียงกระซิบในคลิปลับ")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
