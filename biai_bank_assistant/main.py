from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.assistant import BankAssistant
from src.data import load_faq
from ui.main_window import MainWindow


ROOT = Path(__file__).resolve().parent


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Nirbhor Bank — NB Assist")
    entries = load_faq(ROOT / "data" / "bank-faq.csv")
    assistant = BankAssistant(entries)
    window = MainWindow(assistant)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
