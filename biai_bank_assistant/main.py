from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.assistant import BankAssistant
from src.data import load_faq
from src.remote import RemoteAssistant
from ui.main_window import MainWindow


ROOT = Path(__file__).resolve().parent


def build_backend():
    """The Kaggle model over HTTP when BIAI_API_URL is set, the local stack otherwise.

    Both backends answer from bank-faq.csv on this machine: the remote one adds
    the dense retriever and the Qwen rewrite that a laptop CPU is too slow for,
    and falls back to the local pipeline if the tunnel dies mid-conversation.
    """
    local = BankAssistant(load_faq(ROOT / "data" / "bank-faq.csv"))
    url = os.getenv("BIAI_API_URL", "").strip()
    return RemoteAssistant(url, fallback=local) if url else local


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Nirbhor Bank — NB Assist")
    window = MainWindow(build_backend())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
