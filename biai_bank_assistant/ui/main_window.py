from __future__ import annotations

from html import escape

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

from src.assistant import BankAssistant
from ui.theme import APP_QSS


class AnswerWorker(QObject):
    finished = Signal(object)

    def __init__(self, assistant: BankAssistant, question: str):
        super().__init__()
        self.assistant = assistant
        self.question = question

    @Slot()
    def run(self):
        self.finished.emit(self.assistant.answer(self.question))


class MainWindow(QMainWindow):
    def __init__(self, assistant: BankAssistant):
        super().__init__()
        self.assistant = assistant
        self.thread: QThread | None = None
        self.worker: AnswerWorker | None = None
        self.setMinimumSize(1180, 760)
        self.resize(1400, 860)
        self.setStyleSheet(APP_QSS)
        self._build()

    @staticmethod
    def _neo_shadow(widget: QWidget, offset: int = 7):
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(0)
        shadow.setOffset(offset, offset)
        shadow.setColor(Qt.black)
        widget.setGraphicsEffect(shadow)

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(16)
        outer.addWidget(self._sidebar(), 0)

        center = QVBoxLayout()
        center.setSpacing(14)
        center.addWidget(self._header())
        center.addWidget(self._hero(), 0)
        center.addWidget(self._chat_card(), 1)
        center.addWidget(self._composer_card(), 0)
        outer.addLayout(center, 1)
        outer.addWidget(self._inspector(), 0)

    def _sidebar(self):
        panel = QFrame(objectName="sidebar")
        panel.setFixedWidth(218)
        self._neo_shadow(panel, 7)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(9)

        brand_row = QHBoxLayout()
        mark = QLabel("N", objectName="brandMark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(34, 34)
        brand_row.addWidget(mark)
        brand = QLabel("NB Assist", objectName="brand")
        brand_row.addWidget(brand)
        brand_row.addStretch()
        layout.addLayout(brand_row)
        layout.addSpacing(24)
        for label in ("◉   Assistant", "▦   Evaluation", "⌁   FAQ library"):
            button = QPushButton(label, objectName="navButton")
            layout.addWidget(button)
        layout.addStretch()
        safe = QLabel("●  Offline-first\n\nYour question stays on this device when the local model is active.", objectName="subtle")
        safe.setWordWrap(True)
        layout.addWidget(safe)
        return panel

    def _header(self):
        container = QWidget()
        row = QHBoxLayout()
        container.setLayout(row)
        title = QLabel("Customer support workspace", objectName="title")
        row.addWidget(title)
        row.addStretch()
        pill = QLabel("●  OFFLINE READY", objectName="statusPill")
        row.addWidget(pill)
        return container

    def _hero(self):
        card = QFrame(objectName="heroCard")
        self._neo_shadow(card, 7)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        eyebrow = QLabel("NIRBHOR BANK PLC  /  NB GO", objectName="eyebrow")
        layout.addWidget(eyebrow)
        title = QLabel("Clear answers, in the language\nyour customer uses.", objectName="title")
        layout.addWidget(title)
        subtitle = QLabel("Grounded only in the bank FAQ. Unsupported questions are routed to a person.", objectName="subtle")
        layout.addWidget(subtitle)
        chips = QHBoxLayout()
        for text in ("English", "বাংলা", "Banglish", "Offline by design"):
            chips.addWidget(QLabel(text, objectName="chip"))
        chips.addStretch()
        layout.addLayout(chips)
        return card

    def _chat_card(self):
        self.chat_card = QFrame(objectName="chatCard")
        self._neo_shadow(self.chat_card, 7)
        layout = QVBoxLayout(self.chat_card)
        layout.setContentsMargins(18, 14, 18, 14)
        label = QLabel("Conversation", objectName="eyebrow")
        layout.addWidget(label)
        self.chat_view = QTextEdit(objectName="chatView")
        self.chat_view.setReadOnly(True)
        self.chat_view.setPlaceholderText("Ask about accounts, cards, transfers, loans, or NB Go…")
        self.chat_view.setHtml(self._welcome_html())
        layout.addWidget(self.chat_view)
        return self.chat_card

    def _composer_card(self):
        card = QFrame(objectName="composerCard")
        self._neo_shadow(card, 7)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        chips = QHBoxLayout()
        for text in ("Card er PIN reset korbo kibhabe?", "ক্রেডিট কার্ডের বিল কবে দিতে হয়?", "What is the ATM limit?"):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, t=text: self._set_question(t))
            chips.addWidget(button)
        chips.addStretch()
        layout.addLayout(chips)
        row = QHBoxLayout()
        self.composer = QTextEdit(objectName="composer")
        self.composer.setFixedHeight(72)
        self.composer.setPlaceholderText("Write in English, বাংলা, or Banglish…")
        row.addWidget(self.composer, 1)
        self.send_button = QPushButton("Send  ↗", objectName="sendButton")
        self.send_button.setFixedWidth(104)
        self.send_button.clicked.connect(self._send)
        row.addWidget(self.send_button, 0, Qt.AlignBottom)
        layout.addLayout(row)
        return card

    def _inspector(self):
        panel = QFrame(objectName="inspector")
        panel.setFixedWidth(242)
        self._neo_shadow(panel, 7)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(QLabel("Response details", objectName="eyebrow"))
        self.language_value = QLabel("—", objectName="metricValue")
        self.source_value = QLabel("—", objectName="metricValue")
        self.latency_value = QLabel("—", objectName="metricValue")
        for label, value, key in (("Detected language", self.language_value, "Language"), ("FAQ source", self.source_value, "Source"), ("Response time", self.latency_value, "Latency")):
            block = QVBoxLayout()
            block.addWidget(QLabel(key, objectName="metricLabel"))
            block.addWidget(value)
            layout.addLayout(block)
        layout.addSpacing(14)
        layout.addWidget(QLabel("Grounding", objectName="eyebrow"))
        self.grounding = QLabel("Waiting for a question", objectName="subtle")
        self.grounding.setWordWrap(True)
        layout.addWidget(self.grounding)
        layout.addStretch()
        return panel

    @staticmethod
    def _welcome_html() -> str:
        return '<div style="color:#101010;font-weight:900">NB ASSIST / READY</div><div style="color:#252525;margin-top:6px">Hello. Ask me about Nirbhor Bank products and services. I answer only from the supplied FAQ.</div>'

    def _set_question(self, text: str):
        self.composer.setPlainText(text)
        self.composer.setFocus()

    def _send(self):
        question = self.composer.toPlainText().strip()
        if not question or self.thread is not None:
            return
        self.chat_view.append(f'<div style="color:#95d9ff;margin-top:16px"><b>You</b><br>{escape(question)}</div>')
        self.send_button.setEnabled(False)
        self.send_button.setText("Thinking…")
        self.thread = QThread(self)
        self.worker = AnswerWorker(self.assistant, question)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._received)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_done)
        self.thread.start()

    @Slot(object)
    def _received(self, result):
        color = "#101010" if result.status == "answered" else "#a23a20"
        self.chat_view.append(f'<div style="color:{color};margin-top:16px"><b>NB Assist</b><br>{escape(result.answer)}</div>')
        self.language_value.setText(result.language.title())
        self.source_value.setText(f"FAQ {result.source_id}" if result.source_id else "Handoff")
        self.latency_value.setText(f"{result.latency_ms:.0f} ms")
        if result.status == "answered":
            self.grounding.setText(f"Grounded in FAQ {result.source_id}. Support type: {result.support_type.replace('_', ' ')}.")
        else:
            self.grounding.setText("No supported FAQ evidence was strong enough. The assistant offered a human handoff.")

    def _thread_done(self):
        self.thread = None
        self.worker = None
        self.send_button.setEnabled(True)
        self.send_button.setText("Send  ↗")
