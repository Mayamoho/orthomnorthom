APP_QSS = r"""
* { font-family: "Segoe UI", "Noto Sans Bengali", sans-serif; }
QMainWindow { background: #f2eee5; color: #101010; }
QWidget { color: #101010; }
QFrame#sidebar, QFrame#inspector, QFrame#composerCard, QFrame#heroCard, QFrame#chatCard {
  border: 3px solid #101010; border-radius: 0px;
}
QFrame#sidebar { background: #101010; }
QFrame#heroCard { background: #ff7657; }
QFrame#chatCard { background: #fffdf7; }
QFrame#composerCard { background: #f6e85d; }
QFrame#inspector { background: #b9a7ff; }
QLabel#eyebrow { color: #101010; font-size: 11px; font-weight: 900; letter-spacing: 1px; }
QLabel#title { color: #101010; font-size: 28px; font-weight: 900; }
QLabel#subtle { color: #252525; font-size: 13px; }
QFrame#sidebar QLabel#subtle { color: #fffdf7; }
QLabel#brand { color: #ffffff; font-size: 18px; font-weight: 900; }
QLabel#brandMark { background: #b9f35b; color: #101010; border: 3px solid #101010; border-radius: 0px; font-size: 18px; font-weight: 900; }
QLabel#chip { color: #101010; background: #fffdf7; border: 2px solid #101010; border-radius: 0px; padding: 7px 11px; font-weight: 800; }
QLabel#metricValue { color: #101010; font-size: 22px; font-weight: 900; }
QLabel#metricLabel { color: #30265e; font-size: 11px; font-weight: 900; }
QLabel#statusPill { color: #101010; background: #b9f35b; border: 3px solid #101010; border-radius: 0px; padding: 6px 10px; font-size: 11px; font-weight: 900; }
QTextEdit#chatView { background: transparent; border: none; padding: 8px; font-size: 14px; }
QTextEdit#composer { background: #fffdf7; border: 3px solid #101010; border-radius: 0px; padding: 13px; font-size: 14px; }
QTextEdit#composer:focus { border: 3px solid #101010; background: #ffffff; }
QPushButton { background: #fffdf7; border: 3px solid #101010; border-radius: 0px; padding: 9px 13px; color: #101010; font-weight: 900; }
QPushButton:hover { background: #b9f35b; }
QPushButton:pressed { background: #ff7657; }
QPushButton#sendButton { background: #101010; color: #fffdf7; border: 3px solid #101010; border-radius: 0px; padding: 12px 18px; font-weight: 900; }
QPushButton#sendButton:hover { background: #3b3b3b; color: #b9f35b; }
QPushButton#navButton { text-align: left; padding: 12px 14px; border: 3px solid #ffffff; background: #101010; color: #ffffff; }
QPushButton#navButton:hover { color: #101010; background: #b9f35b; border-color: #b9f35b; }
QScrollBar:vertical { background: #f2eee5; width: 12px; margin: 4px; border: 2px solid #101010; }
QScrollBar::handle:vertical { background: #101010; min-height: 32px; }
"""
