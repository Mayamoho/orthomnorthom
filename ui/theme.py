APP_QSS = r"""
* { font-family: "Segoe UI", "Noto Sans Bengali", sans-serif; }
QMainWindow { background: #0a0d13; color: #f6f7fb; }
QWidget { color: #f6f7fb; }
QFrame#sidebar, QFrame#inspector, QFrame#composerCard, QFrame#heroCard, QFrame#statCard, QFrame#chatCard {
  background: rgba(22, 28, 40, 230); border: 1px solid rgba(255,255,255,30); border-radius: 22px;
}
QFrame#sidebar { background: rgba(15, 20, 31, 245); border-radius: 26px; }
QFrame#heroCard { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #27345a, stop:0.45 #171f38, stop:1 #182b2b); }
QLabel#eyebrow { color: #9aa9c4; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#title { color: #ffffff; font-size: 28px; font-weight: 800; }
QLabel#subtle { color: #9aa9c4; font-size: 13px; }
QLabel#brand { color: #ffffff; font-size: 18px; font-weight: 800; }
QLabel#brandMark { background: #a6f36b; color: #0c160e; border-radius: 12px; font-size: 18px; font-weight: 900; }
QLabel#chip { color: #c8d2e5; background: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,22); border-radius: 12px; padding: 7px 11px; }
QLabel#metricValue { color: #ffffff; font-size: 22px; font-weight: 800; }
QLabel#metricLabel { color: #93a0b7; font-size: 11px; }
QLabel#statusPill { color: #b9ff87; background: rgba(166,243,107,24); border: 1px solid rgba(166,243,107,80); border-radius: 10px; padding: 5px 9px; font-size: 11px; font-weight: 700; }
QLabel#sourcePill { color: #95d9ff; background: rgba(91,182,255,18); border: 1px solid rgba(91,182,255,75); border-radius: 10px; padding: 5px 9px; font-size: 11px; }
QTextEdit#chatView { background: transparent; border: none; padding: 8px; font-size: 14px; }
QTextEdit#composer { background: rgba(255,255,255,8); border: 1px solid rgba(255,255,255,38); border-radius: 17px; padding: 13px; font-size: 14px; }
QTextEdit#composer:focus { border: 1px solid #a6f36b; background: rgba(255,255,255,13); }
QPushButton { background: rgba(255,255,255,12); border: 1px solid rgba(255,255,255,26); border-radius: 12px; padding: 9px 13px; color: #dfe7f5; font-weight: 600; }
QPushButton:hover { background: rgba(255,255,255,24); border-color: rgba(166,243,107,90); }
QPushButton#sendButton { background: #a6f36b; color: #101b13; border: none; border-radius: 15px; padding: 12px 18px; font-weight: 800; }
QPushButton#sendButton:hover { background: #c2ff94; }
QPushButton#navButton { text-align: left; padding: 12px 14px; border: none; background: transparent; color: #9aa9c4; }
QPushButton#navButton:hover { color: #ffffff; background: rgba(255,255,255,12); }
QScrollBar:vertical { background: transparent; width: 8px; margin: 4px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,35); border-radius: 4px; }
"""
