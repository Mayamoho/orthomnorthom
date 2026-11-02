from __future__ import annotations

import re
import unicodedata


BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize for retrieval while keeping the original text untouched for display."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^\w\u0980-\u09FF%+./-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def detect_language(text: str) -> str:
    """Return en, bn, banglish, or mixed using script and token heuristics."""
    raw = text or ""
    bn = len(BENGALI_RE.findall(raw))
    latin = len(LATIN_RE.findall(raw))
    if bn and latin:
        return "bn" if bn >= latin * 0.35 else "mixed"
    if bn:
        return "bn"
    if not latin:
        return "en"

    words = tokenize(raw)
    banglish_markers = {
        "amar", "apnar", "apnader", "koto", "kibhabe", "kivabe", "ki", "kon",
        "koren", "korbo", "hobe", "ache", "ase", "hoy", "hoye", "jabe", "jabo",
        "chai", "lagbe", "pabo", "paben", "niye", "theke", "er", "e", "jonno",
        "account", "card", "loan", "fee", "transfer", "pin", "branch", "kora",
    }
    marker_hits = sum(1 for token in words if token in banglish_markers)
    return "banglish" if marker_hits >= 2 or (marker_hits == 1 and len(words) <= 8) else "en"


def language_name(code: str) -> str:
    return {"en": "English", "bn": "Bangla", "banglish": "Banglish", "mixed": "mixed"}.get(code, code)
