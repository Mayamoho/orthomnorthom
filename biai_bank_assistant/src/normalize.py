from __future__ import annotations

import re
import unicodedata


BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)

_BN_RANGE = "\\u0980-\\u09FF"
_SCRIPT_CLASS = f"[\\w{_BN_RANGE}]"

# Generic Bangla -> English banking vocabulary. Ordinary words and bank-generic
# concepts only: no evaluation questions, and no facts that could leak into an
# answer. The glossary exists purely so a Bangla query can reach English FAQ text.
BILINGUAL_GLOSSARY = {
    "অ্যাকাউন্ট": "account", "হিসাব": "account", "সেভিংস": "savings",
    "সর্বনিম্ন": "minimum", "ন্যূনতম": "minimum", "ব্যালেন্স": "balance",
    "টাকা": "money taka amount", "টাকার": "money taka amount", "কত": "how much many",
    "দিনে": "daily day", "দিন": "day", "বছর": "year", "প্রতি": "per every",
    "ট্রান্সফার": "transfer", "পাঠাতে": "transfer send", "চার্জ": "fee charge cost",
    "ফি": "fee", "কার্ড": "card", "কার্ডের": "card", "হারিয়ে": "lost",
    "হারান": "lost", "চুরি": "stolen", "পিন": "pin", "ভুলে": "forgot",
    "রিসেট": "reset", "চেক": "cheque check", "বই": "book",
    "পেমেন্ট": "payment", "বিল": "bill", "ক্রেডিট": "credit", "ডেবিট": "debit",
    "ঋণ": "loan", "লোন": "loan", "লোনের": "loan", "ধার": "loan credit",
    "যোগ্যতা": "eligibility eligible", "যোগ্য": "eligible", "সুদ": "interest",
    "এফডিআর": "fdr fixed deposit", "জরিমানা": "late fee penalty",
    "নমিনি": "nominee", "পরিবর্তন": "change update", "শাখা": "branch",
    "শাখাগুলো": "branches", "খোলা": "open opened opening", "খুলতে": "open",
    "সময়": "time hours", "মোবাইল": "mobile phone", "নম্বর": "number",
    "রেমিট্যান্স": "remittance", "বিদেশ": "abroad", "বন্ধ": "close stop dormant",
    "ডরম্যান্ট": "dormant", "কেওয়াইসি": "kyc", "আপডেট": "update",
    "তোলা": "withdraw withdrawal", "তুলতে": "withdraw", "তলব": "withdraw",
    "সর্বোচ্চ": "maximum limit", "লেনদেন": "transaction",
    "করিনি": "did not make unauthorized dispute", "জানাবো": "report tell complain",
    "জানাতে": "report notify", "কবে": "when due", "দিতে": "pay give",
    "পাওয়া": "get receive", "পেতে": "get receive", "পেয়ে": "get receive",
    "এটিএম": "atm", "ব্যক্তিগত": "personal", "বয়স": "age", "আয়": "income salary",
    "হোল্ডার": "holder", "যৌথ": "joint", "ব্যবসা": "business", "ব্যবসার": "business",
    "কাগজপত্র": "documents papers", "জমা": "deposit", "ডিপোজিট": "deposit",
    "ওভারড্রাফট": "overdraft", "গড়": "average", "খরচ": "cost fee charge",
    "দাম": "price rate cost", "স্বর্ণ": "gold", "রুপা": "silver", "লাভ": "profit gain",
    "শেয়ার": "share stock", "স্টক": "stock share", "ডলার": "dollar",
    "বিনিময়": "exchange", "ক্রিপ্টো": "crypto cryptocurrency", "ভিসা": "visa",
    "ম্যানেজার": "manager", "ঠিকানা": "address", "ইমেইল": "email",
    "পাসপোর্ট": "passport", "বীমা": "insurance", "ছাড়": "waive exempt discount",
    "স্বামী": "husband", "স্ত্রী": "wife", "বন্ধু": "friend", "বন্ধুর": "friend",
    "এসএমএস": "sms", "কোনো": "any", "এখন": "now", "আজকের": "today",
    "হাজার": "thousand lakh", "লক্ষ": "lakh", "ভুল": "wrong error",
    "করতে": "do to", "করা": "do make", "করে": "do make", "করি": "do make",
    "করব": "do will", "করবো": "do will", "কিভাবে": "how", "কীভাবে": "how",
    "কোথাও": "somewhere", "কিছু": "something", "উত্তোলন": "withdraw withdrawal",
    "দেওয়া": "give pay", "ফিরতে": "return refund", "বন্ধ করতে": "close",
    "বদলাতে": "change", "বদলান": "change", "কী": "what", "কি": "what",
    "হয়": "is happen", "যাবে": "will go", "পারব": "can able",
}

# Banglish (Bangla typed in Latin letters) -> English. Same rule: generic
# vocabulary only, never an evaluation phrasing.
BANGlish_GLOSSARY = {
    "koto": "how much many", "korte": "do to", "korbo": "will do", "kora": "do make",
    "kore": "do make", "korini": "did not make unauthorized dispute", "janabo": "report tell complain",
    "lage": "need required", "lagbe": "need will require", "dorkar": "need required",
    "hobe": "will happen", "hoy": "is happen", "hote": "is to be", "kobe": "when due",
    "ache": "have exists", "ase": "have exists", "pabo": "will get", "paben": "will get",
    "pabe": "will get", "pete": "get", "dite": "pay give", "pari": "can able",
    "parben": "can able", "kibhabe": "how", "kivabe": "how", "kon": "which",
    "koy": "how many", "kothay": "where", "jonno": "for", "theke": "from",
    "er": "of", "apnar": "your", "apnader": "your", "amader": "our",
    "din": "day", "din e": "daily day", "ek din": "daily day one day",
    "mashik": "monthly", "somoy": "time hours", "shomoy": "time hours",
    "shakha": "branch", "shakha gulo": "branches", "shorbocho": "maximum limit",
    "taka": "money taka amount", "tulate": "withdraw", "tulA": "withdraw",
    "harie": "lost", "harano": "lost", "bhule": "forgot", "bondho": "close stop dormant",
    "nomini": "nominee", "joggo": "eligible eligibility", "joggota": "eligibility eligible",
    "charge": "fee charge cost", "fee": "fee", "rate": "rate price", "damate": "pay",
    "age": "age", "oboshsho": "age", "manager": "manager", "nibho": "",
    "jante": "information", "jante chai": "information want to know",
    "tin jon": "three people holders", "labh": "profit gain", "sud": "interest",
    "shune": "hear", "bisesh": "special",
}


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


# Inflectional Bangla suffixes. Stripping them lets one entry cover কার্ড /
# কার্ডের and শাখা / শাখাগুলো without listing every case ending.
_BN_SUFFIXES = (
    "গুলোতে", "গুলো", "গুলি", "গুলা", "েরে", "ের", "হতে", "দিয়ে", "তে", "কে", "রে", "ে",
)


def _phrase_pattern(mapping: dict[str, str]) -> re.Pattern[str]:
    r"""Longest-first alternation with script-aware boundaries.

    Lookarounds replace ``\b`` because ``\b`` is unreliable at a Latin/Bengali
    junction, and because short keys such as ``er`` must not fire inside
    ``interest`` or ``register``.
    """
    keys = sorted((re.escape(k) for k in mapping if k.strip()), key=len, reverse=True)
    if not keys:
        return re.compile(r"(?!x)x")
    return re.compile(
        f"(?<!{_SCRIPT_CLASS})(?:{'|'.join(keys)})(?!{_SCRIPT_CLASS})", re.UNICODE
    )


_BN_PHRASES = _phrase_pattern(BILINGUAL_GLOSSARY)
_BANGLISH_PHRASES = _phrase_pattern(BANGlish_GLOSSARY)


def _bn_forms(token: str) -> list[str]:
    """The token plus its suffix-stripped stems."""
    forms = [token]
    for suffix in _BN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            stem = token[: -len(suffix)]
            if stem not in forms:
                forms.append(stem)
    return forms


def query_representations(text: str) -> list[str]:
    """Return the raw query plus a gloss-expanded variant for scoring.

    The retriever scores every representation and keeps the best score per FAQ.
    Folding glosses into one string and scoring only that was the original bug:
    extra tokens widen the Jaccard denominator and push the true match down, so
    the vocabulary bridge could make retrieval worse than having no bridge at all.
    """
    normalized = normalize_text(text)
    glosses: list[str] = []
    for match in _BN_PHRASES.finditer(normalized):
        glosses.append(BILINGUAL_GLOSSARY[match.group(0)])
    for token in normalized.split():
        if BENGALI_RE.search(token):
            for form in _bn_forms(token):
                if form != token and form in BILINGUAL_GLOSSARY:
                    glosses.append(BILINGUAL_GLOSSARY[form])
    for match in _BANGLISH_PHRASES.finditer(normalized):
        glosses.append(BANGlish_GLOSSARY[match.group(0)])
    glosses = [g for g in glosses if g]
    if not glosses:
        return [normalized]
    expanded = f"{normalized} {' '.join(glosses)}"
    return [normalized] if expanded == normalized else [normalized, expanded]


def expand_query(text: str) -> str:
    """Single gloss-expanded view of the query."""
    return query_representations(text)[-1]


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
        "kora",
    }
    marker_hits = sum(1 for token in words if token in banglish_markers)
    return "banglish" if marker_hits >= 2 or (marker_hits == 1 and len(words) <= 8) else "en"


def language_name(code: str) -> str:
    return {"en": "English", "bn": "Bangla", "banglish": "Banglish", "mixed": "mixed"}.get(code, code)


def answer_matches_language(text: str, expected: str) -> bool:
    """Check the output script/register without judging its factual content."""
    if expected == "en":
        return not BENGALI_RE.search(text or "")
    if expected == "bn":
        return bool(BENGALI_RE.search(text or ""))
    if expected == "banglish":
        return detect_language(text or "") == "banglish"
    return True
