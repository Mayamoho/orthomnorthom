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
    # `disputed`, not `dispute`: the English side does no stemming, so a gloss in
    # the wrong form simply never matches the corpus surface it was written for.
    "করিনি": "did not make unauthorized disputed", "জানাবো": "report tell complain",
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
    # রাখা ("to keep") is what separates a minimum-balance question from a
    # minimum-deposit one, and without it the two savings FAQs tied.
    "রাখতে": "keep balance maintain", "রাখা": "keep balance maintain",
    "রাখুন": "keep balance maintain", "রাখে": "keep balance maintain",
    # Loanword spellings. Customers write ব্যক্তিগত ঋণ as পার্সোনাল লোন in Bangla
    # script, and the literary form alone left the informative term of a query
    # like "পার্সোনাল লোনের জন্য কী যোগ্যতা লাগে?" unmatched, so the score was
    # carried by the generic কী and জন্য that every question alias contains.
    "পার্সোনাল": "personal", "ইনকাম": "income salary", "স্যালারি": "salary income",
    "লাগে": "need required", "লাগবে": "need required will", "প্রয়োজন": "need required",
    "অনুমতি": "allow permission", "সীমা": "limit maximum", "নিয়ম": "rule how",
    "ছবি": "photo photos", "ফরম": "form", "ভেরিফিকেশন": "verification",
    "স্বাক্ষর": "sign", "মেয়াদ": "tenure maturity period",
}

# Banglish (Bangla typed in Latin letters) -> English. Same rule: generic
# vocabulary only, never an evaluation phrasing.
BANGlish_GLOSSARY = {
    "koto": "how much many", "korte": "do to", "korbo": "will do", "kora": "do make",
    "kore": "do make", "korini": "did not make unauthorized disputed", "janabo": "report tell complain",
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
    "taka": "money taka amount", "tulate": "withdraw", "tula": "withdraw",
    "harie": "lost", "harano": "lost", "bhule": "forgot", "bondho": "close stop dormant",
    "nomini": "nominee", "joggo": "eligible eligibility", "joggota": "eligibility eligible",
    "charge": "fee charge cost", "fee": "fee", "rate": "rate price", "damate": "pay",
    "age": "age", "oboshsho": "age", "manager": "manager", "nibho": "",
    "jante": "information", "jante chai": "information want to know",
    "tin jon": "three people holders", "labh": "profit gain", "sud": "interest",
    "shune": "hear", "bisesh": "special",
}

# Banglish evidence for detect_language, derived from the glossary so the bridge
# and the detector cannot drift apart. The old hand-list fitted questions only,
# which is why a model-written Banglish *answer* scored as English and tripped
# the language gate. Keys that are ordinary English words cannot be evidence of
# Banglish, so they are dropped here rather than from the glossary.
_ENGLISH_SHADOWED = frozenset({"age", "fee", "rate", "charge", "manager"})

# Copulas, verbs and postpositions an answer uses that a question rarely needs.
_ANSWER_MARKERS = frozenset({
    "ami", "amra", "apni", "tumi", "tomra", "gele", "leo", "hote", "hole", "thake",
    "ason", "asi", "ele", "bole", "bolte", "bolun", "korun", "koren", "koriye",
    "lagse", "lagen", "mone", "gulo", "gula", "shob", "sob", "khola", "notun",
    "puron", "chara", "ebong", "shathe", "sathe", "jani", "janina", "janen",
    "parbi", "parbe", "parben", "parchi", "hoben", "dilen", "niye", "karon",
    "kintu", "abar", "aro", "shudhu", "achhe", "achi", "jache", "kinle", "dile",
})

# Single letters and shared words: never enough on their own, but two of them in
# one sentence is not a coincidence an English sentence produces. Listing a word
# here keeps it weak even when the glossary also carries it, because "taka",
# "din" and "er" appear inside English sentences about Bangladesh too.
_BANGLISH_WEAK = frozenset({"e", "o", "ki", "na", "ta", "er", "din", "taka", "bochor", "age"})

BANGLISH_STRONG = (
    frozenset(
        token.lower() for token in BANGlish_GLOSSARY
        if " " not in token and token not in _ENGLISH_SHADOWED
    )
    | _ANSWER_MARKERS
) - _BANGLISH_WEAK


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


# Postpositions, copulas and question words. They carry no information need, yet
# an English-dominant FAQ gives them a high IDF, so a two-word Bangla overlap used
# to outrank a real concept match: "পার্সোনাল লোনের জন্য কী যোগ্যতা লাগে?" scored
# above the personal-loan rule because the account-opening entry shared কী and
# জন্য. They are removed from the query side only, and their English glosses
# remain, so the concept still reaches the FAQ.
#
# The helper verbs (করতে/করা/লাগবে/চাই/হয় ...) are here for the same reason in
# the other direction: they occur in almost every Bangla question alias, so each
# one looked like a strong match while naming no product at all. The English half
# covers the words this file's own glossary emits (do, make, will, need, what),
# which are artifacts of the bridge rather than evidence of an information need.
QUERY_STOPWORDS = frozenset(
    """কী কি জন্য ও এবং এর থেকে তা তো আবার আছে আছেন হবে হলে যদি তবে এই সেই এটি আর শুধু
    মধ্যে হতে আমি আমার আপনি আপনার আপনাদের আমাদের তোমার কিছু কোন
    করতে করা করে করি করছে করলো করবে করবেন করব করবো করিয়া করা হয় হওয়া হয়ে হয়েছে হচ্ছে
    লাগে লাগবে লাগত লাগতে দরকার চাই চায় চান চান্ পায় পাবে পারি পারবে পারবেন
    যা যার যখন যেখানে সব শুধু মাত্র অথবা কিন্তু অতএব তাই তবেই
    ki na ta e er o the a is of to for am are
    do make will what how much many need required want know get give pay receive able can
    happen exist information one day money amount""".split()
)


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


def _bn_gloss_terms(token: str) -> list[str]:
    """The English words the glossary offers for a Bangla token, stems included."""
    glosses: list[str] = []
    value = BILINGUAL_GLOSSARY.get(token)
    if value:
        glosses.extend(value.split())
    for form in _bn_forms(token):
        if form == token:
            continue
        value = BILINGUAL_GLOSSARY.get(form)
        if value:
            glosses.extend(value.split())
    return glosses


def informative_terms(text: str) -> set[str]:
    """The query terms worth scoring, deduplicated by concept.

    Stopwords are dropped, and so is a Bangla word whose English gloss the same
    query already carries. Both halves of one concept counted as two pieces of
    evidence: the loan-interest entry matched "পার্সোনাল লোনের" *and* "personal
    loan" and so beat the eligibility entry, which shares only the glossed half,
    on a question that was asking about eligibility.
    """
    terms = {term for term in tokenize(text) if term not in QUERY_STOPWORDS}
    covered = {
        term
        for term in terms
        if BENGALI_RE.search(term) and any(gloss in terms for gloss in _bn_gloss_terms(term))
    }
    return terms - covered


def query_representations(text: str) -> list[str]:
    """Return the raw query plus a gloss-expanded variant for scoring.

    The retriever scores every representation and keeps the best score per FAQ.
    Folding glosses into the only view scored was the original bug: extra tokens
    widen a Jaccard denominator and push the true match down, so the vocabulary
    bridge could make retrieval worse than having no bridge at all. The two
    views now do different jobs: the raw one measures surface similarity against
    the FAQ's own Bangla text, and the expanded one carries concept coverage.
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
    strong = sum(1 for token in words if token in BANGLISH_STRONG)
    weak = sum(1 for token in words if token in _BANGLISH_WEAK)
    return "banglish" if strong >= 1 or weak >= 2 else "en"


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


# ---------------------------------------------------------------------------
# Format guards. The evaluator scores register, but a reply that repeats the
# customer's own question, stops in the middle of a word, or invents Bengali
# morphology that no Bangla speaker wrote is not usable even when it passes the
# script check. These are the tests for that, used both to clean a model reply
# and to decide when to discard it.
# ---------------------------------------------------------------------------

_PREFIX_NOISE_RE = re.compile(
    r"^(?:\s*(?:answer|reply|উত্তর|faq answer)\s*[:\-–]?\s*)+", re.IGNORECASE
)
# The same label, once the model has put it after a newline or after a recited
# question, where a start-of-string rule cannot reach it (evaluation id 17).
_INLINE_LABEL_RE = re.compile(r"(?:^|\s)(?:answer|reply|উত্তর)\s*[:\-–]\s*", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[।.!?…][\"'\)\]]?\s*$")
# Words short enough to be case endings, currencies or names; never judged.
_STOP_TERMS = frozenset({"ba", "the", "and", "for", "with", "from", "eci", "ge", "ni", "ta", "ti", "ki"})


def content_terms(text: str) -> list[str]:
    """Judgable words: no bare digits, no currency or product fragments."""
    return [term for term in tokenize(text) if not term.isdigit() and len(term) >= 3 and term not in _STOP_TERMS]


def strip_answer_artifacts(text: str) -> str:
    """Drop label prefixes and stray inline labels, keeping the sentence itself."""
    answer = _PREFIX_NOISE_RE.sub("", (text or "").strip())
    answer = _INLINE_LABEL_RE.sub(" ", answer)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer


def echoes_question(answer: str, question: str) -> bool:
    """True when the reply opens by reciting the question it was asked.

    Small models answer a prompt that ends with the customer's line by continuing
    that line, so the copy sits at the front. A shared phrase further down is not
    an echo: the bank's own wording for the debit card fee legitimately contains
    "debit card er annual fee", and judging that as a repeat threw away correct
    answers. Two rules survive that: the question recited *in full* is always an
    echo, however short ("…kibhabe of will do how?", "…koto: 500 BDT"), and a long
    question needs four shared words before anything is a recital. Fewer than four
    is how a Banglish reply to the personal-loan rate question, which legitimately
    opens with "Personal loan-er", got misjudged.
    """
    a_terms = tokenize(answer)
    q_terms = tokenize(question)
    if not q_terms:
        return False
    shared = 0
    for left, right in zip(a_terms, q_terms):
        if left != right:
            break
        shared += 1
    if shared >= 2 and shared >= len(q_terms):
        return True
    return shared >= 4 and shared >= 0.5 * len(q_terms)


def looks_truncated(answer: str) -> bool:
    """True when the reply has no sentence end, i.e. it was cut mid-word.

    Bengali is token-expensive on Latin-oriented vocabularies, so a short token
    budget silently amputates the last word and leaves a dangling matra.
    """
    text = (answer or "").strip()
    return not text or not _TERMINAL_RE.search(text)


def reuses_faq_wording(answer: str, source: str, question: str = "", min_share: float = 0.6) -> bool:
    """True when the reply is built from this entry's own words.

    The FAQ is English, so this is a test only an English reply can be judged by:
    it catches the boilerplate a small model falls back to ("this product has no
    investment feature") while keeping a genuinely re-used answer, and the
    customer's own words count as fair game. It is deliberately not applied to
    Bangla or Banglish, where a faithful reply shares no surface word with the
    entry and the test would reject good answers.
    """
    terms = set(content_terms(answer))
    if not terms:
        return False
    allowed = set(content_terms(source)) | set(content_terms(question))
    return len(terms & allowed) / len(terms) >= min_share


def answer_is_wellformed(
    answer: str,
    language: str,
    question: str,
    source: str = "",
    min_share: float = 0.6,
) -> bool:
    """Shape, script and, where it can be judged, fidelity to the bank's wording.

    The script a reply must be in is the script the *customer* used. The FAQ is
    English, so for Bangla and Banglish the wording cannot be compared word for
    word; what carries across scripts is the shape (a finished sentence that is not
    the customer's question), the script, and `preserves_key_facts` on the figures.
    """
    if not answer or looks_truncated(answer) or echoes_question(answer, question):
        return False
    if not answer_matches_language(answer, language):
        return False
    if language == "en" and source and not reuses_faq_wording(answer, source, question, min_share):
        return False
    return True
