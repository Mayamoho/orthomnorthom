from __future__ import annotations

import re
import time

from .generator import LocalGenerator, refusal
from .models import AssistantResult, FAQEntry, RetrievalHit
from .normalize import BENGALI_RE, answer_matches_language, detect_language, query_representations, tokenize
from .retrieval import FAQRetriever, extract_numbers, preserves_key_facts

# Concepts a customer might ask about that have zero presence in the supplied
# FAQ. The guard below drops any term the FAQ does mention, so this list can
# only ever refuse a question the corpus genuinely cannot answer: market and
# commodity prices, securities, foreign exchange, crypto, immigration, branch
# locations, named staff, and other people's accounts. A co-applicant such as a
# spouse is deliberately absent: naming who you would open an account with is a
# question the joint-account rule answers, whereas asking for someone else's
# balance is a data lookup the FAQ can never support.
UNCOVERED_CONCEPTS = (
    "gold", "silver", "commodity", "price", "crypto", "cryptocurrency", "bitcoin",
    "trading", "dollar", "euro", "forex", "exchange", "share", "stock", "invest",
    "dividend", "profit", "visa", "immigration", "work permit", "manager",
    "officer", "friend", "relative", "waive", "waiver",
    "exempt", "discount", "singapore", "dubai", "london", "sylhet", "chattogram",
    "chittagong", "khulna", "rajshahi", "barishal", "rangpur", "mymensingh",
    "স্বর্ণ", "রুপা", "দাম", "ক্রিপ্টো", "ডলার", "বিনিময়", "শেয়ার", "স্টক", "লাভ",
    "ভিসা", "ম্যানেজার", "বন্ধু", "সিঙ্গাপুর", "দুবাই", "সিলেট", "চট্টগ্রাম",
)

# Gloss output is deliberately generic; these words describe grammar, not an
# information need, so they must not count as evidence of a generalization.
_FUNCTION_GLOSSES = {
    "is", "to", "be", "do", "make", "will", "happen", "have", "exists", "get",
    "receive", "pay", "give", "how", "much", "many", "what", "where", "when",
    "due", "of", "for", "from", "your", "our", "any", "now", "today", "day",
    "days", "daily", "per", "every", "can", "able", "need", "required", "report",
    "tell", "notify", "information", "want", "know", "something", "somewhere",
    "lost", "forgot", "change", "update", "call", "which", "that", "this", "and",
}


class BankAssistant:
    def __init__(self, entries: list[FAQEntry], retriever: FAQRetriever | None = None, generator: LocalGenerator | None = None):
        self.entries = entries
        self.retriever = retriever or FAQRetriever(entries)
        self.generator = generator or LocalGenerator()
        self._faq_vocab = {term for entry in entries for term in tokenize(entry.searchable_text)}
        self._uncovered_concepts = self._active_concepts(entries)

    @staticmethod
    def _active_concepts(entries: list[FAQEntry]) -> tuple[str, ...]:
        """Keep only concepts the FAQ never mentions.

        A term the corpus does use is dropped, so adding a market-data FAQ would
        automatically stop this guard from refusing market questions.
        """
        vocabulary = {term for entry in entries for term in tokenize(entry.searchable_text)}
        return tuple(concept for concept in UNCOVERED_CONCEPTS if concept not in vocabulary)

    def answer(self, question: str) -> AssistantResult:
        started = time.perf_counter()
        language = detect_language(question)
        hits = self.retriever.search(question, top_k=5)
        best = hits[0] if hits else None
        if best is None:
            return self._handoff(language, 0.0, started, hits, "no_candidates")

        blocked = self._support_reason(question, best, hits)
        if blocked:
            return self._handoff(language, best.score, started, hits, blocked)

        model_support = self.generator.supports(question, best.entry)
        if model_support is False:
            return self._handoff(language, best.score, started, hits, "faq_entailment_rejected")

        answer = self.generator.answer(question, best.entry, language)
        if self.generator.available and not answer_matches_language(answer, language):
            return self._handoff(language, best.score, started, hits, "language_generation_failed")
        grounded = preserves_key_facts(best.entry.answer, answer)
        if not grounded:
            answer = self.generator._fallback(best.entry, language)
        generalization = self._is_safe_generalization(question, best.entry)
        return AssistantResult(
            answer=answer,
            language=language,
            status="answered",
            support_type="safe_generalization" if generalization else "direct_match",
            source_id=best.entry.faq_id,
            confidence=best.score,
            latency_ms=(time.perf_counter() - started) * 1000,
            hits=hits,
            metadata={
                "mode": self.retriever.mode,
                "grounding_verified": grounded,
                "answer_form": "model" if self.generator.available else "faq_verbatim",
            },
        )

    def _handoff(self, language: str, confidence: float, started: float, hits: list[RetrievalHit], reason: str) -> AssistantResult:
        return AssistantResult(
            answer=refusal(language), language=language, status="handoff",
            support_type=reason, source_id=None, confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000, hits=hits,
            metadata={"mode": self.retriever.mode},
        )

    def _support_reason(self, question: str, best: RetrievalHit, hits: list[RetrievalHit]) -> str | None:
        """Return why the retrieved FAQ is not enough, or None when it is."""
        if best.score < self.retriever.support_floor:
            return "below_threshold"

        # Match on the customer's own words only. Gloss output is ours, so a
        # gloss such as rate -> "rate price" must not trip the price guard.
        raw = query_representations(question)[0]
        for concept in self._uncovered_concepts:
            if len(concept) >= 3 and concept in raw:
                return "topic_not_in_faq"

        asked = extract_numbers(question)
        if asked and self._numeric_conflict(asked, best.entry):
            # The customer cites a figure that contradicts what the FAQ states,
            # so any answer about it would be invented rather than grounded.
            return "figure_not_in_faq"

        if len(hits) > 1 and best.score - hits[1].score < 0.003:
            # Two FAQs are indistinguishable: guessing would be a coin flip.
            return "ambiguous_evidence"
        return None

    @staticmethod
    def _numeric_conflict(asked: set[str], entry: FAQEntry) -> bool:
        """True when a figure the customer gave contradicts a range the FAQ states.

        Only explicit "A to B" ranges are used, and only for numbers of the
        same order of magnitude as that range, so an income figure is never
        compared against an age range. Everything subtler than a range
        violation is left to the model's entailment check rather than to regex
        arithmetic, because getting it wrong here silently refuses real
        customers.
        """
        stated = extract_numbers(entry.searchable_text)
        if asked <= stated:
            return False
        ranges = [
            (int(low.replace(",", "")), int(high.replace(",", "")))
            for low, high in re.findall(r"(\d[\d,]*) to (\d[\d,]*)", entry.searchable_text)
        ]
        for value in asked:
            if value in stated:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            for low, high in ranges:
                if low / 10 <= number <= high * 10:
                    if not low <= number <= high:
                        return True
                    break
        return False

    def _is_safe_generalization(self, question: str, entry: FAQEntry) -> bool:
        """True when the question names something the FAQ does not name.

        Replaces the earlier hard-coded spouse/FAQ-14 special case with a
        generic check: the retrieved rule still applies, but the customer
        introduced an entity the corpus never mentions.
        """
        entry_text = entry.searchable_text.lower()
        for surface in query_representations(question):
            for term in tokenize(surface):
                if BENGALI_RE.search(term) or len(term) < 4 or term in _FUNCTION_GLOSSES:
                    continue
                if term not in entry_text:
                    return True
        return False
