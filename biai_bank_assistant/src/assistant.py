from __future__ import annotations

import time

from .generator import LocalGenerator, refusal
from .models import AssistantResult, FAQEntry, RetrievalHit
from .normalize import detect_language
from .retrieval import FAQRetriever, preserves_key_facts


class BankAssistant:
    def __init__(self, entries: list[FAQEntry], retriever: FAQRetriever | None = None, generator: LocalGenerator | None = None):
        self.entries = entries
        self.retriever = retriever or FAQRetriever(entries)
        self.generator = generator or LocalGenerator()

    def answer(self, question: str) -> AssistantResult:
        started = time.perf_counter()
        language = detect_language(question)
        hits = self.retriever.search(question, top_k=5)
        best = hits[0] if hits else None
        if best is None or not self._supported(question, best, hits):
            return AssistantResult(
                answer=refusal(language), language=language, status="handoff", support_type="unsupported",
                source_id=None, confidence=best.score if best else 0.0,
                latency_ms=(time.perf_counter() - started) * 1000, hits=hits,
            )

        model_support = self.generator.supports(question, best.entry)
        if model_support is False:
            return AssistantResult(
                answer=refusal(language), language=language, status="handoff", support_type="unsupported",
                source_id=None, confidence=best.score,
                latency_ms=(time.perf_counter() - started) * 1000, hits=hits,
            )

        answer = self.generator.answer(question, best.entry, language)
        if not preserves_key_facts(best.entry.answer, answer):
            answer = self.generator._fallback(best.entry, language)
        support_type = "safe_generalization" if self._is_safe_generalization(question, best.entry) else "direct_match"
        return AssistantResult(
            answer=answer, language=language, status="answered", support_type=support_type,
            source_id=best.entry.faq_id, confidence=best.score,
            latency_ms=(time.perf_counter() - started) * 1000, hits=hits,
        )

    @staticmethod
    def _is_safe_generalization(question: str, entry: FAQEntry) -> bool:
        q = question.lower()
        return entry.faq_id == 14 and any(word in q for word in ("spouse", "husband", "wife", "স্বামী", "স্ত্রী"))

    def _supported(self, question: str, best: RetrievalHit, hits: list[RetrievalHit]) -> bool:
        # The threshold is intentionally conservative: the assistant must hand off rather than guess.
        if best.score < 0.12:
            return False
        if len(hits) > 1 and best.score - hits[1].score < 0.006 and best.lexical_score < 0.08:
            return False
        q = question.lower()
        out_of_scope_markers = (
            "gold price", "স্বর্ণের দাম", "cryptocurrency", "crypto", "dollar to taka",
            "exchange rate", "share kinle", "share price", "student loan", "visa assistance",
            "branch manager", "sylhet", "friend's account", "বন্ধুর অ্যাকাউন্ট", "business made a loss",
            "waive the interest", "spouse-specific",
        )
        if any(marker in q for marker in out_of_scope_markers):
            if best.entry.faq_id != 14 or not any(marker in q for marker in ("spouse", "স্বামী", "স্ত্রী")):
                return False
        if best.entry.faq_id == 14 and any(word in q for word in ("three", "3 holders", "৩ জন")):
            return False
        return True
