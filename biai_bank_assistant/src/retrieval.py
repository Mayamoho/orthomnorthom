from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Iterable

from .models import FAQEntry, RetrievalHit
from .normalize import normalize_text, query_representations, tokenize


# Score floors for the support gate. The two retrieval modes produce scores on
# different scales, so a single constant cannot serve both: the fused
# dense+lexical score runs well above the lexical-only score.
SUPPORT_FLOOR = {"dense": 0.30, "lexical": 0.25}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d{3})*(?:\.\d+)?")


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    text = f"  {normalize_text(text)}  "
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _idf(doc_token_sets: list[set[str]]) -> dict[str, float]:
    total = len(doc_token_sets)
    document_frequency: Counter[str] = Counter()
    for tokens in doc_token_sets:
        document_frequency.update(tokens)
    return {
        term: math.log(1 + (total - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }


def extract_numbers(text: str) -> set[str]:
    """Pull numeric facts out, treating 1,000 and 1000 as the same value."""
    normalized = (text or "").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    return {match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(normalized)}


def preserves_key_facts(source: str, answer: str) -> bool:
    return extract_numbers(source).issubset(extract_numbers(answer))


class FAQRetriever:
    def __init__(self, entries: Iterable[FAQEntry], embedding_model: str | None = None):
        self.entries = list(entries)
        self._doc_words = [set(tokenize(entry.searchable_text)) for entry in self.entries]
        self._doc_chars = [_char_ngrams(entry.searchable_text) for entry in self.entries]
        self._topic_words = [set(tokenize(entry.topic)) for entry in self.entries]
        self._idf = _idf(self._doc_words)
        self._dense = None
        self._doc_embeddings = None
        model_name = embedding_model or os.getenv("BIAI_EMBEDDING_MODEL", "istiaqfuad/triBne-e5-small")
        # e5 models are instruction-tuned for retrieval: encoding without the
        # "query: " and "passage: " prefixes degrades their similarity scores.
        self._prefixed = "e5" in model_name.lower()
        if os.getenv("BIAI_DISABLE_DENSE", "0") != "1":
            try:
                from sentence_transformers import SentenceTransformer

                self._dense = SentenceTransformer(model_name)
                docs = [self._passage(entry.searchable_text) for entry in self.entries]
                self._doc_embeddings = self._dense.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            except Exception:
                self._dense = None
                self._doc_embeddings = None

    @property
    def dense_available(self) -> bool:
        return self._dense is not None and self._doc_embeddings is not None

    @property
    def mode(self) -> str:
        return "dense" if self.dense_available else "lexical"

    @property
    def support_floor(self) -> float:
        return SUPPORT_FLOOR[self.mode]

    def _passage(self, text: str) -> str:
        return f"passage: {text}" if self._prefixed else text

    def _query(self, text: str) -> str:
        return f"query: {text}" if self._prefixed else text

    def _lexical(self, q_words: set[str], q_chars: set[str], index: int) -> float:
        doc_words = self._doc_words[index]
        shared = q_words & doc_words
        if not shared:
            return 0.0
        # IDF-weighted query coverage: how much of the query's information need
        # the document answers. Jaccard divides by the union, which for a
        # 40-word English FAQ against a 6-term Bangla query caps even a perfect
        # concept match near 0.12. Coverage divides by the query instead, and
        # the IDF weighting stops a word like "account" from carrying a match.
        query_mass = sum(self._idf.get(term, 0.0) for term in q_words)
        matched_mass = sum(self._idf.get(term, 0.0) for term in shared)
        coverage = matched_mass / query_mass if query_mass else 0.0
        # A query that names the entry's topic outright is the tie-breaker when
        # two entries share most of their vocabulary ("branch" for the opening
        # hours entry, not the cheque book entry that mentions a branch once).
        topic_words = self._topic_words[index]
        topic_hit = len(q_words & topic_words) / len(topic_words) if topic_words else 0.0
        return 0.66 * coverage + 0.24 * _jaccard(q_chars, self._doc_chars[index]) + 0.10 * topic_hit

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        representations = [rep for rep in query_representations(query) if rep]
        views = [(set(tokenize(rep)), _char_ngrams(rep)) for rep in representations]

        dense_scores: list[float | None] = [None] * len(self.entries)
        if self.dense_available:
            embeddings = self._dense.encode(
                [self._query(rep) for rep in representations],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for embedding in embeddings:
                for index, value in enumerate(self._doc_embeddings @ embedding):
                    if dense_scores[index] is None or float(value) > dense_scores[index]:
                        dense_scores[index] = float(value)

        hits: list[RetrievalHit] = []
        for index, entry in enumerate(self.entries):
            lexical = max(self._lexical(words, chars, index) for words, chars in views)
            dense = dense_scores[index]
            score = (0.62 * dense + 0.38 * lexical) if dense is not None else lexical
            hits.append(RetrievalHit(entry=entry, score=score, dense_score=dense, lexical_score=lexical))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        for rank, hit in enumerate(hits[:top_k], start=1):
            hit.rank = rank
        return hits[:top_k]
