from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Iterable

from .models import FAQEntry, RetrievalHit
from .normalize import normalize_text, tokenize


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    text = f"  {normalize_text(text)}  "
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class FAQRetriever:
    def __init__(self, entries: Iterable[FAQEntry], embedding_model: str | None = None):
        self.entries = list(entries)
        self._dense = None
        self._doc_embeddings = None
        model_name = embedding_model or os.getenv("BIAI_EMBEDDING_MODEL", "istiaqfuad/triBne-e5-small")
        if os.getenv("BIAI_DISABLE_DENSE", "0") != "1":
            try:
                from sentence_transformers import SentenceTransformer

                self._dense = SentenceTransformer(model_name)
                docs = [entry.searchable_text for entry in self.entries]
                self._doc_embeddings = self._dense.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            except Exception:
                self._dense = None
                self._doc_embeddings = None

    @property
    def dense_available(self) -> bool:
        return self._dense is not None and self._doc_embeddings is not None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        q_norm = normalize_text(query)
        q_words = set(tokenize(query))
        q_chars = _char_ngrams(query)
        dense_scores: list[float] | None = None
        if self.dense_available:
            query_embedding = self._dense.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            dense_scores = [float(value) for value in self._doc_embeddings @ query_embedding]

        hits: list[RetrievalHit] = []
        for idx, entry in enumerate(self.entries):
            doc_words = set(tokenize(entry.searchable_text))
            lexical = 0.65 * _jaccard(q_words, doc_words) + 0.35 * _jaccard(q_chars, _char_ngrams(entry.searchable_text))
            dense = dense_scores[idx] if dense_scores is not None else None
            # Dense scores and lexical scores have different scales; rank fusion is stable on small FAQ sets.
            score = (0.72 * dense + 0.28 * lexical) if dense is not None else lexical
            hits.append(RetrievalHit(entry=entry, score=score, dense_score=dense, lexical_score=lexical))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        for rank, hit in enumerate(hits[:top_k], start=1):
            hit.rank = rank
        return hits[:top_k]


def extract_numbers(text: str) -> set[str]:
    normalized = (text or "").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    return set(re.findall(r"\d+(?:\.\d+)?", normalized))


def preserves_key_facts(source: str, answer: str) -> bool:
    source_numbers = extract_numbers(source)
    return source_numbers.issubset(extract_numbers(answer))
