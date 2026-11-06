from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Iterable

from .models import FAQEntry, RetrievalHit
from .normalize import detect_language, informative_terms, normalize_text, query_representations, tokenize


# Score floor of each channel on its own. The two retrieval modes produce scores
# on different scales, so a single constant cannot serve both: the fused
# dense+lexical score runs well above the lexical-only score.
CHANNEL_FLOOR = {"dense": 0.33, "lexical": 0.25}

# How much of the fused score the embedder gets, by the script the customer
# used. triBne-e5 is a Bangla/English model: it bridges Bangla script to English
# FAQ text, and English to English, but Roman-script Banglish is outside its
# training distribution — there its similarity is weak and undiscriminative, and
# at 0.62 it buried the glossary score that actually reads the query. That is
# why every Banglish question missed in the dense run while the lexical run got
# them right. Banglish therefore leans on the hand-written bridge, which is the
# auditable channel, and dense stays a tiebreaker.
DENSE_WEIGHT = {"en": 0.62, "bn": 0.62, "banglish": 0.25, "mixed": 0.40}

# A term no FAQ contains is not free: charging it nothing made a two-word
# overlap read as a full match, which is how a Banglish query about a disputed
# transaction landed on the dormant-account rule. Unmatched terms still weigh on
# the denominator, so a query the corpus can only partly answer cannot score as
# though every concept it raised was satisfied.
UNKNOWN_TERM_WEIGHT = 0.3

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
    """Reject answers that invent figures, drop the whole point, or contradict a required action.

    A reply may state no number the entry does not state — the card-PIN answer that
    invented "Fee holo BDT 500" was quoting the debit card fee, and a one-directional
    test would have called that grounded. Dropping is judged by how many figures the
    entry has: on a one- or two-figure entry ("SMS alerts cost a flat BDT 120") the
    figure is the answer, so dropping it is a failure; a seven-figure entry like the
    FDR rates can be answered correctly in two short sentences about one of them, and
    a reply that tried to recite all seven would be a worse answer.
    Keep the last check deliberately narrow: it only rejects a negated no-requirement
    claim when the source itself clearly contains a required action or document.
    """
    source_numbers = extract_numbers(source)
    answer_numbers = extract_numbers(answer)
    if not answer_numbers.issubset(source_numbers):
        return False
    if source_numbers and not answer_numbers:
        return False
    if len(source_numbers) <= 2 and not source_numbers.issubset(answer_numbers):
        return False

    source_text = normalize_text(source)
    answer_text = normalize_text(answer)
    has_requirements = bool(
        re.search(
            r"\b(?:need|required|must|submit|visit|bring|fill|sign|document|nid|passport|photo|form)\b",
            source_text,
        )
    )
    says_nothing_needed = bool(
        re.search(
            r"\b(?:nothing|no documents?|not required|no need)\b|"
            r"কিছু(?:ই|ও)? লাগবে না|কোনো কিছু লাগবে না|প্রয়োজন নেই",
            answer_text,
        )
    )
    return not (has_requirements and says_nothing_needed)


# Two FAQs a hair apart are a coin flip only when nothing separates them. The
# fused score can tie while the auditable channels still disagree, and a customer
# who named an entry's topic outright is not a coin flip: "Ek din e koto taka
# transfer kora jay?" tied with the ATM withdrawal rule on score, but only the
# transfer-limit entry has "transfer" in its topic.
TIE_MARGIN = 0.003
LEXICAL_LEAD = 0.02


def is_ambiguous(hits: list[RetrievalHit]) -> bool:
    """True when the two best entries cannot be separated on any signal."""
    if len(hits) < 2:
        return False
    best, second = hits[0], hits[1]
    if best.topic_hit > second.topic_hit:
        return False
    if best.lexical_score - second.lexical_score >= LEXICAL_LEAD:
        return False
    return best.score - second.score < TIE_MARGIN


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

    def dense_weight(self, language: str) -> float:
        """The embedder's share of the fused score for this query language."""
        if not self.dense_available:
            return 0.0
        return DENSE_WEIGHT.get(language, DENSE_WEIGHT["en"])

    def support_floor(self, language: str = "en") -> float:
        """The score a query must beat to be answered rather than handed off.

        Derived from the same blend as the score itself, so each language is
        judged against a threshold calibrated for the channel that carries it.
        """
        if not self.dense_available:
            return CHANNEL_FLOOR["lexical"]
        weight = self.dense_weight(language)
        return weight * CHANNEL_FLOOR["dense"] + (1 - weight) * CHANNEL_FLOOR["lexical"]

    def _passage(self, text: str) -> str:
        return f"passage: {text}" if self._prefixed else text

    def _query(self, text: str) -> str:
        return f"query: {text}" if self._prefixed else text

    def _lexical(self, concepts: set[str], char_sets: Iterable[set[str]], index: int) -> tuple[float, float]:
        doc_words = self._doc_words[index]
        # Judge the query on the words a customer cannot drop: postpositions and
        # copulas are so rare in an English FAQ that their IDF made them look
        # like the strongest evidence in the sentence.
        if not concepts:
            return 0.0, 0.0
        shared = concepts & doc_words
        if not shared:
            return 0.0, 0.0
        # IDF-weighted query coverage: how much of the query's information need
        # the document answers. Jaccard divides by the union, which for a
        # 40-word English FAQ against a 6-term Bangla query caps even a perfect
        # concept match near 0.12. Coverage divides by the query instead, and
        # the IDF weighting stops a word like "account" from carrying a match.
        query_mass = sum(self._idf.get(term, UNKNOWN_TERM_WEIGHT) for term in concepts)
        matched_mass = sum(self._idf.get(term, 0.0) for term in shared)
        coverage = matched_mass / query_mass if query_mass else 0.0
        # Surface similarity still reads the raw query, so a Bangla question can
        # reach the FAQ's own Bangla wording without the gloss terms diluting it.
        char = max((_jaccard(grams, self._doc_chars[index]) for grams in char_sets), default=0.0)
        # A query that names the entry's topic outright is the tie-breaker when
        # two entries share most of their vocabulary ("branch" for the opening
        # hours entry, not the cheque book entry that mentions a branch once).
        topic_words = self._topic_words[index]
        topic_hit = len(concepts & topic_words) / len(topic_words) if topic_words else 0.0
        return 0.66 * coverage + 0.24 * char + 0.10 * topic_hit, topic_hit

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        weight = self.dense_weight(detect_language(query))
        representations = [rep for rep in query_representations(query) if rep]
        concepts = informative_terms(representations[-1])
        char_sets = [_char_ngrams(rep) for rep in representations]

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
            lexical, topic_hit = self._lexical(concepts, char_sets, index)
            dense = dense_scores[index]
            score = weight * dense + (1 - weight) * lexical if dense is not None else lexical
            hits.append(
                RetrievalHit(
                    entry=entry, score=score, dense_score=dense, lexical_score=lexical, topic_hit=topic_hit
                )
            )

        # Ordering on the fused score alone put two FAQs a hair apart in an
        # arbitrary sequence, so ties fall back to the signals that are auditable:
        # whether the customer named the topic, then the lexical channel.
        hits.sort(key=lambda hit: (hit.score, hit.topic_hit, hit.lexical_score), reverse=True)
        for rank, hit in enumerate(hits[:top_k], start=1):
            hit.rank = rank
        return hits[:top_k]
