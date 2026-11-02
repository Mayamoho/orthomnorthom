from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FAQEntry:
    faq_id: int
    topic: str
    question: str
    answer: str

    @property
    def searchable_text(self) -> str:
        return f"{self.topic}. {self.question}. {self.answer}"


@dataclass
class RetrievalHit:
    entry: FAQEntry
    score: float
    dense_score: float | None = None
    lexical_score: float = 0.0
    rank: int = 0


@dataclass
class AssistantResult:
    answer: str
    language: str
    status: str
    support_type: str
    source_id: int | None
    confidence: float
    latency_ms: float
    hits: list[RetrievalHit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
