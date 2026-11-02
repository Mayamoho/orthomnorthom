from __future__ import annotations

import csv
from pathlib import Path

from .models import FAQEntry


def load_faq(path: str | Path) -> list[FAQEntry]:
    entries: list[FAQEntry] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entries.append(
                FAQEntry(
                    faq_id=int(row["id"]),
                    topic=row["topic"].strip(),
                    question=row["question"].strip(),
                    answer=row["answer"].strip(),
                )
            )
    return entries
