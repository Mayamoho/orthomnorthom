from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from src.assistant import BankAssistant
from src.data import load_faq


ROOT = Path(__file__).resolve().parent
UNKNOWN_IDS = {11, 13, 18, 19, 21, 32, 33, 36, 37, 39}
EXPECTED_SOURCE = {
    1: 19, 2: 8, 3: 24, 4: 17, 5: 23, 6: 22, 7: 7, 8: 11, 9: 25, 10: 27,
    12: 16, 14: 9, 15: 14, 16: 1, 17: 28, 20: 2, 22: 18, 23: 3, 24: 5,
    25: 10, 26: 15, 27: 4, 28: 6, 29: 26, 30: 20, 31: 21, 34: 12, 35: 13,
    38: 30, 40: 29,
}


def main() -> int:
    assistant = BankAssistant(load_faq(ROOT / "data" / "bank-faq.csv"))
    rows = []
    with (ROOT / "data" / "test-questions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = int(row["id"])
            started = time.perf_counter()
            result = assistant.answer(row["question"])
            elapsed = (time.perf_counter() - started) * 1000
            expected_unknown = question_id in UNKNOWN_IDS
            refusal_correct = (result.status == "handoff") == expected_unknown
            source_correct = expected_unknown or result.source_id == EXPECTED_SOURCE.get(question_id)
            rows.append({
                "id": question_id,
                "question": row["question"],
                "expected_unknown": expected_unknown,
                "status": result.status,
                "source_id": result.source_id,
                "expected_source_id": EXPECTED_SOURCE.get(question_id),
                "source_correct": source_correct,
                "refusal_correct": refusal_correct,
                "language": result.language,
                "support_type": result.support_type,
                "confidence": result.confidence,
                "latency_ms": elapsed,
                "answer": result.answer,
            })
            print(f"{question_id:02d} {'OK' if refusal_correct and source_correct else 'MISS'}  {result.status:8s} source={result.source_id} lang={result.language}  {row['question']}")

    answered = [row for row in rows if not row["expected_unknown"]]
    unknown = [row for row in rows if row["expected_unknown"]]
    report = {
        "total": len(rows),
        "answerable": len(answered),
        "unknown": len(unknown),
        "source_accuracy": sum(row["source_correct"] for row in answered) / max(1, len(answered)),
        "refusal_accuracy": sum(row["refusal_correct"] for row in rows) / max(1, len(rows)),
        "overall_gate_accuracy": sum(row["source_correct"] and row["refusal_correct"] for row in rows) / max(1, len(rows)),
        "median_latency_ms": sorted(row["latency_ms"] for row in rows)[len(rows) // 2],
        "rows": rows,
    }
    output = ROOT / "reports" / "offline_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
