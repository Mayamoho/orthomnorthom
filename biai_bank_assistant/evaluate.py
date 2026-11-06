from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from src.assistant import BankAssistant
from src.data import load_faq
from src.normalize import answer_is_wellformed, answer_matches_language, echoes_question, looks_truncated


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
    entries_by_id = {entry.faq_id: entry for entry in assistant.entries}
    # LocalGenerator loads lazily. Load it before recording the report status so
    # generator_loaded reflects the actual model-stack run.
    assistant.generator.load()
    mode = assistant.retriever.mode
    model_loaded = assistant.generator.available
    print(f"retrieval mode: {mode} | generator: {'loaded' if model_loaded else 'not loaded (deterministic fallback)'}")
    if not model_loaded:
        print(f"generator status: {assistant.generator.load_error or 'dependencies absent - answers are verbatim FAQ text in English'}")
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
            # Script alone is not enough: a reply that recites the customer's own
            # question back, or stops in the middle of a word, passes the script
            # test and is still not something to show a customer. These are the
            # same gates the assistant applies before serving a model reply,
            # re-run on what was actually served so the report cannot flatter
            # the pipeline that produced it.
            entry = entries_by_id.get(result.source_id)
            source_text = entry.answer if entry else ""
            # A hand-off is one of the four fixed refusal sentences, not a reply
            # grounded in one entry, so it has no source to be faithful to. The
            # wording test is left out for it and the shape and script tests still
            # run.
            format_correct = answer_is_wellformed(
                result.answer,
                result.language,
                row["question"],
                source_text if result.status == "answered" else "",
            )
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
                "language_correct": answer_matches_language(result.answer, result.language),
                "echo_free": not echoes_question(result.answer, row["question"]),
                "complete": not looks_truncated(result.answer),
                "format_correct": format_correct,
                "support_type": result.support_type,
                "mode": result.metadata.get("mode", mode),
                "answer_form": result.metadata.get("answer_form", "unknown"),
                "confidence": result.confidence,
                "latency_ms": elapsed,
                "render_latency_ms": result.metadata.get("render_latency_ms", 0.0),
                "answer": result.answer,
            })
            flag = "OK" if refusal_correct and source_correct and format_correct else "MISS"
            print(f"{question_id:02d} {flag}  {result.status:8s} source={result.source_id} lang={result.language} "
                  f"lang_ok={rows[-1]['language_correct']} fmt_ok={format_correct}  {row['question']}")

    answered = [row for row in rows if not row["expected_unknown"]]
    unknown = [row for row in rows if row["expected_unknown"]]
    report = {
        "total": len(rows),
        "answerable": len(answered),
        "unknown": len(unknown),
        "source_accuracy": sum(row["source_correct"] for row in answered) / max(1, len(answered)),
        "refusal_accuracy": sum(row["refusal_correct"] for row in rows) / max(1, len(rows)),
        "language_accuracy": sum(row["language_correct"] for row in rows) / max(1, len(rows)),
        "format_accuracy": sum(row["format_correct"] for row in rows) / max(1, len(rows)),
        # The two Kaggle failure shapes measured separately, because an answer
        # that recites the question and an answer that stops mid-word used to be
        # one indistinguishable "language failure" in the aggregate.
        "echo_free_accuracy": sum(row["echo_free"] for row in rows) / max(1, len(rows)),
        "complete_answer_accuracy": sum(row["complete"] for row in rows) / max(1, len(rows)),
        "language_accuracy_by_script": {
            script: round(
                sum(row["language_correct"] for row in rows if row["language"] == script)
                / max(1, sum(row["language"] == script for row in rows)), 3)
            for script in ("en", "bn", "banglish", "mixed")
            if any(row["language"] == script for row in rows)
        },
        "overall_gate_accuracy": sum(row["source_correct"] and row["refusal_correct"] for row in rows) / max(1, len(rows)),
        "retrieval_mode": mode,
        "generator_loaded": model_loaded,
        "median_latency_ms": sorted(row["latency_ms"] for row in rows)[len(rows) // 2],
        "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[max(0, int(len(rows) * 0.95) - 1)],
        "median_render_latency_ms": sorted(row["render_latency_ms"] for row in rows)[len(rows) // 2],
        "p95_render_latency_ms": sorted(row["render_latency_ms"] for row in rows)[max(0, int(len(rows) * 0.95) - 1)],
        "rows": rows,
    }
    output = ROOT / "reports" / "offline_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
