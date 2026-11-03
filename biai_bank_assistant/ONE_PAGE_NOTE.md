# NB Assist — one-page evaluation note

## What I built

NB Assist is an offline-first PySide6 assistant for Nirbhor Bank's FAQ. It retrieves only from the supplied FAQ, detects whether the customer wrote in English, Bangla or Banglish, validates that every number in the reply also appears in the source FAQ, and hands any question the FAQ cannot support to a human. Generation is backed by a small local open model (Qwen3-1.7B, 4-bit) when that stack is available; without it the same retrieval and safety gates run in a deterministic smoke-test mode that returns the verbatim FAQ text.

## How I tested it

I ran the supplied 40-question test set: 30 answerable from the FAQ and 10 deliberately uncovered. `evaluate.py` automates it and writes `reports/offline_results.json` with the per-question source FAQ ID, refusal status and refusal reason, detected language, answer, retrieval mode and latency. The 40 questions are held out: they never train or tune anything, and every answer is grounded in `bank-faq.csv` alone.

## Results

Two modes exist, so I report them separately rather than quoting whichever looks better.

| | Lexical smoke-test (no model, CPU) | Model stack (Kaggle T4) |
|---|---|---|
| Correct FAQ source, 30 answerable | **29/30** | `__/30` pending run |
| Correct hand-offs, 10 uncovered | **10/10** | `__/10` pending run |
| Overall groundedness gate | **39/40 (97.5%)** | `__/40` pending run |
| Median / p95 latency | 1.5 ms / 2.2 ms | `__ ms / __ ms` pending run |
| Reply in the customer's language | No — verbatim English FAQ text | expected: yes |

The smoke-test numbers are real but they are not the numbers to judge: they show retrieval and the safety gates working on a machine where the language model could not load at all. They also overstate how much of the refusal behaviour comes from model judgement, because some hand-offs are produced by the out-of-scope topic guards instead.

The one miss is test question 3 ("a transaction I did not make, where do I report it?"), which retrieves the dormant-account entry rather than the disputed-transaction entry. Both mention an unrecognised transaction, and I stopped short of hard-coding a fix for a single question.

**Still outstanding:** the brief asks for the same 40 questions run through an online AI model as a comparison. That baseline is not in this submission yet — `notebooks/kaggle_demo.ipynb` section 7 is the cell that produces it, and it needs a hosted-model key, so it runs once on Kaggle rather than here. The same notebook's section 6 is the brief's *optional* extra (further training); I have written it but not run it, and point 3 of the brief — running on a small open model — is what sections 3 to 5 do.

## Outside data used, and where it came from

All public and general-purpose; no real bank or customer data; none of it contributes facts to any answer. Full table with licences is in `README.md`. In short:

- `istiaqfuad/bangla-english-banglish-pairs` (CC-BY-4.0, derived from `Helsinki-NLP/opus-100`) — Bangla/Banglish↔English pairs.
- `ShayonSarker/Bengali-to-Banglish-Dataset` (MIT) — Banglish term lexicon.
- Mendeley Data `5h5vkhm9n5` (CC-BY-4.0) — standard Bangla → Banglish → English triples.
- Vashantor (MIT) — regional dialect robustness only.
- Inherited indirectly: the embedder `istiaqfuad/triBne-e5-small` (MIT) is a third-party fine-tune of `intfloat/multilingual-e5-small` trained on the CC-BY pairs above, so outside language data already reaches the system through the retrieval model.
- Rejected after inspection: a Bangla grammar QA set with no stated licence, and a religious hate-speech set built from scraped real user comments with author links — that one fails the no-real-data rule outright.

Models are open-licensed and small: Qwen3-1.7B (Apache-2.0, 2.03B parameters, ~1.2–1.5 GB at 4-bit) and the ~118M-parameter e5-small embedder (MIT).


## What does not work well

- **Language matching is the weakest part.** Without the model stack the assistant replies in English to a Bangla or Banglish question, which fails requirement 2 of the brief. The refusal messages are translated; the answers are not, except two canned FAQ 14 strings.
- **The vocabulary bridge is hand-written.** Roughly 200 generic Bangla/Banglish terms plus suffix stemming. Any word outside it silently degrades that query, so coverage is the main accuracy lever and the main source of unseen errors.
- **Two hand-off paths are rules, not judgement.** The out-of-scope topic guard and the unstated-figure guard are deterministic checks over the FAQ's own vocabulary. They are honest and auditable, but they are not the model reasoning.
- **Ranking is still wrong on question 3**, and a handful of near-ties are resolved only by the ambiguity margin.
- Regional dialects are not treated as separate banking policies, and the assistant refuses whenever evidence is weak, which is safer but costs recall on unusually phrased questions.
- No online-model baseline comparison has been run yet.

## What I would do next

Replace the hand-written bridge with a small Banglish→English query rewriter LoRA-trained on the disclosed CC-BY parallel pairs above — language data only, with the FAQ still the sole source of facts. Then fine-tune a small answerability/reranking head, add a human-reviewed answer cache for high-volume intents, and expose the same offline core as a secure service for branch and call-centre deployment.

