# NB Assist — one-page note

Task 2, BIAI AI Engineer Round 2. Commands, per-question results and the full
limit list are in `README.md` and `reports/`; this page is the summary.

## What I built

An offline-first PySide6 assistant for Nirbhor Bank's FAQ. It retrieves only from
the supplied `bank-faq.csv`, detects whether the customer wrote English, Bangla or
Banglish, refuses what the FAQ cannot support, and verifies every number in the
reply against the entry it came from before anything is served. English answers and
entailment checks use Qwen3-1.7B (4-bit); Bangla uses a dedicated English-to-Bangla
NLLB checkpoint; Banglish is deterministic romanisation of verified Bangla. Retrieval
is a multilingual e5 embedder plus a concept-based lexical scorer, because an
English/Bangla embedder has never seen Roman-script Bangla. `bank-faq.csv` and
`test-questions.csv` are byte-identical to the pack (checked by hash in the tests),
the FAQ is the only file opened for facts, and the 40 questions are held out — they
never train or tune anything.

## How I tested it

- **The supplied 40 questions** through `evaluate.py` on a Kaggle T4, writing
  `reports/offline_results.json`: per-question FAQ source, refusal status and reason,
  detected language, answer, `answer_form`, mode and latency.
- **Two modes reported separately**, not whichever looks better: the model stack
  (`retrieval_mode: dense`, `generator_loaded: true`) and a lexical smoke test with no
  model loaded.
- **12 regression tests** — `python3 -m unittest tests.test_regression -v` — run with
  no GPU and no weights: the source/hand-off table, the pack's file hashes, refusal
  strings in all three scripts, the fact and format gates, the Banglish bridge, the
  register pass, the remote payload round-trip.
- **Line-by-line reading of the report, not the aggregate.** A run once headlined
  `language accuracy 0.70` while all 40 rows had reached the right FAQ; the defect was
  reciting the customer's question and stopping mid-word. That is why echo and
  completeness are separate columns now.
- **The required online comparison**: notebook section 7 sends the same 40 questions
  plus the same retrieved FAQ entry and the same rules to `gemini-3.5-flash-lite`.
  Only synthetic questions and FAQ text leave the machine; the key is a Kaggle secret.
- Outside data is public language data only — pairs, intents and the e5 embedder —
  disclosed with licences in `README.md`. No real bank or customer data anywhere, and
  none of it contributes a fact to any answer.

## Results

| | Lexical smoke test (CPU, no model) | Model stack (Kaggle T4) |
|---|---|---|
| Correct FAQ source, 30 answerable | 30/30 | **30/30** |
| Correct hand-offs, 10 uncovered | 10/10 | **10/10** |
| Reply in the customer's script | **20/40** — with no model a Bangla question gets the bank's English sentence | **40/40** — en 1.00, Bangla 1.00, Banglish 1.00 |
| No recitation, no mid-word stop | 40/40 | 40/40 |
| Median / p95 latency | not measured as a result — no model, CPU only | **15.4 ms / 3301 ms** (the tail is English Qwen rewrites; the Bangla and Banglish render step is 0.06 ms median, 0.33 ms p95) |

Judge the right-hand column; the smoke test only shows the gates working on a machine
where the model could not load. **Against the hosted model, given the same entry:** it
answered 30/30, handed off 10/10 and invented no figure — groundedness was never its
weakness. Its replies were in the customer's own script for English 10/10, Bangla 7/9
and **Banglish 0/11**; 2 of 30 came back as fragments and 10 dropped a figure the
entry states. Both systems are right about the money; the difference is whether a
Bangla-speaking customer can use the sentence. Further training (section 6,
`train.py`) is built and disclosed, and I will not claim it was needed: the failure
was in the pipeline, not the weights.

## What does not work well

- **The last mile on these entries is pinned, not generated.** All 20 Bangla and
  Banglish replies come from source-derived renderings in `src/bn_render.py`, chosen
  before the translator runs; 6 English replies are Qwen's and 4 fell back to the
  bank's own sentence. The `answer_form` column shows it per row. My wording, no
  added facts — but it is a pin, and a question outside these entries still depends on
  the translator.
- **No Bangla reply at all without the model** — 20/40, reported as 20/40.
- **Wording cannot be verified across scripts.** Numbers, acronyms, concepts, script
  and completeness are checked, but one garbled yet complete Bangla sentence can still
  pass, because the FAQ holds no Bangla to compare phrasing against.
- **The Banglish vocabulary bridge is hand-written** (~200 terms), so a question built
  from words outside it is still a coin flip. That coverage is the main accuracy lever
  and the main source of unseen errors.
- **Two refusal paths are deterministic rules**, not model judgement — honest and
  auditable, but not reasoning.
- **The register is code, not the model:** Bengali digits, `টাকা` after the amount,
  no dangling hyphens and the English service words a customer uses are applied by one
  style pass after the translator.
- **The tunnel is a demo path, not hosting.** Section 8.1's ngrok address is permanent,
  but Kaggle has no always-on container: when the session stops the app says so and
  answers from the FAQ on the machine — which is why there is no public web link to
  send, only this recording and the repo.

## What I would do next

Replace the hand-written bridge with a small Banglish→English query rewriter
LoRA-trained on the disclosed CC-BY parallel pairs — language into the weights, the
FAQ still the only source of facts. Then an answerability/reranking head, a
human-reviewed cache for high-volume intents, and `api_server.py` deployed properly
instead of tunnelled.
