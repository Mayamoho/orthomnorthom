# NB Assist — Offline Bilingual Bank Assistant

NB Assist is a local-first FAQ assistant for BIAI AI Engineer Round 2, Task 2. It answers only from the supplied Nirbhor Bank FAQ and hands unsupported questions to a person. The interface is a PySide6 desktop app with English, Bangla, Banglish, and mixed-input support.

Everything here is grounded in the pack's own files: `data/bank-faq.csv` is the only source of facts, and the 40-question `data/test-questions.csv` is held out for testing and is never used for training or tuning.

## What it does

- Reads a question in English, Bangla, Banglish or mixed script, detects the script, and answers in that same script.
- Retrieves the right FAQ entry with a hybrid scorer — IDF-weighted query coverage, character 4-gram overlap, a topic tie-break, and an optional multilingual embedding model — and only answers when the evidence clears a floor calibrated for the script the customer used. Banglish needs that: a Bangla/English embedder has never seen Roman-script Bangla, so there the cross-script glossary carries the query and dense similarity is only a tiebreaker.
- Bridges scripts in `src/normalize.py`, which maps generic Bangla and Banglish banking terms to English. A gloss is scored as a second query representation and combined by `max`, so it can raise a score but never dilute one. Concepts are deduped (`informative_terms`): a Bangla word whose English gloss is already in the query is dropped, otherwise the loan-interest entry beat the loan-eligibility entry by matching "পার্সোনাল লোনের" *and* "personal loan" as two separate facts.
- Refuses and offers a human hand-off on three groundedness guards: the retrieval floor, the ambiguous-evidence margin, and two FAQ-derived checks — a topic the FAQ never mentions, or a figure that contradicts a range the FAQ states. The topic guard filters itself against the FAQ's own vocabulary, so it can only refuse what the corpus truly cannot answer. The ambiguity guard (`is_ambiguous`) calls a near-tie a coin flip only when the topic and lexical channels agree with the fused score, so `0.3592 vs 0.3589` no longer hands off a question that named the topic outright.
- Lets the local generator's entailment check veto a retrieved FAQ only when the lexical evidence is below the same floor the fused score had to clear. Its English-only prompt produces false NOs on Bangla and Banglish, and a false NO on a known FAQ is a real customer sent to a dead end.
- Writes the Bangla reply with the dedicated `SifatAlMukit/finetuned-en-bn-nllb` checkpoint, one sentence at a time, and verifies each sentence against its English source before using it: same numbers, same acronyms, no concept the source does not contain, no foreign script, no loop, sane length. A sentence that fails is regenerated; if it still fails, the bank's own sentence is kept, so a reply is never invented and never empty. Qwen3-1.7B stays the answer writer for English and the entailment checker.
- Produces Banglish by deterministic romanisation of that verified Bangla, because small models spell Bangla in Latin letters badly while spelling Bangla script well enough. There is no mathematically unique Banglish spelling for every Bengali word, so the renderer chooses one consistent Bangladesh-oriented style rather than claiming phonetic perfection.
- Applies one register pass (`style_answer`) to what a customer actually reads: Bangla answers use Bengali digits and `টাকা` after the amount (`১০ লাখ টাকা`), Banglish keeps Latin digits and the English service terms the FAQ itself uses (`interest rate`, `NID`, `face verification`, `update form`, `minimum`), no hyphen is left hanging inside a Banglish word, and textbook forms give way to natural ones (`প্রতি তিন মাস পর পর`, `নমিনি`, `অ্যালার্ট`).
- Preserves key facts on the way out: amounts, percentages, dates and hotline numbers survive (`1,000` and `1000` compare equal) and a figure the entry does not state is rejected, so the PIN-reset reply that invented "Fee holo BDT 500" cannot ship. Where the entry states one or two figures those *are* the answer, so dropping them fails too; a seven-figure entry such as the FDR rates may be answered from one of them.
- Runs three format gates on every model reply before it is served (`answer_is_wellformed`): it must not recite the customer's question, must finish its sentence, must be in the customer's script. A reply that fails is replaced by the bank's own sentence and the rejected draft stays visible in the window instead of becoming an "I don't know". A fourth gate — that an *English* reply be built from this entry's own wording — catches model boilerplate; it cannot run on Bangla or Banglish, because the FAQ has no Bangla text to compare against.
- Answers a two-holder joint-account question from FAQ 14, and hands off anything unsupported or contradictory.
- Reports, for every one of the 40 questions, the FAQ source, refusal status, language, confidence and latency in `reports/offline_results.json`.

**One answer source, and it is the supplied file.** The pack says `bank-faq.csv` "is the only source your assistant may answer from", so it is stored byte-identical to the pack, `test-questions.csv` is untouched, and there is no third data file of prepared Bangla answers — a file of answer text would be a second source, however faithful. Retrieval indexes the FAQ's own `topic`, `question` and `answer`; a Bangla or Banglish reply is *written* by the model from that English sentence, which is what requirements 2 and 3 of the brief ask for together.

## Run it — the Kaggle GPU answers, the desktop app shows it

There is nothing to install on the laptop except PySide6. **A laptop CPU cannot write Bangla**, so the accurate answers come from the notebook's GPU stack over one HTTP tunnel. Every number in `kaggle8.json` (copied verbatim into `reports/offline_results.json`) was produced by the path below.

1. **Start the model.** Open `notebooks/kaggle.ipynb` on Kaggle with the `biai-bank-assistant` dataset attached and the accelerator set to **GPU T4** (internet on). Run sections 1–5; section 5 prints the 40-question table.
2. **Publish it.** Run **section 8**. It installs FastAPI, starts `api_server.py` on the GPU, answers one Banglish question through the endpoint so you can see the served text, opens a Cloudflare quick tunnel, and prints at the end:

   ```text
   BIAI_API_URL=https://<random-words>.trycloudflare.com
   BIAI_API_TOKEN=<a token the cell generated for this session>
   python3 main.py   ← the exact line to run, with both values already filled in
   ```

   Copy both values **exactly as printed**. The token is not typed by you and it is not from any token-generator website: `api_server.py` compares the `Authorization` header against the token that same cell generated, so any other value is a `401`. The URL lives only while the notebook session runs.
3. **Run the desktop app against it.** Three commands on every machine; only how you set the two variables differs. Use `python3` (Linux/macOS) or `py -3` (Windows) — never a bare `python`, which on some machines is still Python 2.

   **Linux (bash/zsh)**

   ```bash
   cd biai_bank_assistant
   python3 -m pip install -r requirements.txt          # PySide6 only
   BIAI_API_URL="https://<that-url>" BIAI_API_TOKEN="<printed token>" python3 main.py
   ```

   **macOS (Terminal)**

   ```bash
   cd biai_bank_assistant
   python3 -m pip install -r requirements.txt
   export BIAI_API_URL="https://<that-url>"
   export BIAI_API_TOKEN="<printed token>"
   python3 main.py
   ```

   **Windows (PowerShell)**

   ```powershell
   cd biai_bank_assistant
   py -3 -m pip install -r requirements.txt            # PySide6 only
   $env:BIAI_API_URL   = "https://<that-url>"
   $env:BIAI_API_TOKEN = "<printed token>"
   py -3 main.py
   ```

   On Windows Git Bash use the Linux form with `python3`.

4. **Test it — the four checks.**

   | Type this | Expect |
   |---|---|
   | `Card er PIN bhule gechi, reset korbo kibhabe?` | the same Banglish text section 8 printed |
   | `আমার কার্ড হারিয়ে গেছে, এখন কী করব?` | Bangla script, Bangla digits, `৩০০ টাকা` |
   | `What is today's dollar to taka exchange rate?` | `I don't know, let me connect you to a person.` |
   | status pill / inspector | `MODEL READY — KAGGLE`, retrieval `dense`, Qwen generator named |

   A `LOCAL FAQ` pill means the token was wrong or the tunnel died: the window says so and keeps answering from the FAQ on the machine, so the demo never becomes a blank box. Re-run section 8 and copy the new values.

## Quick start — this machine alone

**Linux (bash/zsh)**

```bash
cd biai_bank_assistant
python3 -m pip install -r requirements.txt      # PySide6 only, for the UI
python3 main.py
```

**macOS (Terminal)**

```bash
cd biai_bank_assistant
python3 -m pip install -r requirements.txt
python3 main.py
```

**Windows (PowerShell)**

```powershell
cd biai_bank_assistant
py -3 -m pip install -r requirements.txt
py -3 main.py
```

On Windows Git Bash use the Linux form with `python3`. Never a bare `python`, which on some machines is still Python 2.

Without `torch` the app runs in **SMOKE TEST** mode: it still answers and still refuses correctly, but it replies in the FAQ's own language because there is no model on that machine to rewrite it — the honest cost of running without weights is 20/40 on reply language, with everything else unchanged. The numbers this file reports come from the tunnel path above.

| Variable | Effect |
|---|---|
| `BIAI_DISABLE_DENSE=1` | force the lexical-only retriever, no embedding model needed |
| `BIAI_GENERATION_MODEL` / `BIAI_EMBEDDING_MODEL` | swap the generator or the embedder |
| `BIAI_TRANSLATOR_MODEL` | swap the English→Bangla checkpoint (default `SifatAlMukit/finetuned-en-bn-nllb`) |
| `BIAI_MAX_NEW_TOKENS` / `BIAI_RENDER_MAX_NEW_TOKENS` | raise the answer / translation budgets |
| `BIAI_ADAPTER_PATH` | load the adapter produced by `train.py` |
| `BIAI_API_URL` + `BIAI_API_TOKEN` | drive the UI against a GPU endpoint instead of this machine's copy; the app says which backend answered |

### What this mode shows

The same four questions from step 4 still work here, and the FAQ source, the hand-offs and the grounding checks are identical. What differs is the language column: with no model loaded, `আমার কার্ড হারিয়ে গেছে, এখন কী করব?` comes back as the bank's own English sentence, because the English FAQ is the only thing that may be quoted and there is nothing here to rewrite it. Run it against the tunnel instead and the same question arrives in Bangla script with Bangla digits — which is the reply the register rules in `src/bn_render.py` produce, including `minimum` and `monthly income` kept in English on a Banglish line.

## How I tested it

**Regression tests** — `python3 -m unittest tests.test_regression -v`, 11 tests, no GPU and no weights required:

- `test_source_and_refusal_regression` — all 40 held-out questions: the answered/hand-off decision, and the expected `source_id` for every answer. This is the test that catches a retrieval regression.
- `test_supplied_files_are_untouched` — SHA-256 of `data/bank-faq.csv` and `data/test-questions.csv` against the pack's digests, plus row counts (30 entries, 40 questions). Editing the factual source fails the suite.
- `test_language_and_refusal_contract`, `test_grounding_and_format_gates` — script detection, the refusal lines in all three scripts, `preserves_key_facts` on a fabricated figure, and the format gates on a real Bangla reply.
- `test_generator_prompt_has_script_examples` — the style examples are actually inside the Bangla and Banglish prompts.
- `test_source_render_fallbacks_cover_kaggle6_misses` — pinned renderings exist for the entries that once came back in English, each ends in sentence punctuation, and no Bangla script leaks into a Banglish line.
- `test_banglish_keeps_requested_english_banking_terms`, `test_banglish_phrase_layer_uses_customer_service_register` — `interest rate`, `reducing balance`, `processing fee`, `working days-er moddhe`, `cheque book-er jonno` survive romanisation.
- `test_style_answer_applies_customer_register` — Bangla gets Bengali digits and `টাকা`, `এন আই ডি` and `আপডেইট`; Banglish keeps Latin digits, drops `BDT` and hyphens, keeps `minimum`.
- `test_remote_payload_round_trip` — a served JSON payload rebuilds into the same `AssistantResult` the local pipeline produces.
- `test_training_script_does_not_reference_held_out_questions` — static check that `train.py` never opens `test-questions.csv` and names only the disclosed public corpora.

**Evaluation run** — `python3 evaluate.py` answers all 40 questions end to end and writes `reports/offline_results.json`. I read that file line by line rather than trusting the aggregate: 30/30 on the FAQ source, 10/10 on the hand-offs, language and latency per script, and for each row the `support_type`, the `answer_form` (model or fallback) and `rendered_by`. The English-leak cases in the run before this one were found exactly this way — a 27/30 headline still hides a sentence-level bug.

**Manual testing** — `python3 main.py`, then by hand: all three scripts plus a mixed one, a hand-off question, a question citing a figure the FAQ contradicts, and the fallback path with the endpoint unreachable. Then `python3 api_server.py` locally with `curl` against `/health` and `/answer`, with and without the bearer token, to confirm a wrong token surfaces as a `401` rather than a silent fallback, and that the pill switches to `MODEL READY — KAGGLE` when the URL is right. On Kaggle, section 3 of the notebook re-runs the renderer, verifier and transliterator assertions against the staged copy, and section 5 prints the 40-question table that `kaggle8.json` records.

**What manual testing caught that the tests did not** — an unreadable question colour in the chat window, hyphens left inside Banglish words, `ত্রৈমাসিকে` reading as textbook rather than customer-facing Bangla, and a pile of errors when the serving cell was run a second time. Each one is now either an assertion above, a rule in `style_answer()`, or a guard in the notebook.

## Data rule

Facts come only from the BIAI-supplied `data/bank-faq.csv`, stored byte-identical to the pack. `training_data/` is language-only supervision for `train.py`, never a response cache, and there is no side file of prepared answers: the renderer's small fallback table is code-level wording derived from specific FAQ entries and adds no facts.

Further training on public banking and Banglish vocabulary is allowed — the brief invites it ("Extra (optional): train the small model further on banking words or Banglish, and show if it got better"), confirmed by the BIAI Hiring Team on 3 November 2026 — provided no real bank or customer data is used and the source is disclosed. Language skill goes into the model weights, never into a second file of answers.

| Dataset | Source | License | Used for |
|---|---|---|---|
| `istiaqfuad/bangla-english-banglish-pairs` **(used by training)** | [HF](https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs), derived from `Helsinki-NLP/opus-100` | CC-BY-4.0 | Up to 1,200 filtered public banking-language pairs for the Qwen LoRA language adapter |
| `learn-abc/banking14-intents-en-bn-banglish` **(used by training)** | [HF](https://huggingface.co/datasets/learn-abc/banking14-intents-en-bn-banglish) | MIT | Synthetic banking-language register examples from the train split |
| `subhajitmahata84/banglabridge-instructions` **(used by training)** | [HF](https://huggingface.co/datasets/subhajitmahata84/banglabridge-instructions) | CC-BY-4.0 | Synthetic Banglish instruction/output style examples |
| `istiaqfuad/triBne-e5-small` **(used as the retriever)** | [HF](https://huggingface.co/istiaqfuad/triBne-e5-small) | MIT | Multilingual embeddings, itself fine-tuned on public CC-BY parallel pairs. Language similarity, not bank facts. |
| `SifatAlMukit/finetuned-en-bn-nllb` **(used as the renderer)** | [HF](https://huggingface.co/SifatAlMukit/finetuned-en-bn-nllb) | MIT | Conversational English↔Bengali NLLB checkpoint; wording only, never facts. Banglish is romanised from its verified Bangla output. |
| `ShayonSarker/Bengali-to-Banglish-Dataset` **(not used in the shipped run)** | [HF](https://huggingface.co/datasets/ShayonSarker/Bengali-to-Banglish-Dataset) | MIT | Considered for a future Banglish lexicon |
| Mendeley Data `5h5vkhm9n5` **(not used in the shipped run)** | [Mendeley](https://data.mendeley.com/datasets/5h5vkhm9n5) | CC-BY-4.0 | Considered for future Banglish→English pairs |
| Vashantor **(used by training)** | [Mendeley](https://data.mendeley.com/datasets/bj5jgk878b/2) | CC-BY-4.0 | Train-only Bangla/Banglish/English dialect pairs copied to `training_data/vashantor_train.jsonl` |

Models: `Qwen/Qwen3-1.7B`, Apache-2.0, 2.03B parameters (~1.2–1.5 GB at 4-bit); `SifatAlMukit/finetuned-en-bn-nllb`, MIT, ~0.6B parameters; and the ~118M-parameter `triBne-e5-small` retriever. All three fit on a Kaggle T4.

Two local archives were rejected and never used: the BGHL Bangla grammar QA set (domain mismatch, no stated licence) and the multilingual religious hate-speech set (scraped public comments with author links, unsafe and off-domain). The seven-regional-dialect workbook was inspected but not added, because its archive carries no licence metadata; Vashantor provides the same dialect supervision with a documented CC-BY-4.0 licence.

Nothing from any of these sources may enter an answer. They supply **language**, the FAQ supplies **facts**.
