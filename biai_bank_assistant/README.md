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

1. **Start the model.** Open `notebooks/kaggle.ipynb` on Kaggle with the [`biai-bank-assistant`](https://www.kaggle.com/datasets/mdabukawser/biai-bank-assistant) dataset attached (how to attach it is under **The Kaggle dataset** below) and the accelerator set to **GPU T4** (internet on). Run sections 1–5; section 5 prints the 40-question table.
2. **Publish it — ngrok, set up once.** Sign up free at **<https://dashboard.ngrok.com>** (Google, GitHub or a work email; no card). Then in the dashboard's left menu:
   - **Your Authtoken** → copy the long string on that page. That is the value for `NGROK_AUTHTOKEN`.
   - **Domains** → the free plan reserves **one domain to your account**, shown there as `https://xxxxxxxx-ngrok-free.app`. **That page is where the link comes from.** The `xxxxxxxx` half is generated by ngrok — it is *not* your username and it is not a placeholder you invent; click **Edit** on the domain if you want a readable prefix such as `mdabukawser-ngrok-free.app`. Whatever it says there is yours in every future session, notebook and machine.

   Add two secrets in the Kaggle notebook: right-hand panel → **Add-ons and Secrets → Secrets → New Secret**, exactly these names.

   | Secret name | Value | Printed by the notebook? |
   |---|---|---|
   | `NGROK_AUTHTOKEN` | the authtoken from **Your Authtoken** | never |
   | `BIAI_API_TOKEN` | a string you invent **once**: `python3 -c "import secrets;print(secrets.token_urlsafe(18))"` | never — but you type it on the laptop, so keep it written somewhere private |

   Nothing is pasted into a cell, the dataset or git. A secret is injected at run time, and notebook output is shared with the notebook, which is exactly why the token is not echoed.

3. **Run section 8.1** (section 5 first, so the models are loaded). It installs FastAPI and pyngrok, starts `api_server.py` on the GPU, answers one Banglish question through the local endpoint so you can see the served text, opens the tunnel on your reserved domain, checks `/health` through the public URL and only then prints:

   ```text
   BIAI_API_URL="https://xxxxxxxx-ngrok-free.app"
   BIAI_API_TOKEN="<the value you put in the BIAI_API_TOKEN secret>"
   ```

   Copy both **exactly**. `api_server.py` compares the `Authorization` header against the token it loaded from that secret, so any other value — including one from a token-generator website — is a `401`. Because the domain is reserved and the token is yours, these two values are now permanent: the same next week, in a different notebook, on a different laptop.

   **No ngrok account?** **Section 8** does the same thing through a Cloudflare quick tunnel: no account, no domain, no setup — and a random `https://<words>.trycloudflare.com` URL plus a token the cell generated, both dead when the session stops, so the laptop command must be retyped every single run. That is the only reason Cloudflare is still in the notebook: it is the zero-setup fallback, not the way to publish this.

   Section 8 prints its URL **only after `/health` answers through it**. It used to take the first `https://` it found in cloudflared's log, and cloudflared prints `https://www.cloudflare.com/website-terms/` in its own diagnostics — that is a Cloudflare web page, it answers a POST with **403 Forbidden**, and the desktop app then says the endpoint did not answer. The cell now matches only `*.trycloudflare.com` hosts and polls the edge for up to a minute before printing anything.

   **When the window says the endpoint did not answer**, the inspector now prints the real reason. Three causes, in the order they actually happen:

   | What it says | What it means | Fix |
   |---|---|---|
   | `endpoint replied 403` on an ngrok URL | the hostname is not bound to the tunnel your `NGROK_AUTHTOKEN` opened — a second ngrok account owns that domain, or an older session still holds it | check the **Domains** page belongs to the same account as the authtoken, then re-run section 8.1 |
   | `endpoint replied 401` | wrong `BIAI_API_TOKEN` on the laptop | copy the value you put in the secret, exactly |
   | `URLError` / `RemoteDisconnected`, pill `LOCAL FAQ` | the notebook session is stopped, so nothing is listening | `Run All`, or section 8.1 alone; the app keeps answering from the FAQ meanwhile |

4. **Run the desktop app against it.** Three commands on every machine; only how you set the two variables differs. Use `python3` (Linux/macOS) or `py -3` (Windows) — never a bare `python`, which on some machines is still Python 2.

   **Linux (bash/zsh)**

   ```bash
   git clone https://github.com/Mayamoho/orthomnorthom.git
   cd orthomnorthom/biai_bank_assistant
   python3 -m pip install -r requirements.txt          # PySide6 only
   BIAI_API_URL="https://xxxxxxxx-ngrok-free.app" BIAI_API_TOKEN="<your BIAI_API_TOKEN>" python3 main.py
   ```

   **macOS (Terminal)**

   ```bash
   git clone https://github.com/Mayamoho/orthomnorthom.git
   cd orthomnorthom/biai_bank_assistant
   python3 -m pip install -r requirements.txt
   export BIAI_API_URL="https://xxxxxxxx-ngrok-free.app"
   export BIAI_API_TOKEN="<your BIAI_API_TOKEN>"
   python3 main.py
   ```

   **Windows (PowerShell)**

   ```powershell
   git clone https://github.com/Mayamoho/orthomnorthom.git
   cd orthomnorthom/biai_bank_assistant
   py -3 -m pip install -r requirements.txt            # PySide6 only
   $env:BIAI_API_URL   = "https://xxxxxxxx-ngrok-free.app"
   $env:BIAI_API_TOKEN = "<your BIAI_API_TOKEN>"
   py -3 main.py
   ```

   On Windows Git Bash use the Linux form with `python3`. Type these once: after the first run the values do not change, so a stopped notebook is fixed by `Run All`, not by a new command.

5. **Test it — the four checks.**

   | Type this | Expect |
   |---|---|
   | `Card er PIN bhule gechi, reset korbo kibhabe?` | the same Banglish text section 8 printed |
   | `আমার কার্ড হারিয়ে গেছে, এখন কী করব?` | Bangla script, Bangla digits, `৩০০ টাকা` |
   | `What is today's dollar to taka exchange rate?` | `I don't know, let me connect you to a person.` |
   | status pill / inspector | `MODEL READY — KAGGLE`, retrieval `dense`, Qwen generator named |

   A `LOCAL FAQ` pill means the token was wrong or the tunnel died: the window says so and keeps answering from the FAQ on the machine, so the demo never becomes a blank box. Re-run section 8.1; with the ngrok values above the command itself never needs changing.

### Which URL is which, and what "survives" honestly means

| URL | Where it comes from | Use it for |
|---|---|---|
| `http://127.0.0.1:8000` | uvicorn inside the notebook | never typed; it is what the tunnel points at |
| `https://<random>.trycloudflare.com` | **section 8**, printed at the end of the cell, new every run | a one-off demo when you have no ngrok account |
| `https://xxxxxxxx-ngrok-free.app` | your ngrok account's **Domains** page, reprinted by **section 8.1** | the link you keep, hand to a judge, and type once |

`xxxxxxxx` is whatever the Domains page shows after you click **Edit** on the free reserved domain. It is not your ngrok username and not a token; it is a hostname you choose (or accept) once, and ngrok routes every future tunnel of that account to it.

What "survives" honestly means: the **address** is permanent because it belongs to the ngrok account; the **server** is not, because Kaggle has no always-on container and a session ends when you stop it or after roughly 9 hours of GPU time. While the notebook is off, the app falls back to the FAQ on the laptop and the pill reads `LOCAL FAQ` — the designed behaviour, not an error. A 24/7 endpoint needs a real host (or your own machine running `python3 api_server.py`), not a tunnel. Free ngrok domains show an interstitial page to browsers, so `src/remote.py` sends `ngrok-skip-browser-warning: 1` on both of its calls and the app reads JSON, not that page.

## The Kaggle dataset

**<https://www.kaggle.com/datasets/mdabukawser/biai-bank-assistant>** is the input the notebook reads. Its files are uploaded **flat** — one level, no folders — exactly as the dataset page lists them:

| Uploaded file | What it is |
|---|---|
| `bank-faq.csv` | the only source of facts: 30 entries, byte-identical to the pack |
| `test-questions.csv` | the held-out 40 questions, evaluation only, never trained on |
| `__init__.py`, `assistant.py`, `retrieval.py`, `normalize.py`, `generator.py`, `models.py`, `data.py`, `remote.py` | the pipeline. `__init__.py` installs the Bangla/Banglish renderer hook; `remote.py` is the client the desktop app uses against a GPU endpoint |
| `bn_render.py` | the English→Bangla translator, the per-sentence verifier, the deterministic Banglish transliterator and the register pass |
| `main.py`, `main_window.py`, `theme.py` | the PySide6 desktop app |
| `api_server.py` | the FastAPI endpoint that section 8 serves |
| `evaluate.py`, `kaggle_demo.py` | the 40-question run, and the headless demo |
| `train.py`, `vashantor_train.jsonl`, `vashantor_train.metadata.json` | the optional adapter path; the two Vashantor files are train-only public language data (CC-BY-4.0) |
| `requirements-kaggle.txt` | what section 1 installs, and the marker the notebook uses to locate the dataset |
| `README.md`, `ONE_PAGE_NOTE.md` | this documentation |

Flat is deliberate. `/kaggle/input` is read-only, so section 1 rebuilds the layout the code expects — `src/`, `data/`, `ui/`, `training_data/`, `reports/` — under `/kaggle/working/nb_assist`, and asserts each piece exists. A missing or stale file therefore stops in that cell instead of scoring quietly worse.

### Use the dataset in a notebook

1. On Kaggle: **Create → Notebook**.
2. Right-hand panel: **Input → + Add Input → Datasets**, search `biai-bank-assistant` (owner `mdabukawser`) → **Add**. The mount path becomes `/kaggle/input/biai-bank-assistant`.
3. **Settings → Accelerator → GPU T4**, and **Internet → On** — the weights come from Hugging Face on the first run. If you intend to run section 7, add the secret `GEMINI_API_KEY` as in the next block.
4. Clone the repo and copy `notebooks/kaggle.ipynb` into that notebook — **File → New Notebook → Import notebook from local file** does it in one step:

   ```bash
   git clone https://github.com/Mayamoho/orthomnorthom.git
   ```

   Section 1 finds the dataset by searching `/kaggle/input` for `requirements-kaggle.txt`, so no path is hardcoded; set `BIAI_KAGGLE_DATASET` only if you rename the dataset.
5. **Run All.** Section 1 prints `input files: [...]`; sections 2–3 assert the staged code, the renderer hook and the verification gates; section 5 prints the 40-question table that `kaggle8.json` records; section 8.1 prints the permanent ngrok URL for the desktop app (section 8 is the no-account fallback).

### The Gemini API key, for section 7

Section 7 is the brief's required comparison against an online model. The key is **never** in this repo, in the dataset or in the notebook — it lives in Kaggle's secret store and the cell reads it at run time with `UserSecretsClient`.

1. Get a free key at **<https://aistudio.google.com/apikey>** (any Google account; the free tier covers the 40 calls easily).
2. In the open notebook, use the right-hand panel: **Add-ons and Secrets** (older Kaggle UI: **Settings → Add-ons and Secrets**).
3. Under **Secrets**, click **New Secret**, then
   - **Name:** `GEMINI_API_KEY` — exactly this, uppercase, no spaces;
   - **Value:** paste the key you copied;
   - click **Create**.
4. Turn **Internet** on in the same panel and run the section 7 cells. The first one prints `baseline model: gemini-3.5-flash-lite | key found: True`. If it prints `key found: False`, the name is misspelled or the secret was saved in a different Kaggle account; the section then prints the manual fallback instead of failing the run.
5. To compare a different hosted model, add **Environment → Add Variable** `BIAI_BASELINE_MODEL` on the same panel — `gemini-2.0-flash` is shut down and the 2.5 family is retired, which is why the default is `gemini-3.5-flash-lite`.

Nothing from that call is used to build an answer, and no customer data is ever sent. What goes over the wire is the 40 synthetic questions from `test-questions.csv` plus, in the default mode, the FAQ entry the offline retriever already matched for that question — `bank-faq.csv` text, not customer text.

**What the hosted model is given.** `BIAI_BASELINE_MODE` decides it, and the default is `faq`: the same retrieved `bank-faq.csv` entry and the same instruction the local generator gets — answer only from the entry, never add a number the entry does not state, reply `HANDOFF` when the entry does not answer. That makes the two columns differ by model, not by evidence. `no_faq` asks the hosted model as a generic assistant with no bank data; the first run used that and all 17 answers it obtained came back as the literal `HANDOFF`, including the 15 the FAQ covers — a fair result for a model with nothing to read, but not a comparison. Section 7 prints the mode in its header and only scores rows of the current mode, so a report cannot mix the two.

**The free-tier limit is 15 requests per minute per model**, so an unpaced loop over 40 questions returns 15 answers and 25 `429 RESOURCE_EXHAUSTED` errors — that is what the first run of `baseline_results.csv` did. The section 7 cell now spaces calls 4.5 s apart, sleeps the `retryDelay` the API reports, and keeps whatever is already in `reports/baseline_results.csv` for the same mode, so re-running it fetches only the failed rows. Everything stays free: same free key, same free model, roughly three minutes. The comparison cell drops `ERROR` rows from the scoring, prints all 40 questions side by side (offline answer vs the hosted model's reply) and saves the table to `reports/baseline_comparison.md`.

## Quick start — this machine alone

**Linux (bash/zsh)**

```bash
git clone https://github.com/Mayamoho/orthomnorthom.git
cd orthomnorthom/biai_bank_assistant
python3 -m pip install -r requirements.txt      # PySide6 only, for the UI
python3 main.py
```

**macOS (Terminal)**

```bash
git clone https://github.com/Mayamoho/orthomnorthom.git
cd orthomnorthom/biai_bank_assistant
python3 -m pip install -r requirements.txt
python3 main.py
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Mayamoho/orthomnorthom.git
cd orthomnorthom/biai_bank_assistant
py -3 -m pip install -r requirements.txt
py -3 main.py
```

On Windows Git Bash use the Linux form with `python3`. Never a bare `python`, which on some machines is still Python 2.

### What the local run is, and the file it is linked to

Running it here is the same code path that wrote **[`reports/offline_results.json`](reports/offline_results.json)** —
that file is the record of the 40-question run and the source of every number in this README and in
`ONE_PAGE_NOTE.md`. It is in the repo so the claims can be checked line by line without a GPU. Each of
its 40 rows carries `id`, `question`, `status`, `source_id`, `language`, `answer`, `support_type`,
`answer_form` (`model`, `source_style_fallback` or `faq_fallback`), `render_latency_ms` and `latency_ms`,
and the header block says what the run was: `retrieval_mode: "dense"`, `generator_loaded: true`,
`source_accuracy`/`refusal_accuracy`/`language_accuracy`/`overall_gate_accuracy` all `1.0`, median latency
15.3 ms, p95 3172 ms.

Two commands regenerate it, and they are the same two commands the notebook runs:

```bash
python3 -m pip install -r requirements-kaggle.txt   # torch + transformers for the model stack
python3 evaluate.py                                 # writes reports/offline_results.json
```

Without `torch` installed, `python3 evaluate.py` still writes the file, but with
`retrieval_mode: "lexical"` and `generator_loaded: false` — that is the smoke-test column, not this one.
The committed `reports/offline_results.json` is the model-stack run, and the app pointed at the Kaggle
endpoint (next section) answers from the same stack, so what the desktop window shows is the same text
that file records, row for row.

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

**Regression tests** — `python3 -m unittest tests.test_regression -v`, 12 tests, no GPU and no weights required:

- `test_source_and_refusal_regression` — all 40 held-out questions: the answered/hand-off decision, and the expected `source_id` for every answer. This is the test that catches a retrieval regression.
- `test_supplied_files_are_untouched` — SHA-256 of `data/bank-faq.csv` and `data/test-questions.csv` against the pack's digests, plus row counts (30 entries, 40 questions). Editing the factual source fails the suite.
- `test_language_and_refusal_contract`, `test_grounding_and_format_gates` — script detection, the refusal lines in all three scripts, `preserves_key_facts` on a fabricated figure, and the format gates on a real Bangla reply.
- `test_generator_prompt_has_script_examples` — the style examples are actually inside the Bangla and Banglish prompts.
- `test_source_render_fallbacks_cover_kaggle6_misses` — pinned renderings exist for the entries that once came back in English, each ends in sentence punctuation, and no Bangla script leaks into a Banglish line.
- `test_banglish_keeps_requested_english_banking_terms`, `test_banglish_phrase_layer_uses_customer_service_register` — `interest rate`, `reducing balance`, `processing fee`, `working days-er moddhe`, `cheque book-er jonno` survive romanisation.
- `test_style_answer_applies_customer_register` — Bangla gets Bengali digits and `টাকা`, `এন আই ডি` and `আপডেইট`; Banglish keeps Latin digits, drops `BDT` and hyphens, keeps `minimum`.
- `test_remote_payload_round_trip` — a served JSON payload rebuilds into the same `AssistantResult` the local pipeline produces.
- `test_remote_headers_skip_the_ngrok_interstitial` — the endpoint calls carry `ngrok-skip-browser-warning`, so a free ngrok domain answers JSON instead of its HTML warning page, and the bearer token is only sent when one exists.
- `test_training_script_does_not_reference_held_out_questions` — static check that `train.py` never opens `test-questions.csv` and names only the disclosed public corpora.

**Evaluation run** — `python3 evaluate.py` answers all 40 questions end to end and writes [`reports/offline_results.json`](reports/offline_results.json), the file committed in this repo and the one every number in this README and in `ONE_PAGE_NOTE.md` is read from. I read that file line by line rather than trusting the aggregate: 30/30 on the FAQ source, 10/10 on the hand-offs, language and latency per script, and for each row the `support_type`, the `answer_form` (model or fallback) and `rendered_by`. The English-leak cases in the run before this one were found exactly this way — a 27/30 headline still hides a sentence-level bug.

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
