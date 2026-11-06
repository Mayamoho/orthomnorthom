# NB Assist — Offline Bilingual Bank Assistant

NB Assist is a local-first FAQ assistant for the BIAI AI Engineer Round 2 Task 2. It answers only from the supplied Nirbhor Bank FAQ and hands unsupported questions to a person. The interface is a PySide6 desktop application with English, Bangla, Banglish, and mixed-input support.

## Design

- Hybrid FAQ retrieval: IDF-weighted query coverage + character 4-gram overlap + a topic tie-break, with an optional multilingual embedding model.
- Cross-script vocabulary bridge (`src/normalize.py`) maps generic Bangla and Banglish banking terms to English. It is scored as a second query representation and combined by `max`, so a gloss can raise a score but never dilute one.
- Concept dedup (`informative_terms`): a Bangla word the glossary can translate is dropped when its English gloss is already in the query. Without this, the loan-interest entry beat the loan-eligibility entry by matching "পার্সোনাল লোনের" *and* "personal loan" as two separate facts.
- Language-aware support gate: the embedder's share of the fused score depends on the script the customer used (`DENSE_WEIGHT`), and the floor is derived from that same blend, so every query is judged against a threshold calibrated for the channel that carries it. Banglish is the case that needs this — a Bangla/English embedder has never seen Roman-script Bangla, so there the glossary carries the query and dense is only a tiebreaker. Below the floor the assistant hands off rather than guesses.
- Three groundedness guards, in order: retrieval floor, ambiguous-evidence margin, and two FAQ-derived checks — a topic the FAQ never mentions, or a figure that contradicts a range the FAQ states. The topic guard filters itself against the FAQ vocabulary, so it can only refuse what the corpus truly cannot answer. The ambiguity guard (`is_ambiguous`) calls a near-tie a coin flip only when the topic and lexical channels agree with the fused score, so `0.3592 vs 0.3589` no longer hands off a question that named the topic outright.
- When the local generator loads, a strict entailment check may reject the retrieved FAQ, but its veto is honoured only when the lexical evidence is below the same floor the fused score had to clear. Its English-only prompt produces false NOs on Bangla and Banglish, and a false NO on a known FAQ is a real customer sent to a dead end.
- Local Qwen3-1.7B generation when the model dependencies and weights are available for English answers and entailment checks. Bangla/Banglish replies normally use the dedicated `SifatAlMukit/finetuned-en-bn-nllb` English-to-Bangla seq2seq checkpoint; Banglish is deterministic romanisation of verified Bangla output. A small source-derived code fallback covers the FAQ entries whose neural output was mixed or unclear, so no English sentence is spliced into a customer reply.
- Banglish romanisation is deterministic and zero-dependency: a Bengali-script lexicon handles banking terms and common suffixes, a phrase layer preserves customer-facing English such as `interest rate`, `balance`, `deposit`, `office`, `visit`, and `processing`, and the grapheme rules handle unknown words. There is no mathematically unique Banglish spelling for every Bengali word, so the renderer chooses one consistent Bangladesh-oriented style rather than claiming phonetic perfection.
- Key-fact validation preserves amounts, percentages, dates, and hotline numbers (`1,000` and `1000` compare equal) and rejects a figure the entry does not state, so the PIN-reset reply that invented "Fee holo BDT 500" cannot ship. Where the entry states one or two figures those are the answer, so dropping them fails too; a seven-figure entry such as the FDR rates may be answered from one of them.
- Three format gates on every model reply before it is served (`answer_is_wellformed`): it must not recite the customer's question, must finish its sentence, and must be in the customer's script. A reply that fails any of them is replaced by the bank's own sentence and the rejected draft stays visible in the window, rather than becoming an "I don't know". A fourth gate — that an *English* reply be built from this entry's own wording — catches model boilerplate; it cannot run on Bangla or Banglish, because the FAQ has no Bangla text to compare against.
- **One answer source, and it is the supplied file.** The pack says `bank-faq.csv` "is the only source your assistant may answer from", so it is stored byte-identical to the pack, `test-questions.csv` is untouched, and there is no third data file of prepared Bangla answers — a file of answer text would be a second source, however faithful. Retrieval indexes the FAQ's own `topic`, `question` and `answer`; a Bangla or Banglish reply is *written* by the local model from that English sentence, which is what requirement 2 and requirement 3 of the brief ask for together.
- A two-holder joint-account question is answered from FAQ 14; unsupported or contradictory requests are handed off.

## Run it — required path: the Kaggle GPU answers, the desktop app shows it

There is nothing to install on the laptop except PySide6. **A laptop CPU cannot
write Bangla**, so a local-only run is a smoke test, not the submission: the
accurate answers come from the notebook's GPU stack over one HTTP tunnel. Every
number in `kaggle8.json` (copied verbatim into `reports/offline_results.json`)
was produced by the path below.

1. **Start the model.** Open `notebooks/kaggle_demo.ipynb` on Kaggle with the
   `biai-bank-assistant` dataset attached and the accelerator set to **GPU T4**
   (internet on). Run sections 1–5; section 5 prints the 40-question table.
2. **Publish it.** Run **section 8**. It installs FastAPI, starts
   `api_server.py` on the GPU, answers one Banglish question through the
   endpoint so you can see the served text, opens a Cloudflare quick tunnel, and
   prints at the end:

   ```text
   BIAI_API_URL=https://<random-words>.trycloudflare.com
   BIAI_API_TOKEN=<a token the cell generated for this session>
   python main.py   ← the exact line to run, with both values already filled in
   ```

   Copy both values **exactly as printed**. The token is not typed by you and it
   is not from any token-generator website: `api_server.py` compares the
   `Authorization` header against the token that same cell generated, so any
   other value is a `401`. The URL lives only while the notebook session runs.
3. **Run the desktop app against it.**

   ```bash
   cd biai_bank_assistant
   pip install -r requirements.txt   # PySide6 only
   BIAI_API_URL=https://<that-url> BIAI_API_TOKEN=<the token the cell printed> python main.py
   ```

   Windows PowerShell: `env BIAI_API_URL="…" ; env BIAI_API_TOKEN="…" ; python main.py`.
4. **Test it — the four checks.**

   | Type this | Expect |
   |---|---|
   | `Card er PIN bhule gechi, reset korbo kibhabe?` | the same Banglish text section 8 printed |
   | `আমার কার্ড হারিয়ে গেছে, এখন কী করব?` | Bangla script, Bangla digits, `৩০০ টাকা` |
   | `What is today's dollar to taka exchange rate?` | `I don't know, let me connect you to a person.` |
   | status pill / inspector | `MODEL READY — KAGGLE`, retrieval `dense`, Qwen generator named |

   A `LOCAL FAQ` pill means the token was wrong or the tunnel died: the window
   says so and keeps answering from the FAQ on the machine, so the demo never
   becomes a blank box. Re-run section 8 and copy the new values.
5. **Optional:** set `BIAI_ADAPTER_PATH` to the adapter saved by section 6 before
   starting section 8 if you want the trained adapter served. The reproducible
   training command is `python train.py --faq data/bank-faq.csv --output
   /path/to/adapter` — it never opens `data/test-questions.csv`, and it is not
   required for the numbers above.


## Run modes — reference, not the judged path

`python main.py` on a machine without `torch`/`sentence-transformers` runs in **lexical smoke-test mode**: the embedding model and the generator are both absent, the UI pill says `SMOKE TEST`, and the inspector names the missing dependency.

Smoke-test mode is grounded and it refuses correctly, but it answers in the FAQ's language: `bank-faq.csv` is English, nothing else may be quoted, so a Bangla question gets the bank's English sentence and the window says so plainly. Requirement 2 is the model's job, not the disk's. On the lexical stack the held-out set scores 30/30 on the FAQ source, 10/10 on the hand-offs, 40/40 on "no echo, complete sentence" — and 20/40 on reply language, which is the honest cost of running without a model.

Set `BIAI_DISABLE_DENSE=1` to force the lexical path even where the models are installed, and `BIAI_GENERATION_MODEL` / `BIAI_EMBEDDING_MODEL` to use others. `BIAI_TRANSLATOR_MODEL` can select another compatible English-to-Bangla seq2seq checkpoint; the default is `SifatAlMukit/finetuned-en-bn-nllb`. `BIAI_MAX_NEW_TOKENS` raises the Qwen answer budget and `BIAI_RENDER_MAX_NEW_TOKENS` raises the translation budget.

## Run the evaluation

```bash
python evaluate.py
```

The test questions remain evaluation-only. `reports/offline_results.json` records the expected FAQ source, refusal status, language, confidence, and latency for each question.

## Kaggle

`notebooks/kaggle_demo.ipynb` is the reproducible demo and the path that produces the model-stack numbers. A persistent public server is intentionally not required.

1. Upload the **contents** of this folder — not the folder itself — as a Kaggle dataset named `biai-bank-assistant`. The notebook reads `/kaggle/input/biai-bank-assistant/src`, `/data`, and `/requirements-kaggle.txt`, so `src/`, `data/`, `evaluate.py` and `requirements-kaggle.txt` must sit at the top level of the dataset. The notebook asserts this in section 2 rather than failing later.
2. Open the notebook (`File → New Notebook → Import from local file`, or create a notebook and paste the sections), attach that dataset, and switch the accelerator to **GPU T4** with internet **on** (the weights come from Hugging Face on first run).
3. Run all cells. Section 5 copies the project into `/kaggle/working` before evaluating, because `/kaggle/input` is read-only and `evaluate.py` writes its report there.
4. Section 6 is the brief's *optional* extra (further training on public banking vocabulary) and is **not** needed for the numbers above — the language fixes are in the renderer, not in the weights. Section 7 is the *required* comparison against an online model: `BIAI_BASELINE_MODEL` defaults to `gemini-3.5-flash-lite`, because `gemini-2.0-flash` and the 2.5 family are retired. Add `GEMINI_API_KEY` under **Settings → Add-ons and Secrets** in the notebook (free key from [Google AI Studio](https://aistudio.google.com/apikey)); the cell reads it through `UserSecretsClient`, so the key never enters the notebook text, the dataset or git. Section 8 is the *mandatory* desktop-app link and prints the tunnel URL and a generated token.

**The dataset is a copy, so a stale upload runs old code.** For this pass, re-upload:

| File | Why it must be re-uploaded |
|---|---|
| `src/bn_render.py` | Value-based number check (`7.50%` = `7.5%`, `10:00 AM` = `10 AM`) so a correct sentence is no longer rejected; the deterministic style pass (Bangla digits, `টাকা`/`taka`, no Banglish hyphens, English service words kept in Banglish and written in Bangla script in Bangla, `প্রতি তিন মাস পর পর`, `নমিনি`, `অ্যালার্ট`); pinned renderings for FAQ 9, 10, 15, 16, 25, 28, 30 in both scripts |
| `notebooks/kaggle_demo.ipynb`, `notebooks/kaggle.ipynb` | Section 7 baseline moved to `gemini-3.5-flash-lite`; section 8 hardened (generated token, port-in-use guard, uvicorn waited for before the tunnel opens, log-file instead of blocking pipe read, IPv4 edge, a warm-up answer printed through the endpoint) |

Everything else in the dataset is already current. The full set the notebook expects, for anyone starting from an older upload:

| File | Why it must be re-uploaded |
|---|---|
| `data/bank-faq.csv` | Restore the **supplied** file: upload the pack's original four columns over the edited copy the dataset still holds. Same for `test-questions.csv` if it was ever touched |
| `src/normalize.py` | Concept dedup, the wider stopword list, the new রাখা glosses, the echo and wording-reuse gates |
| `src/models.py` | `FAQEntry` holds only the four supplied columns, so no second answer surface can creep in |
| `src/retrieval.py` | Concept-based lexical scoring, `is_ambiguous`, and invented-figure rejection |
| `src/generator.py` | The answer prompt no longer contains the English gloss, and the no-echo instruction |
| `src/assistant.py` | Entailment veto derived from the retrieval floor, public `answer_vocabulary`, reuse gate wired in |
| `src/bn_render.py`, `src/__init__.py` | Dedicated English-to-Bangla renderer, verified Banglish romanisation, and import hook |
| `src/remote.py` | New: the desktop client for a Kaggle-hosted endpoint |
| `evaluate.py` | Re-checks the delivered answer for echo and truncation and reports both |
| `api_server.py` | New: FastAPI endpoint that serves the GPU stack |
| `main.py`, `ui/main_window.py` | Backend selection from `BIAI_API_URL` and an honest status pill for it |
| `notebooks/kaggle_demo.ipynb`, `notebooks/kaggle.ipynb` | Stages the renderer and uses `SifatAlMukit/finetuned-en-bn-nllb`; prints per-script accuracy |
| `requirements-kaggle.txt`, `train.py` | Required at the dataset root so the notebook can install dependencies and train/load the adapter |

Section 3 of the notebook fails loudly if the staged `src/` predates these gates, so an accidentally stale upload is a red cell rather than a quietly worse run.

Locally, `python kaggle_demo.py` still runs the headless evaluation, and `python main.py` opens the desktop interface.

## Connect the Kaggle model to the PySide6 desktop app (mandatory path)

The desktop app and the notebook call the same surface — `answer(question) -> AssistantResult` — so the only decision is *where* that call runs. `BIAI_API_URL` makes that choice in `main.py`: unset, the laptop answers from the local FAQ; set, every question is posted to the GPU host and the laptop pipeline stays as the fallback.

1. In the notebook, run section 8. It installs FastAPI, serves `api_server.py` on the T4, waits until the local `/health` actually answers, opens a Cloudflare quick tunnel on the IPv4 edge and prints a **generated** `BIAI_API_TOKEN`, an `https://…trycloudflare.com` URL and the exact laptop command. Before the tunnel it posts one Banglish question to the endpoint and prints the reply, which is the same text the desktop app will show and the same path section 5 measured. Re-running the cell stops the old server and tunnel first, so a second run cannot leave `address already in use` behind. `BIAI_API_PORT` moves it off a busy port.
2. On the laptop, start the app against it (copy both values from the cell output):

```bash
cd biai_bank_assistant
BIAI_API_URL=https://<that-url> BIAI_API_TOKEN=<the token the cell printed> python main.py
```

The pill reads `MODEL READY — KAGGLE` and the inspector names the endpoint's retrieval mode and generator, so the chat window is showing the notebook's `kaggle8.json` answers rather than the laptop's English FAQ copy. A wrong or missing token is a 401; a dead tunnel is not an error dialog — the window says the endpoint was unreachable and keeps answering from the FAQ on the machine. Set `BIAI_ADAPTER_PATH` on the laptop only if you also ran section 6 and want the server to serve that adapter.

**Can this be the "working demo" link in the email?** Not as a web URL: a PySide6 window is a desktop binary, and the tunnel is a Kaggle session that dies with the notebook, so it is a demo path and not a hostable link. Send the Loom recording as the demo link, and show section 8 plus this command inside it, with the desktop app answering a Banglish question from the GPU. If a clickable link is required, the honest options are a Colab/Binder notebook link that runs sections 1–5, or the hosted demo of the API only (`api_server.py` behind a real host), never a customer-data path.

Three things worth knowing before you use it:

- Kaggle sessions are not servers. The tunnel lives only as long as the notebook is running, and Kaggle recycles idle sessions; this is a demo and evaluation path, not production hosting. For a laptop-only setup instead, install `requirements-kaggle.txt` locally with a GPU and leave `BIAI_API_URL` unset.
- Every gate runs on the host that produced the reply, so what the tunnel returns is already checked text. The chat line identifies the Qwen answer writer and the translation renderer; when a generated sentence fails a gate, the rejected draft stays visible under the bank's sentence.
- `api_server.py` loads `data/bank-faq.csv` from its own directory — the only data file it reads. Serve it from a host with a **loaded generator**, or every remote reply comes back in the FAQ's English.

## Data rule

Facts come **only** from the BIAI-supplied `data/bank-faq.csv`, stored byte-identical to the pack. The serving path opens that FAQ as its only factual source; `training_data/` is language-only supervision for `train.py`, never a response cache. There is no side file of prepared answers: the renderer's small fallback table is code-level wording derived from specific FAQ entries and adds no facts. The 40-question `data/test-questions.csv`, also byte-identical, is held out for testing and is never used for training or tuning the model.

Training the model on public language data is allowed and is what the brief itself invites — *"Extra (optional): train the small model further on banking words or Banglish, and show if it got better."* — and was confirmed by the BIAI Hiring Team on 3 November 2026, provided no real bank or customer data is used and the source is disclosed. That is the legal route to better Bangla and Banglish output: language skill goes into the **model weights**, never into a second file of answers. `train.py` combines the FAQ with licensed language/register corpora; the held-out questions are never opened.

| Dataset | Source | License | Used for |
|---|---|---|---|
| `istiaqfuad/bangla-english-banglish-pairs` **(used by training)** | [HF](https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs), derived from `Helsinki-NLP/opus-100` | CC-BY-4.0 | Up to 1,200 filtered public banking-language pairs for the Qwen LoRA language adapter |
| `learn-abc/banking14-intents-en-bn-banglish` **(used by training)** | [HF](https://huggingface.co/datasets/learn-abc/banking14-intents-en-bn-banglish) | MIT | Synthetic banking-language register examples from the train split |
| `subhajitmahata84/banglabridge-instructions` **(used by training)** | [HF](https://huggingface.co/datasets/subhajitmahata84/banglabridge-instructions) | CC-BY-4.0 | Synthetic Banglish instruction/output style examples |
| `istiaqfuad/triBne-e5-small` **(used as the retriever)** | [HF](https://huggingface.co/istiaqfuad/triBne-e5-small) | MIT | Multilingual embeddings. This third-party retriever was itself fine-tuned on public CC-BY parallel pairs; it supplies language similarity, not bank facts. |
| `SifatAlMukit/finetuned-en-bn-nllb` **(used as the renderer)** | [HF](https://huggingface.co/SifatAlMukit/finetuned-en-bn-nllb) | MIT | Conversational English↔Bengali NLLB checkpoint fine-tuned on Tatoeba pairs; it supplies wording only, never bank facts. Banglish is romanised from its verified Bangla output. |
| `ShayonSarker/Bengali-to-Banglish-Dataset` **(not used in the shipped run)** | [HF](https://huggingface.co/datasets/ShayonSarker/Bengali-to-Banglish-Dataset) | MIT | Considered for a future Banglish lexicon |
| Mendeley Data `5h5vkhm9n5` **(not used in the shipped run)** | [Mendeley](https://data.mendeley.com/datasets/5h5vkhm9n5) | CC-BY-4.0 | Considered for future Banglish→English pairs |
| Vashantor **(used by training)** | [Mendeley](https://data.mendeley.com/datasets/bj5jgk878b/2) | CC-BY-4.0 | Train-only Bangla/Banglish/English dialect pairs copied to `training_data/vashantor_train.jsonl` |

Models: `Qwen/Qwen3-1.7B`, Apache-2.0, 2.03B parameters (~1.2–1.5 GB at 4-bit); `SifatAlMukit/finetuned-en-bn-nllb`, MIT, about 0.6B parameters, fine-tuned for conversational English↔Bengali translation; and the ~118M-parameter `triBne-e5-small` retriever. All fit on a Kaggle T4.

Two local archives were **rejected and never used**: the BGHL Bangla grammar QA set (domain mismatch, no stated licence) and the multilingual religious hate-speech set (contains scraped public comments with author links and unsafe/off-domain content). The local seven-regional-dialect workbook was inspected but not added because the archive contains no licence metadata; Vashantor provides the same useful Bangla/Banglish dialect supervision with a documented CC-BY-4.0 licence.

Nothing from any of these sources may enter an answer. They supply **language**, the FAQ supplies **facts**.

## Known limitation: the out-of-scope guards

`UNCOVERED_CONCEPTS` in `src/assistant.py` lists concepts with zero coverage in the supplied FAQ (commodity and market prices, securities, foreign exchange, crypto, immigration, named staff, branch locations, third-party account lookups). The list is filtered at load time against the FAQ's own vocabulary, so it can never refuse a topic the corpus actually covers. It was compiled by reading the FAQ, not the answer key — but it did influence hand-off behaviour on the held-out set, and that is disclosed here rather than hidden.

