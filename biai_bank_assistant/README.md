# NB Assist — Offline Bilingual Bank Assistant

NB Assist is a local-first FAQ assistant for the BIAI AI Engineer Round 2 Task 2. It answers only from the supplied Nirbhor Bank FAQ and hands unsupported questions to a person. The interface is a PySide6 desktop application with English, Bangla, Banglish, and mixed-input support.

## Design

- Hybrid FAQ retrieval: IDF-weighted query coverage + character 4-gram overlap + a topic tie-break, with an optional multilingual embedding model.
- Cross-script vocabulary bridge (`src/normalize.py`) maps generic Bangla and Banglish banking terms to English. It is scored as a second query representation and combined by `max`, so a gloss can raise a score but never dilute one.
- Mode-aware support gate: the dense-fused and lexical-only scores sit on different scales, so each has its own floor. Below the floor the assistant hands off rather than guesses.
- Three groundedness guards, in order: retrieval floor, ambiguous-evidence margin, and two FAQ-derived checks — a topic the FAQ never mentions, or a figure that contradicts a range the FAQ states. The topic guard filters itself against the FAQ vocabulary, so it can only refuse what the corpus truly cannot answer.
- When the local generator loads, a second strict entailment check must approve the retrieved FAQ before generation.
- Local Qwen3-1.7B generation when the model dependencies and weights are available.
- Key-fact validation preserves amounts, percentages, dates, and hotline numbers (`1,000` and `1000` compare equal).
- A two-holder joint-account question is answered from FAQ 14; unsupported or contradictory requests are handed off.

## Run modes — read this before judging the demo

`python main.py` on a machine without `torch`/`sentence-transformers` runs in **lexical smoke-test mode**. In that mode the retrieval model and the generator are both absent, so:

- answers are returned as the **verbatim English FAQ text** — a Bangla question gets an English reply, except the two canned FAQ 14 strings;
- the UI status pill shows `SMOKE TEST`, not `OFFLINE READY`, and the inspector names the missing dependency.

Same-language replies (requirement 2 of the brief) only work with the model stack installed, i.e. on Kaggle/Colab GPU or a machine with the weights cached. See `notebooks/kaggle_demo.ipynb`.

## Run locally

```bash
cd biai_bank_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

This starts the PySide6 interface in lightweight smoke-test mode. To enable the local model stack, install `requirements-kaggle.txt` on a GPU machine (or install its packages selectively on a larger local environment). The first model-backed response downloads the configured local models. Set `BIAI_DISABLE_DENSE=1` if you want to run only the lightweight lexical smoke test. Set `BIAI_GENERATION_MODEL` and `BIAI_EMBEDDING_MODEL` to use another local model.

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
4. Section 6 is the brief's *optional* extra (further training on public banking vocabulary). Section 7 is the *required* comparison against an online model and needs a `GEMINI_API_KEY` secret.

Locally, `python kaggle_demo.py` still runs the headless evaluation, and `python main.py` opens the desktop interface.

## Data rule

Answers come **only** from the BIAI-supplied `data/bank-faq.csv`. The 40-question `data/test-questions.csv` is held out for testing and is never used for training or tuning the model.

As authorised by the BIAI Hiring Team on 3 November 2026, public general-purpose language data may be used to adapt the small model, provided no real bank or customer data is used and the source is disclosed. The optional Kaggle training cell actually uses only the first dataset below. The other public datasets were inspected or considered, but are not part of the trained adapter.

| Dataset | Source | License | Used for |
|---|---|---|---|
| `istiaqfuad/bangla-english-banglish-pairs` **(used by optional training)** | [HF](https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs), derived from `Helsinki-NLP/opus-100` | CC-BY-4.0 | Up to 1,200 filtered public Bangla↔English banking-language pairs for a temporary Qwen LoRA query-rewriter experiment |
| `istiaqfuad/triBne-e5-small` **(used as the retriever)** | [HF](https://huggingface.co/istiaqfuad/triBne-e5-small) | MIT | Multilingual embeddings. This third-party retriever was itself fine-tuned on public CC-BY parallel pairs; it supplies language similarity, not bank facts. |
| `ShayonSarker/Bengali-to-Banglish-Dataset` **(not used in the shipped run)** | [HF](https://huggingface.co/datasets/ShayonSarker/Bengali-to-Banglish-Dataset) | MIT | Considered for a future Banglish lexicon |
| Mendeley Data `5h5vkhm9n5` **(not used in the shipped run)** | [Mendeley](https://data.mendeley.com/datasets/5h5vkhm9n5) | CC-BY-4.0 | Considered for future Banglish→English pairs |
| Vashantor **(not used in the shipped run)** | [GitHub](https://github.com/Mukaffi28/Vashantor-A-Large-scale-Multilingual-Benchmark-Dataset) | MIT | Considered for dialect robustness |

Generators: `Qwen/Qwen3-1.7B`, Apache-2.0, 2.03B parameters (~1.2–1.5 GB at 4-bit). Embeddings: `multilingual-e5-small` lineage, ~118M parameters. Both are open-licensed and small enough for a Kaggle T4.

Two downloaded datasets were **rejected and never used**: the BGHL Bangla grammar QA set (domain mismatch, no stated licence) and a multilingual religious hate-speech set (contains scraped real user comments with author links — fails the no-real-data bar outright).

Nothing from any of these sources may enter an answer. They supply **language**, the FAQ supplies **facts**.

## Known limitation: the out-of-scope guards

`UNCOVERED_CONCEPTS` in `src/assistant.py` lists concepts with zero coverage in the supplied FAQ (commodity and market prices, securities, foreign exchange, crypto, immigration, named staff, branch locations, third-party account lookups). The list is filtered at load time against the FAQ's own vocabulary, so it can never refuse a topic the corpus actually covers. It was compiled by reading the FAQ, not the answer key — but it did influence hand-off behaviour on the held-out set, and that is disclosed here rather than hidden.

