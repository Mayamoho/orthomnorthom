# NB Assist — Offline Bilingual Bank Assistant

NB Assist is a local-first FAQ assistant for the BIAI AI Engineer Round 2 Task 2. It answers only from the supplied Nirbhor Bank FAQ and hands unsupported questions to a person. The interface is a PySide6 desktop application with English, Bangla, Banglish, and mixed-input support.

## Design

- Hybrid FAQ retrieval with character and word overlap, plus an optional multilingual embedding model.
- Conservative support gate: low-confidence questions are refused instead of guessed.
- When the local generator is available, a second strict entailment check must approve the retrieved FAQ before generation.
- Local Qwen3-1.7B generation when the model dependencies and weights are available.
- Deterministic fallback mode for UI and retrieval smoke tests.
- Key-fact validation preserves amounts, percentages, dates, and hotline numbers.
- FAQ 14 / test question 15 is handled as a safe generalization: the answer states the two-holder rule without inventing a spouse-specific policy.

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

Use a GPU notebook for Qwen3 1.7B with 4-bit loading. Install `requirements-kaggle.txt`, copy the `data/` and `src/` directories, run `python kaggle_demo.py`, and save the generated report. A persistent public server is intentionally not required; the notebook is the reproducible demo. Run `python main.py` locally for the PySide6 desktop interface.

## Data rule

The final judged run uses only the BIAI-supplied Task 2 files. Additional language datasets may be used only for a separately labelled experiment if the task owner authorizes them.
