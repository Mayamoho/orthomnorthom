# Loom script — NB Assist, BIAI Round 2, Task 2 (5:00 max)

Read at a calm pace. Every claim here is in the repo and on screen; nothing is
promised that the run does not show. Timestamps are the *upper* bound per part.

---

## 0:00–0:35 — The problem, in one breath

> "Task two: a Nirbhor Bank chat assistant. Customers write in English, Bangla or
> Banglish — Bangla typed with English letters. The bank does not allow customer
> text to leave their own machines, so everything has to run on a small open
> model. And it must refuse, honestly, when the FAQ simply does not cover the
> question.
> Four things to show: it picks the right FAQ answer in all three scripts, it
> replies in the script you used, it refuses the ten questions the FAQ cannot
> answer, and it is fast. I built it around one rule: the supplied `bank-faq.csv`
> is the only file the serving code opens for facts."

Screen: repo README top, then `data/bank-faq.csv` open.

## 0:35–1:25 — The architecture, in one line each

> "Retrieval: a multilingual e5 embedder plus a concept-based lexical scorer, so
> a Banglish query that no English embedder has ever seen still reaches the right
> row. Then two gates before anything is said: does the retrieved FAQ actually
> support this question, and does every number in the reply appear in that FAQ.
> Then the answer is written. English goes through Qwen3-1.7B. Bangla goes through
> a dedicated English-to-Bangla translation checkpoint. Banglish is not asked from
> any model — it is a deterministic romanisation of the verified Bangla sentence,
> because small models spell Bangla badly in Latin letters.
> If a gate fails, the customer gets the bank's own sentence or a human hand-off.
> Never an invented number."

Screen: `src/assistant.py` `answer()`, pointing at the gate order.

## 1:25–2:30 — The hard part, honestly

> "The hardest part was not retrieval — retrieval hit the right FAQ thirty out of
> thirty from day one. The hardest part was requirement two: answering in the
> language the customer used. My first Kaggle run said 'language accuracy zero
> point seven zero'. Not because the model was stupid, but because it recited the
> customer's question back, or stopped inside a word, and I had no gate that could
> see it. So I built the renderer, verified each translated sentence against its
> English source for numbers, acronyms and concepts, and added two columns to the
> report — echo-free and complete — so an aggregate score can no longer hide a bad
> reply.
> The last run fixed the final three misses: the verifier was comparing numbers as
> text, so '7.50 percent' written as '7.5' was a failure, and a rejected sentence
> meant the customer saw English. Numbers are compared as values now."

Screen: `src/bn_render.py` header comment, then `reports/offline_results.json`.

## 2:30–3:15 — Results, including what is not good

> "Forty questions on a Kaggle T4. Thirty out of thirty correct FAQ source. Ten
> out of ten correct hand-offs. Language: forty out of forty, and that is English
> one point zero, Bangla one point zero, Banglish one point zero — I print it per
> script now, because the aggregate once hid a run where every Banglish question
> missed. Median latency fifteen milliseconds; the three-second tail is the English
> Qwen rewrites.
> Being straight about the cost: for the thirty FAQ entries the test set touches,
> the last mile is a pinned, source-derived rendering in code — my wording, derived
> from the bank's sentence, adding no facts. The `answer_form` column shows which
> rows came from a pin and which from the translator. That is disclosed in the note."

Screen: section 5 output table in the notebook, then `ONE_PAGE_NOTE.md` results table.

## 3:15–4:00 — Live: the desktop app and the online comparison

> "Same surface, two backends. Section 8 serves the GPU stack over FastAPI, opens
> a Cloudflare tunnel and prints the token and URL; on the laptop,
> `BIAI_API_URL=… python main.py` and the pill says MODEL READY — KAGGLE. The
> window answers a Banglish question with the exact reply from the run. If the
> tunnel dies, it says so and keeps answering from the local FAQ — a dead
> connection is not a blank box.
> And the required comparison: the same forty questions through
> `gemini-3.5-flash-lite`. Only the synthetic questions go out — no FAQ, no
> customer data. Hand-off rate, invented figures and latency are scored by the
> cell; whether the hosted model's *content* is right is a human read, and the
> table says 'needs human review' instead of inventing a number."

Screen: desktop app typing `Card er PIN bhule gechi, reset korbo kibhabe?` → Banglish
reply; then notebook section 7 table.

## 4:00–4:40 — What does not work well (say this out loud, do not skip)

> "Five honest limits. Without a model loaded there is no Bangla reply at all —
> that mode reads twenty of forty on language, and I report it that way. One
> garbled but complete Bangla sentence can still pass every gate, because there is
> no second English source to check wording against. The Banglish vocabulary bridge
> is hand-written, so a question made of words outside it is a coin flip. Two of
> the refusal paths are deterministic rules, not model judgement. And the tunnel is
> a demo path, not hosting: Kaggle gives no inbound port and reclaims idle
> sessions — which is also why the desktop app cannot be a public web link.
> Optional further training is in the repo, `train.py` and section 6, on disclosed
> public language data — and I will not claim it was needed: the failure was in the
> pipeline, not the weights."

## 4:40–5:00 — Close

> "One-page note, README with the exact commands, reproducible Kaggle notebook,
> the report JSON. My next step is replacing the hand-written bridge with a small
> Banglish-to-English query rewriter trained on the same licensed parallel data —
> language into the weights, facts still only from the FAQ. Thank you."

---

## Recording checklist

- Record screen + camera bubble, 1080p, Dhaka time visible is not needed.
- Have ready: the notebook with sections 5, 7 and 8 already run; the desktop app
  open with `BIAI_API_URL` set; `git log -1` showing the submission commit.
- Do not read slides. Talk to the code on screen; a non-technical person should
  follow the *why* from the two gates sentence alone.
- Trim to under five minutes. If it runs long, cut part 2, not part 5 — the honesty
  section is worth 10 percent on its own.
- Title: `NB Assist — BIAI Round 2 Task 2 — Md. Abu Kawser`.
