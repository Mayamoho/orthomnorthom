# Loom script — NB Assist, BIAI Round 2, Task 2 (5:00 max)

Plain words, one idea per sentence, and every claim is on screen while you say it.
Timestamps are the *upper* bound per part — aim to land each part early.

---

## Screen map — set all of this up before you press record

Open these eleven things and **do not switch windows mid-sentence**. Kill Slack,
close your email, silence notifications, shut the tabs that are not on this list.

| # | What | Exactly what to have on it | Used in |
|---|---|---|---|
| A | **Browser tab 1 — the Kaggle notebook** `notebooks/kaggle.ipynb` | Sections 5, 7 and 8.1 already run. Parked on section 5's 40-row table at the start; scroll, never re-run, while recording | Parts 4, 5 |
| B | **Browser tab 2 — GitHub** `Mayamoho/orthomnorthom/biai_bank_assistant` | README at the top, so the "Run it" steps and the results table are visible together | Parts 1, 6, 7 |
| C | **Browser tab 3 — ngrok dashboard** `dashboard.ngrok.com` | The **Domains** page, showing your `xxxxxxxx-ngrok-free.app` row | Part 5 (five seconds) |
| D | **Editor pane, left** | `data/bank-faq.csv` open — the header row and FAQ entries 1–4 readable | Part 1 |
| E | **Editor** | `src/assistant.py` scrolled to `answer()`, so the gate order fits one screen | Part 2 |
| F | **Editor** | `src/bn_render.py` — the header comment at the top, then `style_answer()` | Part 3 |
| G | **Editor** | `reports/offline_results.json` — the top block (`language_accuracy_by_script`, `median_latency_ms`) and one `banglish` row below it | Parts 3, 4 |
| H | **Editor** | `reports/baseline_comparison.md` in **rendered preview**, not raw source | Parts 5, 6 |
| I | **The desktop app** — NB Assist window | Pill reading `MODEL READY — KAGGLE`, started with the ngrok URL and token, input box empty | Part 5 |
| J | **Terminal** | Prompt sitting in `biai_bank_assistant/`, `git log -1` already shown | Part 6 |
| K | **Editor** | `ONE_PAGE_NOTE.md` — the disclosure paragraph in section 5 | Part 6 |

One monitor is enough if you full-screen the notebook and the app and switch with
the keyboard; two is better — left holds the thing you are talking about, right
holds everything already loaded. Set the editor zoom to 14–16 px and the notebook
tab to browser zoom 125%, so a reviewer on a laptop reads it without clicking.
Record screen plus a small camera bubble, 1080p.

---

## 0:00–0:35 — Part 1: what this is, in one breath

**Show:** tab B (README top), then window D (`data/bank-faq.csv`).

> "Task two. A chat assistant for Nirbhor Bank's customer questions.
> Customers write three ways: English, Bangla, and Banglish — Bangla words typed
> with English letters, the way you text a friend. The bank does not allow customer
> text to leave its own machines, so I could not call a big hosted model.
> Everything runs on one small open model.
> And it has to say 'I don't know' when the FAQ simply does not cover the question,
> instead of guessing politely.
> Four things I will show: it finds the right FAQ answer in all three scripts, it
> replies in the script you used, it refuses the ten questions the FAQ cannot
> answer, and it is fast. One rule kept me honest throughout — this file on screen,
> `bank-faq.csv`, is the only thing the code reads for facts. It has thirty entries,
> and every answer you see today comes out of one of those thirty rows or out of a
> hand-off."

## 0:35–1:25 — Part 2: how it works, three sentences each

**Show:** window E (`src/assistant.py`, `answer()`). Move the cursor down the gate
order as you speak: retrieve → gate one → gate two → write.

> "First, finding the right FAQ row. A multilingual embedding model handles the
> meaning and a word-overlap scorer handles the literal matching. That matters
> because a Banglish sentence like `FDR korte minimum koto taka lage?` is something
> an English or Bangla embedder has never seen — the vocabulary bridge is what
> reaches the row anyway.
> Then, before anything is said to a customer, two gates. Gate one: does the FAQ row
> I found actually support this question — if not, hand it to a person. Gate two:
> does every number in my reply appear in that FAQ row — if not, the reply is thrown
> away. So a fee the bank never stated cannot reach a customer, no matter how nicely
> the sentence reads.
> Then the answer is written. English comes from Qwen3-1.7B. Bangla comes from a
> checkpoint trained for English-to-Bangla. Banglish is not asked from any model at
> all — it is built by spelling that verified Bangla sentence in English letters,
> because small models spell Bangla badly in Latin script.
> When a gate fails, the customer still gets an answer: the bank's own sentence from
> the FAQ, or a hand-off to a person. Never an invented number."

## 1:25–2:30 — Part 3: the hard part, and what I did about it

**Show:** window F (`src/bn_render.py`, header comment), then window G
(`reports/offline_results.json`) scrolled to one row's `echo_free` and `complete`
fields.

> "The hard part was not finding the FAQ. Retrieval picked the right row thirty out
> of thirty from the first day. The hard part was answering in the language the
> customer used.
> One of my earlier runs — it is still in this repo's history — printed language
> accuracy zero point seven zero. Not because the model was stupid: it recited the
> customer's own question back at them, or stopped halfway through a word. And one
> average cannot show you that, so I could see the score was bad but not which
> reply was bad.
> So two fixes. Every translated sentence is checked against the English sentence it
> came from before it is used — same numbers, same acronyms, same ideas, a finished
> sentence, no foreign letters leaking in. A sentence that fails is retried, and if
> it still fails the bank's own words are used. And the report grew two columns,
> echo-free and complete, so a bad reply now has a row number instead of hiding
> inside one decimal.
> The last fix was a silly one. The checker compared numbers as text, so '7.50
> percent' written as '7.5' counted as a failure — and every failure meant an
> English sentence in front of a customer who had asked in Bangla. Numbers are
> compared as values now."

## 2:30–3:15 — Part 4: results, including the part I do not like

**Show:** tab A, section 5's output table — scroll one row so the columns are clear —
then window G's top block for the per-script numbers and the latency lines.

> "Forty questions, on a Kaggle T4 GPU.
> Right FAQ row: thirty out of thirty. Correct hand-offs: ten out of ten. Language:
> forty out of forty — and that is one point zero for English, one point zero for
> Bangla, one point zero for Banglish. I print it per script now, because one
> average once hid a run where every single Banglish question missed.
> Speed: median fifteen milliseconds. The three-second tail is the English rewrites
> that Qwen had to do itself.
> Now the part I would rather you hear from me than find in the code. For the twenty
> Bangla and Banglish answers in this run, the last mile is a pinned rendering,
> written in code and derived from the bank's own sentence — my wording, no facts
> added. Six English answers came from Qwen, and four kept the bank's sentence
> because the rewrite was rejected. The `answer_form` column says which of these
> each row was, and the one-page note discloses it in plain words."

## 3:15–4:15 — Part 5: live, and the required comparison

**Show:** window I — the desktop app. Type each question, then **stop talking until
the reply lands**. The three quick chips under the chat box are there if you prefer
clicking to typing Bangla.

| Type this | What must appear on screen |
|---|---|
| `Card er PIN bhule gechi, reset korbo kibhabe?` | Banglish in Latin letters, the NB Go path — the same text section 8.1 printed |
| `আমার কার্ড হারিয়ে গেছে, এখন কী করব?` | Bangla script, Bangla digits, `১৬৯৯৯` and `৩০০ টাকা` |
| `FDR korte minimum koto taka lage?` | Banglish with `25,000 taka` and the three tenures |
| `What is today's dollar to taka exchange rate?` | `I don't know, let me connect you to a person.` |

> "Same window, two backends. Notebook section 8.1 serves the GPU stack over FastAPI
> and puts it behind one fixed ngrok address, so the command you type on the laptop
> never changes between sessions. It is `BIAI_API_URL=… python3 main.py`, and the
> pill then reads MODEL READY — KAGGLE. What you just read was written on the T4:
> every gate ran there before the text crossed the tunnel.
> And if the tunnel dies, the pill says so and the window keeps answering from the
> FAQ stored on this machine. A dead connection is not a blank box."

**Show:** tab C (the ngrok **Domains** page) for five seconds while you say where the
address comes from. Then tab A, section 7's own summary block — scroll to the three
rows that begin `reply in the script the customer used`, so the next numbers are
read off the cell and not eyeballed by you. Then window H
(`baseline_comparison.md`, rendered, which ends with the same three findings).

> "The comparison the pack asks for: the same forty questions through
> `gemini-3.5-flash-lite`, given the same retrieved FAQ entry and the same rules our
> own model gets — like-for-like, not a weaker prompt. Only the forty synthetic
> questions and the FAQ text leave the machine; no customer data.
> It ran clean: forty answers obtained, no rate-limit gaps, median 546 milliseconds,
> ten of ten hand-offs correct, and not one invented figure.
> And where it is weak is exactly where I built, and the cell counts it rather than
> me guessing: eleven Banglish questions, **zero** replies in Banglish — all eleven
> came back in English. Bangla: seven of nine. English: ten of ten. Two of the thirty
> replies are fragments instead of sentences, and ten leave out a figure the FAQ
> entry states — the printed-statement price, the twenty-four-hour credit, the FDR
> tenures. Whether each answer is *complete* is a human read, so I am calling it one.
> So the hosted model is right about the money and wrong about the language. That gap
> is the whole reason the renderer exists."

## 4:15–4:45 — Part 6: what does not work well

**Show:** stay on window H for the first two limits, then window K (`ONE_PAGE_NOTE.md`)
and terminal J. Say this part out loud and do not skip it — this is where the
credibility lives.

> "Five limits, honestly.
> Without a model loaded there is no Bangla reply at all — that mode reads twenty of
> forty on language, and the report prints twenty of forty.
> One garbled but finished Bangla sentence can still pass, because the checker proves
> a reply carries the right numbers and ideas — it has nothing to compare *wording*
> against, so there is no second translation to score it against.
> The Banglish vocabulary bridge is hand-written, so a question built from words
> outside it is a coin flip.
> Two of the refusal paths are deterministic rules, not model judgement.
> And the tunnel is a demo path, not hosting: Kaggle gives no inbound port and takes
> back idle sessions — which is also why this cannot be a public web link.
> Optional further training is in the repo, `train.py` and section 6, on disclosed
> public language data — and I will not claim it was needed, because the failure was
> in the pipeline, not the weights. Twelve regression tests run with no GPU and no
> model files, so a reviewer can check the gates on their own laptop in a minute."

## 4:45–5:00 — Part 7: close

**Show:** tab B — README scrolled so the "Run it" steps and the results table are on
one screen.

> "One-page note, README with the exact commands, a reproducible Kaggle notebook, and
> the report JSON behind every number you saw.
> My next step is replacing the hand-written bridge with a small Banglish-to-English
> query rewriter, trained on the same licensed parallel data — language into the
> weights, facts still only from the FAQ. Thank you."

---

## Recording checklist

- **Before you record:** run notebook sections 5, 7 and 8.1; confirm 8.1 printed the
  URL and the token; start the app with them and check the pill reads
  `MODEL READY — KAGGLE`. Then ask the four Part 5 questions once, so the endpoint is
  warm and you have already seen what each reply looks like.
- Load all eleven windows and tabs from the screen map first. Hunting for a window
  while you are talking costs seconds and reads as unrehearsed.
- Terminal J shows `git log -1` with the submission commit, and the app's own command
  line — `BIAI_API_URL="https://xxxxxxxx-ngrok-free.app" BIAI_API_TOKEN=… python3 main.py`
  — visible above it, so the link between notebook and window is on screen.
- Never re-run a notebook cell live. Scroll it. A GPU cell taking ninety seconds is
  ninety seconds of silence and a chance for the tunnel to be reclaimed.
- Do not read slides. Talk to whatever is on screen. A non-technical viewer should
  get the *why* from the two-gates sentence alone.
- If it runs long, cut **Part 2**, not Part 6 — the honesty section carries weight on
  its own, and Part 5's live demo is the only proof the desktop app really works.
- If a live question misbehaves, keep going: say what happened, then point at the
  notebook table, which holds the same answer.
- No customer data in any frame — only the synthetic FAQ and the forty test questions.
- Title: `NB Assist — BIAI Round 2 Task 2 — Md. Abu Kawser`.
