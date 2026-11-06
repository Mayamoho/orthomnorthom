# Language-only training data

`vashantor_train.jsonl` is the **train split only** from Vashantor (CC BY 4.0,
[Mendeley version 2](https://data.mendeley.com/datasets/bj5jgk878b/2)). It contains
Bangla, Banglish, English, and regional dialect pairs. It supplies language and
dialect supervision only; it is never opened by the serving assistant and cannot
add facts to a bank answer.

The training script also streams these public train splits from Hugging Face:

- `learn-abc/banking14-intents-en-bn-banglish` (MIT), synthetic banking-language register examples.
- `subhajitmahata84/banglabridge-instructions` (CC BY 4.0), synthetic Banglish instruction examples.

The BIAI FAQ is the only factual source. `test-questions.csv` is evaluation-only
and is never read by `train.py`.
