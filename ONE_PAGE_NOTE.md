# NB Assist — one-page evaluation note

## What I built

NB Assist is a PySide6 desktop assistant backed by a local small language model. It retrieves only from the supplied Nirbhor Bank FAQ, responds in English, Bangla, or Banglish, validates key facts, and hands unsupported questions to a human.

## How I tested it

I ran the supplied 40-question test set: 30 supported questions and 10 intentionally unsupported questions. The Kaggle notebook records the per-question source FAQ ID, refusal status, detected language, answer, and latency. I also test the direct/generalized/contradictory joint-account cases around FAQ 14.

## Results

Fill this section after the Kaggle run:

- Supported-question accuracy: `__/30`
- Correct unsupported refusals: `__/10`
- Overall groundedness gate accuracy: `__/40`
- Language/register accuracy: `__%`
- Warm-model median latency: `__ ms`
- Warm-model p95 latency: `__ ms`

## What does not work well

Small local models can still produce awkward Banglish phrasing and may need a retry when preserving numbers in Bangla script. Regional dialects are not treated as separate banking policies. The assistant refuses when evidence is weak, which is safer but can reduce recall on unusually phrased questions.

## What I would do next

I would collect consented bank-support paraphrases, fine-tune a small answerability/reranking model for Bangla–English–Banglish, add a human-reviewed answer cache for high-volume intents, and expose the same offline core through a secure service for branch and call-center deployments.
