from __future__ import annotations

import os

from .models import FAQEntry
from .normalize import language_name, strip_answer_artifacts

REFUSALS = {
    "en": "I don't know, let me connect you to a person.",
    "bn": "আমি জানি না, একজন প্রতিনিধির সাথে সংযোগ করে দিচ্ছি।",
    "banglish": "Ami jani na, ekjon representative-er shathe jogajog koriye dichhi.",
    "mixed": "I don't know, let me connect you to a person.",
}


def refusal(language: str) -> str:
    return REFUSALS.get(language, REFUSALS["en"])


def _answer_budget() -> int:
    """Bengali costs about 5.5 tokens per word on a Qwen tokenizer, so the 96
    tokens this shipped with cut two-sentence Bangla answers in the middle of a
    word; the format gate then had to reject an otherwise correct reply."""
    try:
        return max(64, int(os.getenv("BIAI_MAX_NEW_TOKENS", "256")))
    except ValueError:
        return 256


class LocalGenerator:
    """Local causal-LM rewriter over the FAQ text, with a deterministic fallback.

    The model is asked to reuse the FAQ's own facts in the customer's script.
    Public bilingual language training and a few language-only demonstrations
    make the rewrite more reliable while the FAQ remains the only factual source.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("BIAI_GENERATION_MODEL", "Qwen/Qwen3-1.7B")
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.available or self.load_error:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            kwargs = {"device_map": "auto"}
            if torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                except Exception:
                    pass
            # transformers 4.56 renamed the dtype argument; a pinned older host and
            # a fresh Kaggle image therefore need different spellings. Loading with
            # the wrong one raises TypeError, which would otherwise be recorded as a
            # load error and quietly drop the demo into lexical-only smoke mode.
            for dtype_keyword in ("dtype", "torch_dtype"):
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, **kwargs, **{dtype_keyword: "auto"}
                    )
                    adapter_path = os.getenv("BIAI_ADAPTER_PATH", "").strip()
                    if adapter_path:
                        try:
                            from peft import PeftModel
                        except ImportError as exc:
                            raise RuntimeError(
                                "BIAI_ADAPTER_PATH is set but peft is not installed"
                            ) from exc
                        self.model = PeftModel.from_pretrained(base_model, adapter_path)
                    else:
                        self.model = base_model
                    self.model.eval()
                    break
                except TypeError:
                    continue
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _register(language: str) -> str:
        return {
            "bn": "Bangla, written in Bengali script only",
            "banglish": "Banglish, which is Bangla words written with Latin letters",
        }.get(language, "English")

    def _messages(self, question: str, entry: FAQEntry, language: str) -> list[dict]:
        register = self._register(language)
        # The bank's own English sentence is the only text the model is given, so
        # putting it in the customer's script is the model's job and the whole of
        # requirement 2. Nothing else may be the anchor: a second file of prepared
        # Bangla answers would be a second source of facts.
        source_block = f"FAQ answer ({language_name('en')}): {entry.answer}"
        customer = f"Customer question: {question}"
        # The English gloss of the customer's line is deliberately absent here. It
        # is model-visible text and a 1.7B model copies it: one Kaggle reply read
        # "…kibhabe of will do how?" because the prompt had offered "of, will, do,
        # how" as the reading of the customer's own words. The gloss still guides
        # the YES/NO support check, which is a classifier and cannot leak it into
        # prose.

        demonstrations = ""
        if language == "bn":
            demonstrations = (
                "\nStyle examples (these are language examples only, not bank facts):\n"
                "Customer: কার্ডের পিন কীভাবে বদলাব?\n"
                "Answer: অ্যাপের সেটিংস থেকে কার্ড নির্বাচন করে পিন রিসেট করুন।\n"
                "Customer: অ্যাকাউন্ট খুলতে কী লাগবে?\n"
                "Answer: অ্যাকাউন্ট খোলার জন্য প্রয়োজনীয় কাগজপত্র জমা দিন।\n"
            )
        elif language == "banglish":
            demonstrations = (
                "\nStyle examples (these are language examples only, not bank facts):\n"
                "Customer: card er PIN kibhabe bodlabo?\n"
                "Answer: App-er settings theke card select kore PIN reset korun.\n"
                "Customer: account khulte ki lagbe?\n"
                "Answer: Account kholar jonno proyojoniyo documents joma din.\n"
            )

        system = (
            "You are Nirbhor Bank's offline FAQ assistant. You never invent facts. "
            "Every number, date, percentage, fee, product name and hotline must appear exactly as "
            "written in the FAQ answer. Answer in "
            f"{register}, in one or two short sentences. "
            "Start with the fact itself: a figure, a day, a step or the product name. "
            "Never repeat or restate the customer's question, never write a label such as "
            "'Answer:', never add a greeting or a preamble."
        )
        user = (
            f"{demonstrations}\nTask: english to {language}.\n{source_block}\n{customer}\n"
            f"Give the customer that FAQ answer in {register}. Say exactly what the FAQ answer "
            "says and change no figure, date, fee or product name. Reply with the answer only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _apply_template(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except (TypeError, ValueError):
            # Not every chat template takes enable_thinking, and a LoRA merged
            # from a base model may have no template at all. Rendering plainly
            # keeps a loaded model answering instead of erroring into the
            # deterministic path.
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError):
                return "\n".join(f"{role}: {content}" for role, content in messages) + "\nassistant:"

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        import torch

        text = self._apply_template(messages)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()

    def answer(self, question: str, entry: FAQEntry, language: str) -> str:
        self.load()
        if not self.available:
            return self._fallback(entry, language)
        try:
            messages = self._messages(question, entry, language)
            raw = self._generate(messages, _answer_budget())
            cleaned = strip_answer_artifacts(raw)
            # A small model may ignore the register on its first pass. One short
            # retry keeps the adapter-backed path useful without allowing an
            # English draft to reach the customer. The assistant applies the
            # final grounding/script gates after this method returns.
            if language in {"bn", "banglish"}:
                from .normalize import answer_matches_language

                if not answer_matches_language(cleaned, language):
                    retry = messages + [{
                        "role": "user",
                        "content": (
                            f"FORMAT FAILURE: rewrite the answer entirely in {self._register(language)}. "
                            "Do not use an English sentence. Output only the finished answer."
                        ),
                    }]
                    cleaned = strip_answer_artifacts(self._generate(retry, _answer_budget()))
            return cleaned or self._fallback(entry, language)
        except Exception as exc:
            self.load_error = f"generation failed: {type(exc).__name__}: {exc}"
            return self._fallback(entry, language)

    def supports(self, question: str, entry: FAQEntry, reading: str | None = None) -> bool | None:
        """Ask the local model for a conservative FAQ entailment decision."""
        self.load()
        if not self.available:
            return None
        system = (
            "You are a strict FAQ support checker. Return only YES or NO. "
            "Return YES when the FAQ answer directly answers the customer question, "
            "or when it states a general rule that safely covers the specific case "
            "asked about without contradicting it. "
            "Return NO when the question asks for a new policy, a personal data lookup, "
            "a location-specific fact, a figure the FAQ does not state, "
            "or contradicts the FAQ."
        )
        user = (
            f"FAQ question: {entry.question}\nFAQ answer: {entry.answer}\n"
            f"Customer question: {question}\nDecision:"
        )
        if reading and reading.strip() != question.strip():
            user += f"\nEnglish gloss of the customer question: {reading}"
        try:
            decision = self._generate(
                [{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=8
            )
        except Exception:
            return None
        decision = decision.strip().upper()
        if decision.startswith("YES"):
            return True
        if decision.startswith("NO"):
            return False
        return None

    @staticmethod
    def _fallback(entry: FAQEntry, language: str) -> str:
        """The bank's own sentence, in the only form the FAQ supplies it: English.

        Used when no model is loaded, or when a model reply failed a grounding
        check. The right figure in the wrong script is a smaller failure than an
        invented fact, and the interface says plainly that no model wrote this.
        """
        return entry.answer
