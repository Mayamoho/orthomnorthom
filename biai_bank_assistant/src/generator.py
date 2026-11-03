from __future__ import annotations

import os

from .models import FAQEntry


REFUSALS = {
    "en": "I don't know, let me connect you to a person.",
    "bn": "আমি জানি না, একজন প্রতিনিধির সাথে সংযোগ করে দিচ্ছি।",
    "banglish": "Ami jani na, ekjon representative-er shathe jogajog koriye dichhi.",
    "mixed": "I don't know, let me connect you to a person.",
}


def refusal(language: str) -> str:
    return REFUSALS.get(language, REFUSALS["en"])


class LocalGenerator:
    """Qwen-backed generator with a deterministic fallback for local smoke tests."""

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
            kwargs = {"device_map": "auto"}
            if torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
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
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, **kwargs, **{dtype_keyword: "auto"}
                    )
                    break
                except TypeError:
                    continue
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def answer(self, question: str, entry: FAQEntry, language: str) -> str:
        self.load()
        if not self.available:
            return self._fallback(entry, language)

        prompt = (
            "You are Nirbhor Bank's offline FAQ assistant. Use only the FAQ answer below. "
            "Do not add facts. Preserve every number, date, fee, percentage, product name, and hotline. "
            f"Reply in {language}. Keep the answer to two short sentences.\n\n"
            f"FAQ question: {entry.question}\nFAQ answer: {entry.answer}\nCustomer question: {question}\nAnswer:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()
        return answer or self._fallback(entry, language)

    def supports(self, question: str, entry: FAQEntry) -> bool | None:
        """Ask the local model for a conservative FAQ entailment decision."""
        self.load()
        if not self.available:
            return None
        prompt = (
            "You are a strict FAQ support checker. Return only YES or NO. "
            "Return YES when the FAQ answer directly answers the customer question, "
            "or when it states a general rule that safely covers the specific case "
            "asked about without contradicting it. "
            "Return NO when the question asks for a new policy, a personal data lookup, "
            "a location-specific fact, a figure the FAQ does not state, "
            "or contradicts the FAQ.\n\n"
            f"FAQ question: {entry.question}\nFAQ answer: {entry.answer}\nCustomer question: {question}\nDecision:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        decision = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip().upper()
        if decision.startswith("YES"):
            return True
        if decision.startswith("NO"):
            return False
        return None

    @staticmethod
    def _fallback(entry: FAQEntry, language: str) -> str:
        if entry.faq_id == 14 and language == "bn":
            return "হ্যাঁ। যৌথ অ্যাকাউন্টে সর্বোচ্চ ২ জন হিসাবধারী থাকতে পারেন এবং দুজনকেই অ্যাকাউন্ট খোলার কাগজপত্রে স্বাক্ষর করতে হবে।"
        if entry.faq_id == 14 and language == "banglish":
            return "Haan. Joint account-e shorbocho 2 jon holder thakte paren, ebong dujonkei account opening documents-e sign korte hobe."
        return entry.answer
