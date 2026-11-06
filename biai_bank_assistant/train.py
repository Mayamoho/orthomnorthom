"""Train the optional local-language adapter used by NB Assist.

The script deliberately has a narrow data contract: the supplied FAQ is used as
the factual domain corpus, while public licensed language corpora supply
Bangla/Banglish language supervision.  The held-out ``test-questions.csv`` is
never opened.  The resulting LoRA adapter is loaded by ``LocalGenerator`` when
``BIAI_ADAPTER_PATH`` points at its directory.

This is intended for a Kaggle T4 (the same command is used by the notebook):

    python train.py --faq data/bank-faq.csv --output /kaggle/working/nb-adapter
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
BANGlish_MARKERS = frozenset(
    "amar apnar apnader koto kibhabe kivabe korte korbo korini janabo lage lagbe dorkar hobe hoy kobe ache ase pabo paben pete dite pari parben kon koy kothay jonno theke er din taka tulate harie bhule nomini joggo joggota shomoy shakha gulo tulate sud".split()
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--max-banking-intents", type=int, default=1200)
    parser.add_argument("--max-banglabridge", type=int, default=200)
    parser.add_argument(
        "--vashantor",
        type=Path,
        default=Path(__file__).resolve().parent / "training_data" / "vashantor_train.jsonl",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    return parser.parse_args()


def _parallel_pairs(max_pairs: int) -> list[tuple[str, str, str]]:
    """Return (source, target, direction) examples from the public corpus."""
    from datasets import load_dataset

    try:
        kwargs = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
        stream = load_dataset(
            "istiaqfuad/bangla-english-banglish-pairs", split="train", streaming=True, **kwargs
        )
    except Exception as exc:
        print(f"warning: skipping public parallel corpus ({type(exc).__name__}: {exc})")
        return []
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in stream:
        anchor, positive = (row.get("anchor") or "").strip(), (row.get("positive") or "").strip()
        if not anchor or not positive:
            continue
        a_bn, p_bn = bool(BENGALI_RE.search(anchor)), bool(BENGALI_RE.search(positive))
        if a_bn == p_bn:
            continue
        bangla, latin = (anchor, positive) if a_bn else (positive, anchor)
        if not (6 <= len(bangla) <= 240 and 8 <= len(latin) <= 240):
            continue
        # Keep the filter domain-oriented without using the held-out questions.
        is_domain = bool(re.search(
            r"account|bank|loan|card|balance|interest|payment|transfer|deposit|branch|fee|charge|limit|withdraw|credit|debit|statement|cheque|pin|transaction|money|taka",
            latin,
            re.IGNORECASE,
        ))
        is_banglish = any(token in BANGlish_MARKERS for token in re.findall(r"[A-Za-z]+", latin.lower()))
        if not is_domain and not is_banglish:
            continue
        if not is_banglish:
            # Generic English banking pairs.
            directions = ((latin, bangla, "english_to_bangla"), (bangla, latin, "bangla_to_english"))
        else:
            # Romanised Bangla pairs are valuable for the register the base
            # model misses. They are labelled separately so the adapter does not
            # learn to call Banglish English.
            directions = ((bangla, latin, "bangla_to_banglish"), (latin, bangla, "banglish_to_bangla"))
        if not directions:
            continue
        for source, target, direction in directions:
            key = (source, target, direction)
            if key not in seen:
                seen.add(key)
                rows.append(key)
        if len(rows) >= max_pairs * 2:
            break
    return rows


def _local_vashantor(path: Path, max_rows: int = 500) -> list[tuple[str, str, str]]:
    """Load only the local archive's Train-derived, language-only pairs."""
    if not path.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            bn, bl, en = item.get("bangla", ""), item.get("banglish", ""), item.get("english", "")
            if not (bn and bl and en):
                continue
            rows.extend(((en, bn, "english_to_bangla"), (bn, en, "bangla_to_english"),
                         (bn, bl, "bangla_to_banglish"), (bl, bn, "banglish_to_bangla")))
            if len(rows) >= max_rows * 4:
                break
    return rows


def _public_style_examples(max_banking: int, max_banglabridge: int) -> list[tuple[str, str, str]]:
    """Use synthetic, licensed language examples without importing their facts."""
    from datasets import load_dataset

    rows: list[tuple[str, str, str]] = []
    token = os.environ.get("HF_TOKEN")
    datasets_to_try = [
        ("learn-abc/banking14-intents-en-bn-banglish", max_banking, "banking_register"),
        ("subhajitmahata84/banglabridge-instructions", max_banglabridge, "banglish_instruction"),
    ]
    for dataset_id, limit, direction in datasets_to_try:
        try:
            kwargs = {"token": token} if token else {}
            stream = load_dataset(dataset_id, split="train", streaming=True, **kwargs)
            used = 0
            for item in stream:
                if direction == "banking_register":
                    source = target = (item.get("text") or "").strip()
                else:
                    source = (item.get("instruction") or "").strip()
                    target = (item.get("output") or "").strip()
                if source and target:
                    rows.append((source, target, direction))
                    used += 1
                if used >= limit:
                    break
        except Exception as exc:
            print(f"warning: skipping {dataset_id} ({type(exc).__name__}: {exc})")
    return rows


def _faq_examples(faq_path: Path) -> list[tuple[str, str, str]]:
    from src.data import load_faq

    examples = []
    for entry in load_faq(faq_path):
        examples.append((
            f"FAQ answer: {entry.answer}\nRestate the FAQ answer in plain English.",
            entry.answer,
            "faq_grounding",
        ))
    return examples


def main() -> int:
    args = _args()
    # Imports are delayed so the repository remains runnable in lexical smoke
    # mode on a CPU-only laptop with only PySide6 installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    pairs = _parallel_pairs(args.max_pairs)
    local_pairs = _local_vashantor(args.vashantor)
    public_style = _public_style_examples(args.max_banking_intents, args.max_banglabridge)
    examples = _faq_examples(args.faq) + pairs + local_pairs + public_style
    if not (pairs or local_pairs or public_style):
        raise RuntimeError("No language-training examples were available")
    # A previous failed run can leave an empty output directory. Remove it so a
    # later notebook cell never mistakes that directory for a valid adapter.
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.float16, quantization_config=quant
    )
    base.config.use_cache = False
    model = get_peft_model(
        prepare_model_for_kbit_training(base),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    ignore = -100
    def encode(batch):
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for source, target, direction in zip(batch["source"], batch["target"], batch["direction"]):
            prompt = (
                f"Task: {direction.replace('_', ' ')}.\n"
                f"Source:\n{source}\n"
                "Target (answer only):\n"
            )
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"][:384]
            t = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"][:128]
            ids = p + t
            result["input_ids"].append(ids)
            result["attention_mask"].append([1] * len(ids))
            result["labels"].append([ignore] * len(p) + t)
        return result

    class Pad:
        def __call__(self, features):
            width = max(len(item["input_ids"]) for item in features)
            return {
                "input_ids": torch.tensor([x["input_ids"] + [tokenizer.pad_token_id] * (width - len(x["input_ids"])) for x in features]),
                "attention_mask": torch.tensor([x["attention_mask"] + [0] * (width - len(x["attention_mask"])) for x in features]),
                "labels": torch.tensor([x["labels"] + [ignore] * (width - len(x["labels"])) for x in features]),
            }

    dataset = Dataset.from_dict({
        "source": [row[0] for row in examples],
        "target": [row[1] for row in examples],
        "direction": [row[2] for row in examples],
    }).map(encode, batched=True, remove_columns=["source", "target", "direction"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=Pad(),
        processing_class=tokenizer,
    )
    trainer.train()
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(args.output)
    (args.output / "training_manifest.json").write_text(
        json.dumps({
            "base_model": args.model,
            "faq": str(args.faq),
            "public_dataset": "istiaqfuad/bangla-english-banglish-pairs (CC-BY-4.0)",
            "additional_public_datasets": [
                "learn-abc/banking14-intents-en-bn-banglish (MIT; train split)",
                "subhajitmahata84/banglabridge-instructions (CC-BY-4.0; train split)",
                "Vashantor train split (CC-BY-4.0; local archive)",
            ],
            "parallel_examples": len(pairs),
            "local_vashantor_examples": len(local_pairs),
            "public_style_examples": len(public_style),
            "faq_examples": len(_faq_examples(args.faq)),
            "held_out_test_questions_used": False,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"saved adapter to {args.output} ({len(examples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
