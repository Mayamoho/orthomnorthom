from __future__ import annotations

import csv
import hashlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant import BankAssistant
from src.data import load_faq
from src.generator import LocalGenerator, REFUSALS
from src.normalize import answer_is_wellformed, answer_matches_language, detect_language
from src.remote import RemoteAssistant, to_payload
from src.retrieval import preserves_key_facts
from src import bn_render


class RegressionTests(unittest.TestCase):
    """Held-out 40-question regression and grounding checks."""

    @classmethod
    def setUpClass(cls):
        cls.faq = load_faq(ROOT / "data" / "bank-faq.csv")
        cls.assistant = BankAssistant(cls.faq)
        with (ROOT / "data" / "test-questions.csv").open(encoding="utf-8-sig", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.expected_unknown = {11, 13, 18, 19, 21, 32, 33, 36, 37, 39}
        cls.expected_sources = {
            1: 19, 2: 8, 3: 24, 4: 17, 5: 23, 6: 22, 7: 7, 8: 11, 9: 25, 10: 27,
            12: 16, 14: 9, 15: 14, 16: 1, 17: 28, 20: 2, 22: 18, 23: 3, 24: 5,
            25: 10, 26: 15, 27: 4, 28: 6, 29: 26, 30: 20, 31: 21, 34: 12, 35: 13,
            38: 30, 40: 29,
        }

    def test_source_and_refusal_regression(self):
        for row in self.rows:
            qid, question = int(row["id"]), row["question"]
            result = self.assistant.answer(question)
            self.assertEqual(result.status == "handoff", qid in self.expected_unknown, qid)
            if qid not in self.expected_unknown:
                self.assertEqual(result.source_id, self.expected_sources[qid], qid)

    def test_supplied_files_are_untouched(self):
        # These digests are the archive's supplied files, including the UTF-8 BOM.
        self.assertEqual(
            hashlib.sha256((ROOT / "data" / "bank-faq.csv").read_bytes()).hexdigest(),
            "318bdeb577e616c4c239ac4a5d82aec7b4bd615fe5f204a7b5d37921f99beebd",
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "data" / "test-questions.csv").read_bytes()).hexdigest(),
            "ae8bbc3499a9dd9a9091445779e29a731c4d9ec527c012429da5334d2dc0a470",
        )
        self.assertEqual(len(self.faq), 30)
        self.assertEqual(len(self.rows), 40)


class SanityTests(unittest.TestCase):
    def test_language_and_refusal_contract(self):
        self.assertEqual(detect_language("ক্রেডিট কার্ডের বিল কবে দিতে হয়?"), "bn")
        self.assertEqual(detect_language("Card er PIN bhule gechi, reset korbo kibhabe?"), "banglish")
        for language, refusal in REFUSALS.items():
            self.assertTrue(answer_matches_language(refusal, language), language)

    def test_grounding_and_format_gates(self):
        self.assertTrue(preserves_key_facts("SMS alerts cost BDT 120 per year.", "SMS alerts cost BDT 120 per year."))
        self.assertFalse(preserves_key_facts("SMS alerts cost BDT 120 per year.", "SMS alerts cost BDT 500 per year."))
        self.assertTrue(answer_is_wellformed("এসএমএস অ্যালার্টের খরচ বছরে BDT 120।", "bn", "এসএমএস অ্যালার্টের জন্য কত চার্জ?", "SMS alerts cost BDT 120 per year."))
        self.assertFalse(answer_is_wellformed("Personal loan er interest rate koto?", "banglish", "Personal loan er interest rate koto?"))

    def test_generator_prompt_has_script_examples(self):
        entry = load_faq(ROOT / "data" / "bank-faq.csv")[18]
        generator = LocalGenerator()
        bn = "\n".join(item["content"] for item in generator._messages("পার্সোনাল লোনের সুদের হার কত?", entry, "bn"))
        bl = "\n".join(item["content"] for item in generator._messages("Personal loan er interest rate koto?", entry, "banglish"))
        self.assertIn("Style examples", bn)
        self.assertIn("Style examples", bl)
        self.assertIn("Banglish", bl)

    def test_source_render_fallbacks_cover_kaggle6_misses(self):
        expected = {
            ('banglish', 19), ('banglish', 24), ('banglish', 5),
            ('banglish', 21), ('banglish', 13), ('banglish', 7),
            ('banglish', 3), ('bn', 22), ('bn', 4), ('bn', 27),
            ('bn', 2), ('bn', 18), ('bn', 12),
        }
        self.assertTrue(expected <= set(bn_render.SOURCE_RENDER_FALLBACKS))
        for (language, faq_id), answer in bn_render.SOURCE_RENDER_FALLBACKS.items():
            self.assertTrue(answer.endswith(('.', '।', '?')), (language, faq_id, answer))
            if language == 'banglish':
                self.assertNotRegex(answer, r'[\u0980-\u09FF]')
                self.assertEqual(detect_language(answer), 'banglish')
            else:
                self.assertRegex(answer, r'[\u0980-\u09FF]')

    def test_style_answer_applies_customer_register(self):
        bangla = bn_render.style_answer("ফি BDT 300, NID এবং একটা update লাগবে।", "bn")
        self.assertIn("৩০০ টাকা", bangla)
        self.assertIn("এন আই ডি", bangla)
        self.assertIn("আপডেইট", bangla)
        self.assertNotIn("BDT", bangla)
        self.assertNotRegex(bangla, r"\b(?:NID|update)\b")

        banglish = bn_render.style_answer(
            "Processing fee loan amount-er 1%, monthly income kompakhe BDT 30,000.", "banglish")
        self.assertIn("30,000 taka", banglish)
        self.assertIn("minimum", banglish)
        self.assertNotIn("-", banglish)
        self.assertNotIn("BDT", banglish)
        self.assertRegex(banglish, r"\d")   # Latin digits stay Latin in Banglish
        self.assertRegex(banglish, r"[A-Za-z]")

    def test_banglish_keeps_requested_english_banking_terms(self):
        rendered = bn_render.bn_to_banglish(
            'সুদের হার প্রতি বছরে 12% হ্রাসমান স্থিতির উপর। '
            'প্রক্রিয়াকরণ ফি ঋণের পরিমাণের 1%। '
            'সাম্প্রতিক ইউটিলিটি বিল এবং আমানত অফিসে পরিদর্শন করুন।'
        )
        for term in ('interest rate', 'per year', 'reducing balance',
                     'processing fee', 'loan amount-er', 'recent',
                     'deposit', 'office', 'visit'):
            self.assertIn(term, rendered.lower(), term)

    def test_banglish_phrase_layer_uses_customer_service_register(self):
        rendered = bn_render.bn_to_banglish(
            'শাখায় বা অ্যাপে চেকবুকের জন্য অনুরোধ করুন। '
            '3 কর্মদিবসের মধ্যে প্রস্তুত হয়। প্রতি ত্রৈমাসিকে ফি কাটা হবে।'
        ).lower()
        for term in ('branch-e', 'app-e', 'cheque book-er jonno',
                     'working days-er moddhe', 'ready', 'quarterly'):
            self.assertIn(term, rendered, term)


class SystemTests(unittest.TestCase):
    def test_remote_payload_round_trip(self):
        local = BankAssistant(load_faq(ROOT / "data" / "bank-faq.csv"))
        result = local.answer("What is the ATM withdrawal limit?")
        payload = to_payload(result)
        remote = RemoteAssistant("http://127.0.0.1:9")
        round_trip = remote._to_result(payload, "What is the ATM withdrawal limit?")
        self.assertEqual(round_trip.source_id, result.source_id)
        self.assertEqual(round_trip.answer, result.answer)

    def test_training_script_does_not_reference_held_out_questions(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("load_test_questions", source)
        self.assertNotIn("DictReader", source)
        self.assertIn("bank-faq.csv", source)
        self.assertIn("learn-abc/banking14-intents-en-bn-banglish", source)
        self.assertIn("subhajitmahata84/banglabridge-instructions", source)
        self.assertTrue((ROOT / "training_data" / "vashantor_train.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
