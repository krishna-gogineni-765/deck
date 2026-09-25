import json
import unittest
from pathlib import Path

from deck.app import render
from deck.describe import apply_descriptions, description_prompt
from deck.label import plan_label
from deck.taxonomy import description_is_weak, load_taxonomy, save_taxonomy


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "support.json"


class TaxonomyTest(unittest.TestCase):
    def test_round_trip_and_weak_descriptions(self):
        taxonomy = load_taxonomy(EXAMPLE)
        self.assertEqual(taxonomy.id, "support")
        self.assertFalse(description_is_weak(taxonomy.get("track_order")))
        self.assertTrue(description_is_weak(taxonomy.get("billing_question")))

    def test_apply_teacher_descriptions(self):
        taxonomy = load_taxonomy(EXAMPLE)
        updated = apply_descriptions(
            taxonomy,
            {
                "classes": {
                    "billing_question": {
                        "description": "The person is asking about a charge, invoice, or refund.",
                        "exclusions": "Order tracking.",
                        "example": "Why was I charged twice?",
                    },
                    "unknown": {"description": "ignored"},
                }
            },
        )
        self.assertIn("charge", updated.get("billing_question").description)
        self.assertFalse(description_is_weak(updated.get("billing_question")))
        self.assertTrue(description_is_weak(updated.get("cancel_order")))

    def test_prompt_asks_only_for_weak_classes_when_any_are_weak(self):
        prompt = description_prompt(load_taxonomy(EXAMPLE))
        self.assertIn("billing_question", prompt)
        self.assertNotIn("track_order", prompt)

    def test_easy_example_uses_potion_and_hard_example_uses_teacher(self):
        taxonomy = load_taxonomy(EXAMPLE)
        easy = plan_label("Where is my order?", taxonomy)
        hard = plan_label(
            "The policy was amended. Unless the exception applies, "
            "calculate the deadline in business days after applying the later section. " * 40,
            taxonomy,
        )
        self.assertEqual(easy.model, "potion")
        self.assertEqual(easy.decision_model, "deck4b")
        self.assertEqual(hard.model, "teacher")

    def test_save_and_page(self, tmp_path=None):
        import tempfile

        taxonomy = load_taxonomy(EXAMPLE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.json"
            save_taxonomy(taxonomy, path)
            self.assertEqual(load_taxonomy(path), taxonomy)
        page = render(taxonomy, route_text="Where is my order?")
        self.assertIn("potion", page)
        self.assertIn("deck4b", page)
        self.assertIn("billing_question", page)

    def test_apply_rejects_bad_payload(self):
        with self.assertRaises(ValueError):
            apply_descriptions(load_taxonomy(EXAMPLE), json.loads("[]"))


if __name__ == "__main__":
    unittest.main()
