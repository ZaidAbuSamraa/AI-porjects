import json
import tempfile
import unittest
from pathlib import Path
from engine import run_workflow

ITEM = {"source_id":"L1", "title":"Greetings", "category":"Conversations",
        "level":"A1", "language":"en", "body":"Hello!"}

class WorkflowTests(unittest.TestCase):
    def test_approval_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            result = run_workflow([ITEM], store=store)
            self.assertEqual(result["items"][0]["status"], "review_pending")
            self.assertEqual(result["mode"], "review")
            self.assertFalse(store.exists())

    def test_blank_reviewer_is_not_approval(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            result = run_workflow([ITEM], approved_by="   ", store=store)
            self.assertEqual(result["items"][0]["status"], "review_pending")
            self.assertEqual(result["mode"], "review")
            self.assertFalse(store.exists())

    def test_idempotent_and_draft_only(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            first = run_workflow([ITEM], approved_by="Trainer", store=store)
            second = run_workflow([ITEM], approved_by="Trainer", store=store)
            self.assertEqual(first["items"][0]["status"], "draft_saved")
            self.assertEqual(second["items"][0]["status"], "skipped_duplicate")
            self.assertIn('"status": "draft"', store.read_text())
            self.assertEqual(len(json.loads(store.read_text())), 1)

    def test_invalid_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "db.json"
            bad = dict(ITEM, title="")
            result = run_workflow([bad], approved_by="Trainer", store=store)
            self.assertEqual(result["items"][0]["status"], "failed_validation")
            self.assertIn("title", result["items"][0]["error"])
            self.assertEqual(json.loads(store.read_text()), [])

    def test_invalid_cefr_level_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "db.json"
            result = run_workflow([dict(ITEM, level="A9")], approved_by="Trainer", store=store)
            self.assertEqual(result["items"][0]["status"], "failed_validation")
            self.assertEqual(result["items"][0]["error"], "Invalid CEFR level")
            self.assertEqual(json.loads(store.read_text()), [])

    def test_invalid_item_does_not_block_valid_items(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "db.json"
            bad = dict(ITEM, source_id="L2", title="")
            result = run_workflow([bad, ITEM], approved_by="Trainer", store=store)
            self.assertEqual(result["summary"], {"draft_saved": 1, "failed_validation": 1})
            self.assertEqual([x["source_id"] for x in json.loads(store.read_text())], ["L1"])

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            result = run_workflow([ITEM], approved_by="Trainer", dry_run=True, store=store)
            self.assertEqual(result["items"][0]["status"], "dry_run_preview")
            self.assertFalse(store.exists())

    def test_dry_run_leaves_existing_store_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            run_workflow([ITEM], approved_by="Trainer", store=store)
            before = store.read_bytes()
            new = dict(ITEM, source_id="L2", title="Farewells")
            result = run_workflow([new], approved_by="Trainer", dry_run=True, store=store)
            self.assertEqual(result["items"][0]["status"], "dry_run_preview")
            self.assertEqual(store.read_bytes(), before)

    def test_word_count_in_report_and_draft(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            item = dict(ITEM, body="  Hello!  My name   is Sara.  ")
            result = run_workflow([item], approved_by="Trainer", store=store)
            self.assertEqual(result["items"][0]["word_count"], 5)
            self.assertIn("count_words", result["items"][0]["steps"])
            self.assertEqual(json.loads(store.read_text())[0]["word_count"], 5)

    def test_word_count_shown_in_dry_run_preview(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_workflow([ITEM], approved_by="Trainer", dry_run=True,
                                  store=Path(td) / "store.json")
            self.assertEqual(result["items"][0]["word_count"], 1)

if __name__ == "__main__":
    unittest.main()
