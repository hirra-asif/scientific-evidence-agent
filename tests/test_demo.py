import asyncio
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import demo
from agent import AgentReport


class TestDemoData(unittest.TestCase):
    def test_recordings_and_documents_match_the_original_saved_results(self):
        bundle = demo.load_examples()
        path = demo.PROJECT_DIR / bundle["source_path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), bundle["source_sha256"])
        raw = json.loads(path.read_text())
        rows = {(r["claim_id"], r["condition"], r["repetition"]): r for r in raw["runs"]}
        docs = {d["document_id"]: d for r in raw["runs"] for d in r["metadata"].get("supplied_documents", [])}
        for example in bundle["examples"]:
            row = example["run"]
            self.assertEqual(row, rows[row["claim_id"], row["condition"], row["repetition"]])
            for doc in example["documents"]:
                self.assertEqual(doc, docs[doc["document_id"]])

    def test_recordings_never_invoke_the_agent_or_read_the_local_corpus(self):
        with patch.object(demo, "run_agent", side_effect=AssertionError("Unexpected API call")):
            with patch.object(demo, "read_abstract", side_effect=AssertionError("Unexpected data access")):
                for index in range(3):
                    self.assertEqual(demo.recorded_example(index)["mode"], "recorded")

    def test_quotes_come_from_corpus_sentences_and_failure_has_no_assessments(self):
        item = demo.recorded_example(0)["assessments"][0]
        original = demo.load_examples()["examples"][0]["documents"][0]
        for quote in item["quotes"]:
            self.assertEqual(quote["text"], original["sentences"][quote["sentence_id"]]["text"])
        failed = demo.recorded_example(2)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["assessments"], [])
        self.assertEqual(failed["abstracts_read"], 0)

    def test_fabricated_sentence_reference_is_rejected_before_display(self):
        example = copy.deepcopy(demo.load_examples()["examples"][0])
        example["run"]["report"]["assessments"][0]["evidence_sentence_ids"] = [9999]
        with self.assertRaises(ValueError):
            demo.present_run(example["run"], example["documents"], mode="recorded")


class TestLiveAdapter(unittest.TestCase):
    def test_live_run_calls_existing_agent_and_resolves_quotes(self):
        saved = demo.load_examples()["examples"][0]
        metadata = saved["run"]["metadata"]

        async def agent_call(**kwargs):
            kwargs["run_metadata"].update(copy.deepcopy(metadata))
            return AgentReport.model_validate(saved["run"]["report"])

        with patch.object(demo, "live_requirements", return_value=[]):
            with patch.object(demo, "run_agent", side_effect=agent_call) as agent:
                with patch.object(demo, "read_abstract", return_value=saved["documents"][0]):
                    result = demo.run_live(" Test scientific claim. ")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["claim"], "Test scientific claim.")
        self.assertTrue(result["assessments"][0]["quotes"])
        self.assertTrue(agent.call_args.kwargs["use_skill"])
        self.assertEqual(agent.call_args.kwargs["model"], demo.MODEL)

    def test_invalid_input_and_missing_setup_make_no_agent_calls(self):
        with patch.object(demo, "run_agent") as agent:
            for claim in (" ", "x" * 2001):
                with self.assertRaises(ValueError):
                    demo.run_live(claim)
            with patch.object(demo, "live_requirements", return_value=["Missing corpus"]):
                with self.assertRaisesRegex(ValueError, "Missing corpus"):
                    demo.run_live("valid claim")
            agent.assert_not_called()

    def test_error_is_not_an_abstention_and_redacts_the_key(self):
        with patch.object(demo, "live_requirements", return_value=[]):
            with patch.dict(demo.os.environ, {"ANTHROPIC_API_KEY": "test-secret"}):
                with patch.object(demo, "run_agent", side_effect=RuntimeError("test-secret rejected")):
                    result = demo.run_live("test claim")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["assessments"], [])
        self.assertNotIn("test-secret", result["error"])

    def test_timeout_is_reported_as_processing_failure(self):
        async def waiting(**kwargs):
            await asyncio.sleep(10)
        with patch.object(demo, "live_requirements", return_value=[]):
            with patch.object(demo, "run_agent", side_effect=waiting):
                result = demo.run_live("test claim", timeout=0.001)
        self.assertEqual(result["status"], "error")
        self.assertIn("time limit", result["error"])


class TestStreamlitApp(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(demo.PROJECT_DIR / "app.py"), default_timeout=10)

    def test_recorded_success_and_failure_render_without_credentials_or_corpus(self):
        with patch.object(demo, "live_requirements", return_value=["Setup needed"]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button[0].disabled)
            app.radio[0].set_value("Recorded examples").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Recorded run" in element.value for element in app.info))
            self.assertEqual(app.metric[1].value, "1 / 3")
            app.selectbox[0].set_value(2).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Processing failure" in element.value for element in app.error))

    def test_mode_change_clears_old_report_and_live_submission_runs_only_once(self):
        result = demo.recorded_example(0)
        result["mode"] = "live"
        with patch.object(demo, "live_requirements", return_value=[]):
            with patch.object(demo, "run_live", return_value=result) as run:
                app = self.app().run()
                app.text_area[0].input("A scientific claim.")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(run.call_count, 1)
                app.run()
                self.assertEqual(run.call_count, 1)
                app.radio[0].set_value("Recorded examples").run()
                self.assertEqual(len(app.metric), 0)
