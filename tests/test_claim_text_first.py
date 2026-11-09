"""Offline checks for the controlled prompt comparison; no model requests."""

import asyncio
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments import agent_comparison as comparison
from experiments import compare_claim_text_first as runner

CLAIMS = {
    1: {"id": 1, "claim": "The treatment helps.",
        "evidence": {"paper": [{"label": "SUPPORT", "sentences": [0]}]}},
    2: {"id": 2, "claim": "There is insufficient evidence.", "evidence": {}},
}
SKILL = "---\nname: scientific-evidence\n---\n\nFor each abstract:\nKeep the evidence rules.\n$ARGUMENTS\n"


def report(positive=False):
    return {"assessments": [{"document_id": "paper", "verdict": (
        "SUPPORT" if positive else "NOT_ENOUGH_INFO"),
        "evidence_sentence_ids": [0] if positive else [], "explanation": "test"}]}


def completed_row(job, positive=False, trace=None):
    row = {**job, "claim": CLAIMS[job["claim_id"]]["claim"], "elapsed_seconds": 1,
           "trace_complete": True, "metadata": {"estimated_cost_usd": None,
           "tool_trace": trace or [], "opened_document_ids": ["paper"]}}
    comparison.finish_row(row, CLAIMS[job["claim_id"]], "ok", report(positive))
    return row


class TestClaimTextFirst(unittest.TestCase):
    def test_instruction_preserves_every_existing_guideline_and_claim_placeholder(self):
        result = runner.candidate_skill(SKILL)
        self.assertEqual(result.replace(runner.INSTRUCTION, "", 1), SKILL)
        self.assertEqual(result.count("$ARGUMENTS"), 1)
        self.assertTrue(result.index(runner.INSTRUCTION) < result.index("For each abstract:"))

    def test_rejects_changed_layout_or_duplicate_instruction(self):
        for text in ("missing layout", SKILL + SKILL, runner.candidate_skill(SKILL)):
            with self.assertRaises(ValueError):
                runner.candidate_skill(text)

    def test_helper_adapter_is_idempotent_and_rejects_unrelated_edits(self):
        source = (runner.PROJECT_DIR / runner.HELPER).read_text(encoding="utf-8")
        frozen = "a3a021560a8c62b6c8b0568f360c7ace719e22f419131356ab1840fd61698e00"
        adapted = runner.adapt_helper(source, frozen)
        self.assertEqual(runner.adapt_helper(adapted, frozen), adapted)
        with self.assertRaises(ValueError):
            runner.adapt_helper(adapted + "\n# unrelated change\n", frozen)

    def test_schedule_keeps_pairing_and_reverses_order_on_repeat_two(self):
        jobs = comparison.schedule_runs([1, 2], 2, conditions=runner.CONDITIONS)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({comparison.run_key(job) for job in jobs}), 8)
        self.assertEqual([job["condition"] for job in jobs[:2]], list(runner.CONDITIONS))
        self.assertEqual([job["condition"] for job in jobs[4:6]], list(runner.CONDITIONS[::-1]))

    def test_candidate_gain_and_false_flag_guard_use_the_new_labels(self):
        jobs = comparison.schedule_runs([1, 2], 1, conditions=runner.CONDITIONS)
        rows = [completed_row(job, positive=(job["claim_id"] == 1
                                and job["condition"] == runner.CONDITIONS[1])) for job in jobs]
        results = {"plan": {"schedule": jobs}, "runs": rows}
        decision = runner.summarize_results(results)["training_decision"]
        self.assertTrue(decision[runner.KEEP_KEY])
        self.assertNotIn("keep_hybrid_for_combination", decision)
        bad = next(row for row in rows if row["claim_id"] == 2
                   and row["condition"] == runner.CONDITIONS[1])
        comparison.finish_row(bad, CLAIMS[2], "ok", report(True))
        self.assertFalse(runner.summarize_results(results)["training_decision"][runner.KEEP_KEY])

    def test_compliance_needs_exact_claim_and_search_as_first_corpus_action(self):
        row = {"claim": CLAIMS[1]["claim"], "metadata": {"tool_trace": [
            {"tool": "search_abstracts", "arguments": {"query": CLAIMS[1]["claim"]}}]}}
        self.assertTrue(runner.first_query_behavior(row)["follows_first_query_instruction"])
        row["metadata"]["tool_trace"][0]["arguments"]["query"] = "treatment helps"
        self.assertFalse(runner.first_query_behavior(row)["follows_first_query_instruction"])
        row["metadata"]["tool_trace"][0]["arguments"]["query"] = CLAIMS[1]["claim"]
        row["metadata"]["tool_trace"].insert(0, {"tool": "read_abstract"})
        self.assertFalse(runner.first_query_behavior(row)["follows_first_query_instruction"])

    def test_no_search_is_unknown_not_successful_compliance(self):
        row = {"claim": "claim", "metadata": {"tool_trace": []}}
        self.assertIsNone(runner.first_query_behavior(row)["follows_first_query_instruction"])

    def test_noncompliance_keeps_valid_scores_and_incomplete_trace_is_separate(self):
        jobs = comparison.schedule_runs([1], 1, conditions=runner.CONDITIONS)
        rows = [completed_row(job, positive=True, trace=[
            {"tool": "search_abstracts", "arguments": {"query": "rewritten"}}]) for job in jobs]
        rows[0]["trace_complete"] = False
        summary = runner.summarize_results({"plan": {"schedule": jobs}, "runs": rows})
        candidate = summary["conditions"][runner.CONDITIONS[1]]
        self.assertEqual(candidate["metrics_all_attempts"]["evidence_sentences"]["f1"], 1)
        self.assertEqual(candidate["failures"], 0)
        self.assertEqual(candidate["first_query_behavior"]["runs_following_instruction"], 0)
        self.assertEqual(summary["conditions"][runner.CONDITIONS[0]]["first_query_behavior"]
                         ["runs_without_complete_trace"], 1)


class TestPromptRunner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {
            "sample": {"shared_agent_options": {"model": "test-model", "max_turns": 8,
                                                  "max_abstracts": 3}},
            "schedule": comparison.schedule_runs([1, 2], 2, conditions=runner.CONDITIONS),
            "source_sha256": {},
            "skill_profiles": {name: {"text": text, "sha256": runner.text_hash(text)}
                               for name, text in zip(runner.CONDITIONS,
                                                     (SKILL, runner.candidate_skill(SKILL)))},
            "execution": {"sdk_timeout_seconds_per_run": 1, "pause_after_consecutive_failures": 3},
        }
        comparison.save_json(self.root / runner.PLAN, self.plan)
        self.calls = []
        self.search = object()

        async def agent(**kwargs):
            text = (Path(kwargs["skill_workspace"]) /
                    ".claude/skills/scientific-evidence/SKILL.md").read_text()
            self.calls.append((kwargs["claim"], runner.INSTRUCTION in text))
            self.assertIs(kwargs["search_fn"], self.search)
            self.assertNotIn("gold_evidence", kwargs)
            metadata = kwargs["run_metadata"]
            metadata.update(self.plan["sample"]["shared_agent_options"])
            metadata.update(tool_trace=[{"tool": "search_abstracts", "arguments": {
                "query": kwargs["claim"]}}], opened_document_ids=["paper"])
            positive = kwargs["claim"] == CLAIMS[1]["claim"]
            return types.SimpleNamespace(model_dump=lambda: report(positive))

        self.agent = AsyncMock(side_effect=agent)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "PROJECT_DIR", self.root))
        self.stack.enter_context(patch.object(runner, "verify_plan"))
        self.stack.enter_context(patch.object(runner, "verify_files"))
        self.stack.enter_context(patch.object(runner, "load_claims", return_value=([], CLAIMS)))
        self.stack.enter_context(patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}))
        self.stack.enter_context(patch.dict(sys.modules, {
            "agent": types.SimpleNamespace(main=self.agent),
            "corpus_tools": types.SimpleNamespace(search_abstracts=self.search),
            "dotenv": types.SimpleNamespace(load_dotenv=lambda path: None),
        }))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def results(self):
        return comparison.read_json(self.root / runner.OUTPUT)

    async def test_pilot_resume_finish_and_finished_rerun_make_no_duplicate_calls(self):
        await runner.run(2)
        first = copy.deepcopy(self.results()["runs"])
        self.assertEqual(self.results()["status"], "paused")
        self.assertEqual([candidate for _, candidate in self.calls], [False, True])
        await runner.run(None)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(self.results()["runs"][:2], first)
        self.assertEqual(self.results()["status"], "finished")
        await runner.run(None)
        self.assertEqual(len(self.calls), 8)

    async def test_three_failures_pause_and_failed_nei_never_earns_abstention(self):
        self.agent.side_effect = RuntimeError("simulated SDK error")
        await runner.run(None)
        results = self.results()
        self.assertEqual(len(results["runs"]), 3)
        self.assertFalse(results["summary"]["complete"])
        failed_nei = results["runs"][2]
        self.assertEqual(failed_nei["nei_outcome"], "processing_failure")
        self.assertEqual(failed_nei["score_counts"]["correct_nei"], 0)

    async def test_timeout_is_saved_as_failure(self):
        self.agent.side_effect = asyncio.TimeoutError()
        await runner.run(1)
        self.assertEqual(self.results()["runs"][0]["status"], "timeout")

    async def test_saved_running_attempt_becomes_interruption_without_retry(self):
        await runner.run(1)
        results = self.results()
        results["runs"][0]["status"] = "running"
        comparison.save_json(self.root / runner.OUTPUT, results)
        self.calls.clear()
        await runner.run(1)
        self.assertEqual(self.results()["runs"][0]["status"], "interrupted")
        self.assertEqual(self.calls, [(CLAIMS[1]["claim"], True)])

    async def test_different_manifest_rejected_before_model_call(self):
        await runner.run(1)
        self.plan["changed"] = True
        comparison.save_json(self.root / runner.PLAN, self.plan)
        with self.assertRaises(ValueError):
            await runner.run(None)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
