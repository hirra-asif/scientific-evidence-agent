import asyncio
import hashlib
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from experiments.agent_comparison import (
    CONDITIONS, PATCHES, adapt_agent, finish_row, recover_interrupted,
    run_key, schedule_runs, summarize, validate_checkpoint,
)

CLAIMS = {
    1: {"id": 1, "claim": "positive", "evidence": {
        "paper": [{"label": "SUPPORT", "sentences": [0]}],
    }},
    2: {"id": 2, "claim": "NEI", "evidence": {}},
}


def report(positive=False):
    return {"assessments": [{
        "document_id": "paper", "verdict": "SUPPORT" if positive else "NOT_ENOUGH_INFO",
        "evidence_sentence_ids": [0] if positive else [], "explanation": "test",
    }]}


def row(job, positive=False, status="ok"):
    result = {**job, "elapsed_seconds": 1.0, "trace_complete": True,
              "metadata": {"estimated_cost_usd": None, "opened_document_ids": ["paper"],
                           "tool_trace": [{"tool": "search_abstracts"}]}}
    finish_row(result, CLAIMS[job["claim_id"]], status, report(positive))
    return result


class TestAgentComparison(unittest.TestCase):
    def test_schedule_balances_order_and_keeps_repetitions_distinct(self):
        jobs = schedule_runs([1, 2], 2)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({run_key(job) for job in jobs}), 8)
        self.assertEqual([job["condition"] for job in jobs[:2]], list(CONDITIONS))
        self.assertEqual([job["condition"] for job in jobs[4:6]], list(CONDITIONS[::-1]))

    def test_adapter_is_idempotent_and_rejects_other_edits(self):
        source = "\n".join(old for old, _ in PATCHES) + '\nmodel="frozen-model"\n'
        digest = hashlib.sha256(source.encode()).hexdigest()
        adapted = adapt_agent(source, digest)
        self.assertEqual(adapt_agent(adapted, digest), adapted)
        self.assertIn('model="frozen-model"', adapted)
        with self.assertRaises(ValueError):
            adapt_agent(adapted.replace("frozen-model", "different-model"), digest)

    def test_failed_nei_is_not_an_abstention(self):
        job = {"claim_id": 2, "condition": CONDITIONS[0], "repetition": 1}
        failed = row(job, status="error")
        self.assertEqual(failed["score_counts"]["correct_nei"], 0)
        self.assertEqual(failed["nei_outcome"], "processing_failure")
        self.assertFalse(failed["abstained"])

    def test_all_attempts_retain_failed_gold_and_success_pairs_are_separate(self):
        jobs = schedule_runs([1, 2], 1)
        rows = [row(job, positive=job["claim_id"] == 1,
                    status="error" if job["claim_id"] == 1 and job["condition"] == CONDITIONS[0] else "ok")
                for job in jobs]
        summary = summarize(rows, jobs)
        self.assertEqual(summary["successful_pairs"], 1)
        baseline = summary["conditions"][CONDITIONS[0]]
        self.assertEqual(baseline["failures"], 1)
        self.assertEqual(baseline["metrics_all_attempts"]["counts"]["gold_documents"], 1)
        self.assertEqual(baseline["metrics_all_attempts"]["document_labels"]["f1"], 0.0)
        self.assertEqual(baseline["metrics_on_successful_pairs"]["counts"]["gold_documents"], 0)

    def test_false_flags_can_block_an_f1_gain(self):
        jobs = schedule_runs([1, 2], 1)
        rows = [row(job, positive=job["condition"] == CONDITIONS[1]) for job in jobs]
        decision = summarize(rows, jobs)["training_decision"]
        self.assertTrue(decision["checks"]["higher_evidence_f1"])
        self.assertFalse(decision["keep_hybrid_for_combination"])
        rows = [row(job, positive=job["condition"] == CONDITIONS[1] and job["claim_id"] == 1)
                for job in jobs]
        self.assertTrue(summarize(rows, jobs)["training_decision"]["keep_hybrid_for_combination"])

    def test_partial_checkpoint_has_no_adoption_decision(self):
        jobs = schedule_runs([1, 2], 1)
        rows = [row(job) for job in jobs[:2]]
        decision = summarize(rows, jobs)["training_decision"]
        self.assertFalse(decision["ready"])
        self.assertIsNone(decision["keep_hybrid_for_combination"])

    def test_resume_preserves_completed_runs_and_marks_unfinished_nei(self):
        jobs = schedule_runs([2], 1)
        rows = [row(jobs[0]), {**jobs[1], "status": "running", "metadata": {}}]
        validate_checkpoint(rows, jobs)
        recover_interrupted(rows, CLAIMS)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[1]["status"], "interrupted")
        self.assertEqual(rows[1]["nei_outcome"], "processing_failure")
        self.assertFalse(rows[1]["trace_complete"])
        self.assertEqual(len(rows), 2)

    def test_checkpoint_rejects_duplicate_or_out_of_order_attempts(self):
        jobs = schedule_runs([1], 1)
        with self.assertRaises(ValueError):
            validate_checkpoint([row(jobs[1])], jobs)
        with self.assertRaises(ValueError):
            validate_checkpoint([row(jobs[0]), row(jobs[0])], jobs)


class TestInjectedAgentSearch(unittest.IsolatedAsyncioTestCase):
    async def test_default_search_still_uses_bm25(self):
        import agent
        search = Mock(return_value=[])
        with patch.object(agent, "search_abstracts", search), redirect_stdout(io.StringIO()):
            state = agent.RunState()
            await state.search({"query": "test query"})
        search.assert_called_once_with("test query")

    async def test_overlapping_main_calls_route_their_own_searches(self):
        import agent

        def server(name, tools):
            return {"type": "sdk", "name": name, "instance": {item.name: item for item in tools}}

        def document(document_id):
            return {"document_id": document_id, "title": "Test",
                    "sentences": [{"sentence_id": 0, "text": "Test evidence"}]}

        async def query(prompt, options):
            tools = options.mcp_servers["scifact"]["instance"]
            await tools["search_abstracts"].handler({"query": prompt})
            await asyncio.sleep(0)
            await tools["read_abstract"].handler({"document_id": "paper"})
            yield agent.ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                num_turns=1, session_id="test", total_cost_usd=0.0,
                structured_output=report(True),
            )

        first, second = Mock(return_value=[]), Mock(return_value=[])
        with redirect_stdout(io.StringIO()), patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}):
            with patch.object(agent, "create_sdk_mcp_server", server), patch.object(agent, "query", query):
                with patch.object(agent, "read_abstract", document):
                    with patch.object(agent, "search_abstracts", side_effect=AssertionError("Wrong retriever")):
                        await asyncio.gather(
                            agent.main("first", use_skill=False, search_fn=first),
                            agent.main("second", use_skill=False, search_fn=second),
                        )
        first.assert_called_once_with("first")
        second.assert_called_once_with("second")


if __name__ == "__main__":
    unittest.main()
