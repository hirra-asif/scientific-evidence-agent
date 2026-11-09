"""Exercise pilot routing and checkpoint recovery without API requests."""

import copy
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments import agent_comparison as comparison
from experiments import final_conditions as conditions
from experiments import run_final_pilot as runner

CLAIMS = {1: {"id": 1, "claim": "First test claim", "evidence": {}},
          2: {"id": 2, "claim": "Second test claim", "evidence": {}}}


class TestFinalPilotRunner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cache = self.root / "embedding-cache-test"
        cache.write_bytes(b"test cache")
        skill = "Frozen test Skill"
        self.plan = {
            "role": "Technical pilot", "sample": {"skill_text": skill, "skill_sha256": conditions.digest(skill)},
            "schedule": conditions.schedule_runs([1, 2]), "conditions": conditions.configurations(0.5),
            "assessment_guidelines": "Same rules", "system_prompts": {"pipeline": "Pipeline contract"},
            "source_sha256": {},
            "execution": {"timeout_seconds_per_attempt": 1, "pause_after_consecutive_failures": 3},
        }
        comparison.save_json(self.root / runner.PLAN, self.plan)
        self.calls = []
        self.search_functions = []
        self.bm25 = lambda query: []
        self.hybrid = types.SimpleNamespace(cache_path=cache, embed_queries=lambda queries: None,
                                            search=lambda query, dense_weight: [])
        async def agent(**kwargs):
            self.calls.append(("agent", kwargs["model"]))
            self.search_functions.append(kwargs["search_fn"])
            path = Path(kwargs["skill_workspace"]) / ".claude/skills/scientific-evidence/SKILL.md"
            self.assertEqual(path.read_text(), skill)
            self.assertNotIn("gold_evidence", kwargs)
            return finish(kwargs["run_metadata"], kwargs["model"], 8)
        async def pipeline(claim, **kwargs):
            self.calls.append(("pipeline", kwargs["config"]["model"]))
            self.search_functions.append(kwargs["search_fn"])
            self.assertFalse((Path(kwargs["workspace"]) / ".claude").exists())
            self.assertEqual(kwargs["guidelines"], "Same rules")
            self.assertNotIn("gold_evidence", kwargs)
            return finish(kwargs["run_metadata"], kwargs["config"]["model"], 2)
        def finish(metadata, model, turns):
            metadata.update(model=model, sdk_model=model, max_turns=turns, max_abstracts=3,
                            api_timeout_ms=120000, max_api_retries=2)
            return types.SimpleNamespace(model_dump=lambda: {"assessments": [{
                "document_id": "paper", "verdict": "NOT_ENOUGH_INFO",
                "evidence_sentence_ids": [], "explanation": "Test"}]})
        self.agent = AsyncMock(side_effect=agent)
        self.pipeline = AsyncMock(side_effect=pipeline)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(runner, "PROJECT_DIR", self.root))
        stack.enter_context(patch.object(runner, "verify_plan"))
        stack.enter_context(patch.object(runner, "verify_files"))
        stack.enter_context(patch.object(runner, "load_claims", return_value=([], CLAIMS)))
        stack.enter_context(patch.object(runner, "run_pipeline", self.pipeline))
        stack.enter_context(patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}))
        stack.enter_context(patch.dict(sys.modules, {
            "agent": types.SimpleNamespace(main=self.agent),
            "corpus_tools": types.SimpleNamespace(search_abstracts=self.bm25),
            "hybrid_retrieval": types.SimpleNamespace(HybridRetriever=lambda: self.hybrid),
            "dotenv": types.SimpleNamespace(load_dotenv=lambda path: None),
        }))
        stack.enter_context(redirect_stdout(io.StringIO()))

    def results(self):
        return comparison.read_json(self.root / runner.OUTPUT)

    async def test_routes_four_conditions_then_resumes_without_duplicate_attempts(self):
        await runner.run(4)
        saved = copy.deepcopy(self.results()["runs"])
        self.assertEqual(self.calls, [("agent", conditions.HAIKU), ("agent", conditions.HAIKU),
                                     ("agent", conditions.SONNET), ("pipeline", conditions.HAIKU)])
        self.assertIs(self.search_functions[0], self.bm25)
        self.assertIs(self.search_functions[1], self.search_functions[2])
        self.assertIs(self.search_functions[1], self.search_functions[3])
        await runner.run()
        self.assertEqual(self.results()["runs"][:4], saved)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(self.results()["status"], "finished")
        await runner.run()
        self.assertEqual(len(self.calls), 8)

    async def test_failures_are_preserved_and_pause_after_three(self):
        self.agent.side_effect = RuntimeError("Test SDK failure")
        await runner.run()
        results = self.results()
        self.assertEqual(len(results["runs"]), 3)
        self.assertEqual(results["status"], "paused")
        self.assertTrue(all(row["nei_outcome"] == "processing_failure" for row in results["runs"]))
        self.assertTrue(all(row["score_counts"]["correct_nei"] == 0 for row in results["runs"]))

    async def test_interrupted_attempt_is_kept_and_next_condition_runs(self):
        await runner.run(1)
        result = self.results()
        result["runs"][0]["status"] = "running"
        comparison.save_json(self.root / runner.OUTPUT, result)
        self.calls.clear()
        await runner.run(1)
        self.assertEqual(self.results()["runs"][0]["status"], "interrupted")
        self.assertEqual(self.results()["runs"][1]["condition"], "improved_haiku")
        self.assertEqual(len(self.calls), 1)

    async def test_changed_manifest_rejected_before_new_attempt(self):
        await runner.run(1)
        self.plan["unexpected"] = True
        comparison.save_json(self.root / runner.PLAN, self.plan)
        with self.assertRaises(ValueError):
            await runner.run()
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
