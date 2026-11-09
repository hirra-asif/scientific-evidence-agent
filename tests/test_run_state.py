import asyncio
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import agent


def fake_document(document_id):
    return {
        "document_id": str(document_id),
        "title": "Test abstract",
        "sentences": [{"sentence_id": 0, "text": "Example evidence."}],
    }


def fake_server(name, tools):
    # Retain the real decorated handlers without starting an MCP server.
    return {
        "type": "sdk",
        "name": name,
        "instance": {item.name: item for item in tools},
    }


def successful_result(document_id):
    return agent.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=f"test-{document_id}",
        total_cost_usd=0.0,
        structured_output={
            "assessments": [{
                "document_id": document_id,
                "verdict": "SUPPORT",
                "evidence_sentence_ids": [0],
                "explanation": "Test explanation.",
            }]
        },
    )


class TestRunState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.reader = self.enterContext(
            patch.object(agent, "read_abstract", side_effect=fake_document)
        )
        self.searcher = self.enterContext(
            patch.object(agent, "search_abstracts", return_value=[
                {"document_id": "paper", "title": "Test abstract", "score": 1.0}
            ])
        )
        self.enterContext(patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "test-key-not-a-real-key"}
        ))
        self.enterContext(patch.object(
            agent, "create_sdk_mcp_server", side_effect=fake_server
        ))
        self.enterContext(patch.object(
            agent, "query", side_effect=AssertionError("Unexpected SDK call")
        ))

    async def test_fourth_distinct_abstract_is_blocked(self):
        state = agent.RunState()
        for document_id in ("a", "b", "c"):
            await state.read({"document_id": document_id})

        result = await state.read({"document_id": "d"})

        self.assertTrue(result["is_error"])
        self.assertEqual(set(state.opened_documents), {"a", "b", "c"})
        self.assertEqual(self.reader.call_count, 3)
        self.assertEqual(state.tool_trace[-1]["status"], "limit_reached")

    async def test_cached_read_is_allowed_at_limit(self):
        state = agent.RunState()
        for document_id in ("a", "b", "c"):
            await state.read({"document_id": document_id})

        result = await state.read({"document_id": "a"})

        self.assertFalse(result.get("is_error", False))
        self.assertEqual(self.reader.call_count, 3)
        self.assertEqual(len(state.opened_documents), 3)
        self.assertEqual(state.tool_trace[-1]["status"], "cached")

    async def test_failed_read_does_not_use_a_slot(self):
        state = agent.RunState()
        self.reader.side_effect = KeyError("missing")
        with self.assertRaises(KeyError):
            await state.read({"document_id": "missing"})

        self.assertEqual(state.opened_documents, {})
        self.assertEqual(state.tool_trace[-1]["status"], "error")

        self.reader.side_effect = fake_document
        for document_id in ("a", "b", "c"):
            await state.read({"document_id": document_id})
        self.assertEqual(len(state.opened_documents), 3)

    async def test_search_logs_results_without_opening_documents(self):
        state = agent.RunState()
        await state.search({"query": "test topic"})

        self.searcher.assert_called_once_with("test topic")
        self.assertEqual(state.opened_documents, {})
        self.assertEqual(state.tool_trace[0]["status"], "ok")
        self.assertEqual(state.tool_trace[0]["results"][0]["document_id"], "paper")

    async def test_failed_search_is_logged(self):
        state = agent.RunState()
        self.searcher.side_effect = RuntimeError("Search unavailable")
        with self.assertRaisesRegex(RuntimeError, "Search unavailable"):
            await state.search({"query": "test topic"})

        self.assertEqual(state.opened_documents, {})
        self.assertEqual(state.tool_trace[0]["status"], "error")
        self.assertIn("Search unavailable", state.tool_trace[0]["error"])

    async def test_each_state_has_its_own_read_limit(self):
        first = agent.RunState()
        for document_id in ("a", "b", "c"):
            await first.read({"document_id": document_id})

        second = agent.RunState()
        result = await second.read({"document_id": "d"})

        self.assertFalse(result.get("is_error", False))
        self.assertEqual(set(first.opened_documents), {"a", "b", "c"})
        self.assertEqual(set(second.opened_documents), {"d"})
        self.assertEqual(len(first.tool_trace), 3)
        self.assertEqual(len(second.tool_trace), 1)

    async def test_overlapping_main_calls_keep_state_separate(self):
        async def fake_query(prompt, options):
            tools = options.mcp_servers["scifact"]["instance"]
            await tools["read_abstract"].handler({"document_id": prompt})
            # Let the other run read its paper before either report is validated.
            await asyncio.sleep(0)
            yield successful_result(prompt)

        first_metadata, second_metadata = {}, {}
        with patch.object(agent, "query", fake_query):
            first, second = await asyncio.gather(
                agent.main("first", use_skill=False, run_metadata=first_metadata),
                agent.main("second", use_skill=False, run_metadata=second_metadata),
            )

        self.assertEqual(first.assessments[0].document_id, "first")
        self.assertEqual(second.assessments[0].document_id, "second")
        self.assertEqual(first_metadata["opened_document_ids"], ["first"])
        self.assertEqual(second_metadata["opened_document_ids"], ["second"])
        self.assertIsNot(first_metadata["tool_trace"], second_metadata["tool_trace"])
        self.assertEqual(len(first_metadata["tool_trace"]), 1)
        self.assertEqual(len(second_metadata["tool_trace"]), 1)
        self.assertEqual(first_metadata["tool_trace"][0]["document_id"], "first")
        self.assertEqual(second_metadata["tool_trace"][0]["document_id"], "second")

    async def test_sdk_failure_preserves_documents_and_trace(self):
        async def fake_query(prompt, options):
            yield agent.SystemMessage(subtype="init", data={"skills": []})
            tools = options.mcp_servers["scifact"]["instance"]
            await tools["read_abstract"].handler({"document_id": "paper"})
            raise RuntimeError("Simulated SDK failure")

        metadata = {}
        with patch.object(agent, "query", fake_query):
            with self.assertRaisesRegex(RuntimeError, "Simulated SDK failure"):
                await agent.main("test claim", use_skill=False, run_metadata=metadata)

        self.assertEqual(metadata["opened_document_ids"], ["paper"])
        self.assertEqual(len(metadata["tool_trace"]), 1)
        self.assertEqual(metadata["tool_trace"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()