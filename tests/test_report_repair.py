"""Policy and repair tests use controlled tool results, never paid model calls."""

import asyncio
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

import agent


def document(document_id):
    return {
        "document_id": str(document_id),
        "title": "Test abstract",
        "sentences": [{"sentence_id": 0, "text": "Evidence from the corpus."}],
    }


def report(document_id="a", sentence_id=0):
    return {"assessments": [{
        "document_id": document_id, "verdict": "SUPPORT",
        "evidence_sentence_ids": [sentence_id], "explanation": "Test explanation.",
    }]}


def payload(result):
    return json.loads(result["content"][0]["text"])


def server_stub(name, tools):
    return {"type": "sdk", "name": name, "instance": {t.name: t for t in tools}}


def terminal_result(**kwargs):
    return agent.ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
        num_turns=5, session_id="test", total_cost_usd=0.0, **kwargs,
    )


class TestReportRepair(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.reader = self.enterContext(patch.object(agent, "read_abstract", side_effect=document))
        self.enterContext(patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}))
        self.enterContext(patch.object(agent, "create_sdk_mcp_server", side_effect=server_stub))
        self.enterContext(patch.object(agent, "query", side_effect=AssertionError("Unexpected SDK call")))
        self.state = agent.RepairRunState()

    async def before(self, document_id, call_id=None):
        return await self.state.before_read(
            {"tool_input": {"document_id": document_id}}, call_id or document_id, {},
        )

    async def after(self, call_id):
        await self.state.after_read({}, call_id, {})

    async def test_hook_reserves_slots_before_parallel_reads_execute(self):
        decisions = await asyncio.gather(*(self.before(doc) for doc in ("a", "b", "c", "d")))
        self.assertEqual(decisions[:3], [{}, {}, {}])
        self.assertEqual(decisions[3]["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(set(self.state.pending_reads.values()), {"a", "b", "c"})
        self.reader.assert_not_called()

    async def test_cached_and_duplicate_pending_reads_do_not_cost_extra_slots(self):
        for doc in ("a", "b", "c"):
            await self.before(doc)
        self.assertEqual(await self.before("a", "a-again"), {})
        for doc in ("a", "b", "c"):
            await self.state.read({"document_id": doc})
            await self.after(doc)
        await self.after("a-again")
        self.assertEqual(await self.before("a", "cached"), {})
        await self.state.read({"document_id": "a"})
        self.assertEqual(self.reader.call_count, 3)
        self.assertEqual((await self.before("d"))["hookSpecificOutput"]["permissionDecision"], "deny")

    async def test_failed_read_releases_reservation(self):
        for doc in ("a", "b", "missing"):
            await self.before(doc)
        self.reader.side_effect = KeyError("missing")
        with self.assertRaises(KeyError):
            await self.state.read({"document_id": "missing"})
        failure_hook = self.state.sdk_hooks()["PostToolUseFailure"][0].hooks[0]
        await failure_hook({"tool_use_id": "missing"}, None, {})
        self.assertEqual(await self.before("c"), {})
        self.assertNotIn("missing", self.state.opened_documents)

    async def test_read_guard_still_works_without_hooks(self):
        for doc in ("a", "b", "c"):
            await self.state.read({"document_id": doc})
        result = await self.state.read({"document_id": "d"})
        self.assertTrue(result["is_error"])
        self.assertEqual(self.reader.call_count, 3)

    async def test_invalid_citation_can_be_corrected(self):
        await self.state.read({"document_id": "a"})
        rejected = await self.state.submit(report(sentence_id=99))
        self.assertTrue(rejected["is_error"])
        self.assertIn("Invalid sentence ID 99", payload(rejected)["error"])
        self.assertEqual(payload(rejected)["submissions_remaining"], 2)
        self.assertEqual(payload(rejected)["read_documents"], {"a": {"sentence_count": 1}})
        accepted = await self.state.submit(report())
        self.assertTrue(payload(accepted)["accepted"])
        self.assertEqual(self.state.final_report().assessments[0].evidence_sentence_ids, [0])
        self.assertEqual([e["status"] for e in self.state.submission_trace], ["rejected", "accepted"])
        self.assertEqual(self.state.submission_trace[0]["report"], report(sentence_id=99))

    async def test_no_reads_can_be_repaired_without_fabricating_evidence(self):
        result = await self.state.submit({"assessments": []})
        self.assertIn("did not read", payload(result)["error"])
        await self.state.read({"document_id": "a"})
        self.assertTrue(payload(await self.state.submit(report()))["accepted"])

    async def test_schema_error_is_returned_by_handler(self):
        result = await self.state.submit({"assessments": "not a list"})
        self.assertTrue(result["is_error"])
        self.assertIsNone(self.state.accepted_report)

    async def test_third_rejection_is_terminal_even_if_later_report_is_valid(self):
        await self.state.read({"document_id": "a"})
        for _ in range(3):
            rejected = await self.state.submit(report(sentence_id=99))
        self.assertEqual(payload(rejected)["submissions_remaining"], 0)
        result = await self.state.submit(report())
        self.assertFalse(payload(result)["accepted"])
        self.assertEqual(len(self.state.submission_trace), 3)
        with self.assertRaisesRegex(RuntimeError, "without an accepted"):
            self.state.final_report()

    async def test_pending_read_prevents_acceptance(self):
        await self.state.read({"document_id": "a"})
        await self.before("b")
        self.assertIn("pending reads", payload(await self.state.submit(report()))["error"])

    async def test_accepted_report_is_immutable_and_closes_reads(self):
        await self.state.read({"document_id": "a"})
        submitted = report()
        await self.state.submit(submitted)
        submitted["assessments"][0]["evidence_sentence_ids"] = [99]
        returned = self.state.final_report()
        returned.assessments.clear()
        await self.state.submit(report("unread"))
        self.assertEqual(self.state.final_report().model_dump(), report())
        self.assertEqual((await self.before("b"))["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertTrue((await self.state.read({"document_id": "b"}))["is_error"])
        self.assertEqual(set(self.state.opened_documents), {"a"})

    async def test_runs_have_independent_reservations_and_submissions(self):
        second = agent.RepairRunState()
        await self.before("a")
        await self.state.submit(report())
        self.assertEqual(second.pending_reads, {})
        self.assertEqual(second.submission_trace, [])
        self.assertIsNone(second.accepted_report)

    async def test_main_uses_accepted_tool_report_not_final_model_text(self):
        async def fake_query(prompt, options):
            self.assertEqual(options.max_turns, 12)
            self.assertIsNone(options.output_format)
            self.assertIn("mcp__scifact__submit_report", options.allowed_tools)
            self.assertEqual(options.hooks["PreToolUse"][0].on_failure, "block")
            tools = options.mcp_servers["scifact"]["instance"]
            await options.hooks["PreToolUse"][0].hooks[0](
                {"tool_input": {"document_id": "a"}}, "read-a", {},
            )
            await tools["read_abstract"].handler({"document_id": "a"})
            await options.hooks["PostToolUse"][0].hooks[0]({}, "read-a", {})
            await tools["submit_report"].handler(report(sentence_id=99))
            await tools["submit_report"].handler(report())
            yield terminal_result(result="A different final claim", structured_output=report("unread"))

        metadata = {}
        with patch.object(agent, "query", fake_query):
            actual = await agent.main("claim", use_skill=False, report_repair=True, run_metadata=metadata)
        self.assertEqual(actual.model_dump(), report())
        self.assertTrue(metadata["report_repaired"])
        self.assertEqual(len(metadata["submission_trace"]), 2)
        self.assertEqual(metadata["opened_document_ids"], ["a"])

    async def test_main_cannot_bypass_submission_with_valid_final_json(self):
        async def fake_query(prompt, options):
            tools = options.mcp_servers["scifact"]["instance"]
            await tools["read_abstract"].handler({"document_id": "a"})
            yield terminal_result(structured_output=report())

        metadata = {}
        with patch.object(agent, "query", fake_query):
            with self.assertRaisesRegex(RuntimeError, "without an accepted"):
                await agent.main("claim", use_skill=False, report_repair=True, run_metadata=metadata)
        self.assertFalse(metadata["report_accepted"])

    async def test_sdk_failure_after_submission_still_fails_and_keeps_trace(self):
        async def fake_query(prompt, options):
            tools = options.mcp_servers["scifact"]["instance"]
            await tools["read_abstract"].handler({"document_id": "a"})
            await tools["submit_report"].handler(report())
            result = terminal_result()
            result.is_error = True
            result.subtype = "error_max_turns"
            yield result

        metadata = {}
        with patch.object(agent, "query", fake_query):
            with self.assertRaisesRegex(RuntimeError, "error_max_turns"):
                await agent.main("claim", use_skill=False, report_repair=True, run_metadata=metadata)
        self.assertTrue(metadata["report_accepted"])
        self.assertTrue(metadata["sdk_is_error"])
        self.assertEqual(len(metadata["submission_trace"]), 1)

    async def test_default_still_uses_original_tools_turn_budget_and_output_schema(self):
        async def fake_query(prompt, options):
            tools = options.mcp_servers["scifact"]["instance"]
            self.assertEqual(set(tools), {"search_abstracts", "read_abstract"})
            self.assertEqual(options.max_turns, 8)
            self.assertFalse(options.hooks)
            self.assertEqual(options.output_format["schema"], agent.AgentReport.model_json_schema())
            await tools["read_abstract"].handler({"document_id": "a"})
            yield terminal_result(structured_output=report())

        with patch.object(agent, "query", fake_query):
            self.assertEqual((await agent.main("claim", use_skill=False)).model_dump(), report())


class TestRepairMcpBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_server_returns_repair_errors_and_acceptance(self):
        """Exercise the real MCP schema and error conversion, without Claude."""
        state = agent.RepairRunState()
        server = agent.create_sdk_mcp_server(name="scifact", tools=state.sdk_tools())["instance"]
        with patch.object(agent, "read_abstract", side_effect=document), redirect_stdout(io.StringIO()):
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                async with anyio.create_task_group() as group:
                    group.start_soon(server.run, *server_streams, server.create_initialization_options())
                    async with ClientSession(*client_streams) as client:
                        await client.initialize()
                        await client.call_tool("read_abstract", {"document_id": "a"})
                        rejected = await client.call_tool("submit_report", report(sentence_id=99))
                        self.assertTrue(rejected.model_dump(by_alias=True)["isError"])
                        self.assertIn("Invalid sentence ID 99", rejected.content[0].text)
                        accepted = await client.call_tool("submit_report", report())
                        self.assertFalse(accepted.model_dump(by_alias=True)["isError"])
                        self.assertTrue(json.loads(accepted.content[0].text)["accepted"])
                    group.cancel_scope.cancel()
        self.assertEqual(state.final_report().model_dump(), report())


if __name__ == "__main__":
    unittest.main()
