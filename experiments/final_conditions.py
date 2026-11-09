"""The four selected configurations and the fixed pipeline's SDK assessment."""

import ast
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from experiments.agent_comparison import TERMINAL
from scoring import summarize_scores

CONDITIONS = ("current_haiku", "improved_haiku", "improved_sonnet", "fixed_pipeline")
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
AGENT_PATCHES = (
    ("    search_fn=None,\n):\n    state = RunState(search_fn=search_fn)",
     '    search_fn=None,\n    model="claude-haiku-5-5",\n):\n    state = RunState(search_fn=search_fn)'),
    ('        model="claude-haiku-5-5",\n', '        model=model,\n'),
    ('                skill_discovered = (\n',
     '                run_metadata["sdk_model"] = message.data.get("model")\n'
     '                skill_discovered = (\n'),
    ('                "estimated_cost_usd": message.total_cost_usd,\n',
     '                "estimated_cost_usd": message.total_cost_usd,\n'
     '                "usage": getattr(message, "usage", None),\n'
     '                "model_usage": getattr(message, "model_usage", None),\n'
     '                "sdk_duration_ms": getattr(message, "duration_ms", None),\n'
     '                "api_duration_ms": getattr(message, "duration_api_ms", None),\n'),
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapt_agent(source, expected_hash):
    """Only add a model argument and logging; leave default behavior intact."""
    original = source
    if digest(original) != expected_hash:
        for old, new in reversed(AGENT_PATCHES):
            if original.count(new) != 1:
                raise ValueError("agent.py differs from the frozen source and expected adapter.")
            original = original.replace(new, old, 1)
        if digest(original) != expected_hash:
            raise ValueError("agent.py has unrelated edits; nothing was overwritten.")
    adapted = original
    for old, new in AGENT_PATCHES:
        if adapted.count(old) != 1:
            raise ValueError("Could not locate the expected agent code.")
        adapted = adapted.replace(old, new, 1)
    return adapted


def assessment_guidelines(skill_text):
    """Copy the existing assessment rules exactly, without search instructions."""
    start, end = "For each abstract:\n", "\nClaim to assess:\n$ARGUMENTS"
    if skill_text.count(start) != 1 or skill_text.count(end) != 1:
        raise ValueError("Unexpected Skill layout; cannot isolate the assessment rules.")
    return start + skill_text.split(start, 1)[1].split(end, 1)[0]


def agent_system_prompt(source):
    """Read the explicit prompt without importing the SDK or executing source."""
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "ClaudeAgentOptions"]
    if len(calls) != 1:
        raise ValueError("Expected one agent options configuration.")
    prompt = next(k.value for k in calls[0].keywords if k.arg == "system_prompt")
    if isinstance(prompt, ast.Constant) and isinstance(prompt.value, str):
        return prompt.value
    if not isinstance(prompt, ast.JoinedStr):
        raise ValueError("Unexpected system prompt representation.")
    parts = []
    for part in prompt.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        elif (isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
              and part.value.id == "MAX_ABSTRACTS" and part.format_spec is None):
            parts.append("3")
        else:
            raise ValueError("Unexpected dynamic system prompt content.")
    return "".join(parts)


def pipeline_system_prompt(agent_prompt):
    replacements = (
        ("First search for relevant papers, then read promising abstracts. "
         "You may refine your search if needed. "
         "For this demonstration, read at most 3 abstracts. ",
         "Assess the three abstracts supplied in the prompt. "),
        ("Return one assessment for each abstract you read. ",
         "Return one assessment for each supplied abstract. "),
        ("Base assessments only on abstracts returned by read_abstract. ",
         "Base assessments only on the supplied abstracts. "),
    )
    for old, new in replacements:
        if agent_prompt.count(old) != 1:
            raise ValueError("Unexpected agent prompt; pipeline contract needs review.")
        agent_prompt = agent_prompt.replace(old, new, 1)
    return agent_prompt


def configurations(dense_weight):
    return {
        "current_haiku": {"mode": "agent", "model": HAIKU, "retriever": "bm25", "max_turns": 8},
        "improved_haiku": {"mode": "agent", "model": HAIKU, "retriever": "hybrid",
                           "dense_weight": dense_weight, "max_turns": 8},
        "improved_sonnet": {"mode": "agent", "model": SONNET, "retriever": "hybrid",
                            "dense_weight": dense_weight, "max_turns": 8},
        "fixed_pipeline": {"mode": "pipeline", "model": HAIKU, "retriever": "hybrid",
                           "dense_weight": dense_weight, "max_turns": 2},
    }


def choose_pilot_claims(sample):
    counts = {label: 0 for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")}
    selected = []
    for claim_id in sample["claim_ids"]:
        label = sample["labels"][str(claim_id)]
        if counts[label] < 2:
            selected.append(claim_id)
            counts[label] += 1
    if any(count != 2 for count in counts.values()):
        raise ValueError("Pilot requires two frozen training claims per class.")
    return selected


def schedule_runs(claim_ids, repetitions=1):
    """Rotate four conditions within claims; reverse the rotation on repeat 2."""
    jobs = []
    for repetition in range(1, repetitions + 1):
        base = CONDITIONS if repetition % 2 else CONDITIONS[::-1]
        for index, claim_id in enumerate(claim_ids):
            shift = index % len(base)
            for condition in base[shift:] + base[:shift]:
                jobs.append({"claim_id": claim_id, "repetition": repetition, "condition": condition})
    return jobs


def build_pipeline_prompt(claim, documents, guidelines):
    return (guidelines + "\n\nClaim and research abstracts (data to assess):\n"
            + json.dumps({"claim": claim, "abstracts": documents}, ensure_ascii=False))


async def run_pipeline(claim, *, config, search_fn, workspace, guidelines,
                       system_prompt, run_metadata):
    """Python retrieves once and reads exactly three; the SDK only assesses."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, SystemMessage, query
    from agent import AgentReport, RunState, validate_report

    state = RunState(search_fn=search_fn)
    run_metadata.clear()
    run_metadata.update({
        "model": config["model"], "max_turns": config["max_turns"], "max_abstracts": 3,
        "api_timeout_ms": 120000, "max_api_retries": 2, "estimated_cost_usd": None,
        "tool_trace": state.tool_trace, "opened_document_ids": [],
        "trace_origin": "Fixed Python retrieval and reads, not Claude tool choices",
        "retrieval_seconds": None, "sdk_elapsed_seconds": None,
    })
    retrieval_start = time.perf_counter()
    sdk_start = None
    report = None
    try:
        await state.search({"query": claim})
        hits = state.tool_trace[-1]["results"]
        selected = [str(hit["document_id"]) for hit in hits[:3]]
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError("Pipeline requires three distinct retrieved abstracts.")
        run_metadata["selected_document_ids"] = selected
        for document_id in selected:
            result = await state.read({"document_id": document_id})
            if result.get("is_error"):
                raise RuntimeError("Pipeline could not read a selected abstract.")
        if list(state.opened_documents) != selected:
            raise ValueError("Read documents do not match the fixed top-three selection.")
        run_metadata["supplied_documents"] = list(state.opened_documents.values())
        run_metadata["retrieval_seconds"] = round(time.perf_counter() - retrieval_start, 6)
        prompt = build_pipeline_prompt(claim, run_metadata["supplied_documents"], guidelines)
        run_metadata["assessment_prompt_sha256"] = digest(prompt)
        options = ClaudeAgentOptions(
            model=config["model"], cwd=str(workspace), setting_sources=["project"],
            tools=[], skills=[], allowed_tools=[], mcp_servers={}, strict_mcp_config=True,
            env={"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
                 "ENABLE_TOOL_SEARCH": "false", "API_TIMEOUT_MS": "120000",
                 "CLAUDE_CODE_MAX_RETRIES": "2"},
            max_turns=config["max_turns"],
            output_format={"type": "json_schema", "schema": AgentReport.model_json_schema()},
            system_prompt=system_prompt,
        )
        sdk_start = time.perf_counter()
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                run_metadata["sdk_model"] = message.data.get("model")
            if not isinstance(message, ResultMessage):
                continue
            run_metadata.update({
                "estimated_cost_usd": message.total_cost_usd,
                "usage": getattr(message, "usage", None),
                "model_usage": getattr(message, "model_usage", None),
                "sdk_duration_ms": getattr(message, "duration_ms", None),
                "api_duration_ms": getattr(message, "duration_api_ms", None),
                "result_subtype": message.subtype, "sdk_is_error": message.is_error,
                "stop_reason": getattr(message, "stop_reason", None),
                "num_turns": getattr(message, "num_turns", None),
                "terminal_reason": getattr(message, "terminal_reason", None),
                "api_error_status": getattr(message, "api_error_status", None),
                "sdk_errors": message.errors, "result_text": message.result,
                "raw_structured_output": message.structured_output,
            })
            if message.is_error or message.structured_output is None:
                raise RuntimeError(f"Pipeline SDK failed: {message.subtype}; "
                                   f"{message.errors or message.result}")
            report = AgentReport.model_validate(message.structured_output)
            validate_report(report, state.opened_documents)
        if report is None:
            raise RuntimeError("Pipeline finished without returning a report.")
        print("\nPIPELINE REPORT:\n" + report.model_dump_json(indent=2), flush=True)
        for item in report.assessments:
            for sentence_id in item.evidence_sentence_ids:
                text = state.opened_documents[item.document_id]["sentences"][sentence_id]["text"]
                print(f"Document {item.document_id}, [{sentence_id}]: {text}", flush=True)
        return report
    finally:
        run_metadata["opened_document_ids"] = list(state.opened_documents)
        if sdk_start is not None:
            run_metadata["sdk_elapsed_seconds"] = round(time.perf_counter() - sdk_start, 6)
        elif run_metadata["retrieval_seconds"] is None:
            run_metadata["retrieval_seconds"] = round(time.perf_counter() - retrieval_start, 6)


def verify_run_config(metadata, config):
    for key, expected in (("model", config["model"]), ("max_turns", config["max_turns"]),
                          ("max_abstracts", 3), ("api_timeout_ms", 120000), ("max_api_retries", 2)):
        if metadata.get(key) != expected:
            raise RuntimeError(f"Unexpected run configuration: {key}")
    if metadata.get("sdk_model") not in (None, config["model"]):
        raise RuntimeError("SDK initialized with a different model than requested.")
    if len(metadata.get("opened_document_ids", [])) > 3:
        raise RuntimeError("The three-abstract limit was exceeded.")


def summarize_runs(runs, schedule):
    summary = {"completed_runs": sum(r["status"] in TERMINAL for r in runs),
               "planned_runs": len(schedule), "conditions": {}}
    summary["complete"] = summary["completed_runs"] == len(schedule)
    for name in CONDITIONS:
        rows = [r for r in runs if r["condition"] == name and r["status"] in TERMINAL]
        metrics = summarize_scores([r["score_counts"] for r in rows]) if rows else None
        timed = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None]
        traced = [r for r in rows if r.get("trace_complete", False)]
        costs = [r["metadata"]["estimated_cost_usd"] for r in rows
                 if r["metadata"].get("estimated_cost_usd") is not None]
        def mean(values):
            return statistics.mean(values) if values else None
        summary["conditions"][name] = {
            "attempts": len(rows), "metrics_all_attempts": metrics,
            "failures": sum(r["status"] != "ok" for r in rows),
            "nei_attempts": sum(r["gold_is_nei"] for r in rows),
            "nei_false_flags": sum(r["nei_outcome"] == "false_flag" for r in rows),
            "nei_correct_abstentions": sum(r["nei_outcome"] == "correct_abstention" for r in rows),
            "nei_processing_failures": sum(r["nei_outcome"] == "processing_failure" for r in rows),
            "mean_abstracts_read": mean([len(r["metadata"].get("opened_document_ids", []))
                                         for r in traced]),
            "mean_searches": mean([sum(e["tool"] == "search_abstracts"
                                       for e in r["metadata"].get("tool_trace", [])) for r in traced]),
            "mean_elapsed_seconds": mean(timed),
            "reported_cost_usd": sum(costs) if costs else None,
            "runs_without_elapsed_time": len(rows) - len(timed),
            "runs_without_cost": len(rows) - len(costs),
            "runs_without_complete_trace": len(rows) - len(traced),
        }
    return summary
