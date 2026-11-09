"""Small UI adapter. Live runs use the existing agent; recordings never call it."""

import asyncio
import copy
import json
import os
import time
from pathlib import Path

from agent import AgentReport, main as run_agent, validate_report
from corpus_tools import DATA_PATH, read_abstract

PROJECT_DIR = Path(__file__).resolve().parent
EXAMPLES_PATH = PROJECT_DIR / "docs/demo_examples.json"
MODEL = "claude-haiku-5-5"


def load_examples():
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


def live_requirements():
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("Add ANTHROPIC_API_KEY to your local .env file.")
    if not DATA_PATH.is_file():
        missing.append("Download data/corpus.jsonl using the README Setup instructions.")
    return missing


def present_run(row, documents, *, mode, source=None):
    """Resolve quotes by sentence ID; never quote the generated explanation."""
    row = copy.deepcopy(row)
    metadata = row["metadata"]
    result = {
        "mode": mode, "claim": row["claim"], "status": row["status"],
        "model": metadata.get("model", MODEL),
        "retriever": row.get("configuration", {}).get("retriever", "bm25"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "estimated_cost_usd": metadata.get("estimated_cost_usd"),
        "tool_trace": metadata.get("tool_trace", []),
        "abstracts_read": len(metadata.get("opened_document_ids", [])),
        "assessments": [], "error": row.get("error"), "source": source,
    }
    if row["status"] != "ok":
        return result
    report = AgentReport.model_validate(row["report"])
    by_id = {doc["document_id"]: doc for doc in documents}
    if len(by_id) != len(documents) or set(by_id) != set(metadata["opened_document_ids"]):
        raise ValueError("The displayed documents do not match the recorded reads.")
    validate_report(report, by_id)
    for item in report.assessments:
        doc = by_id[item.document_id]
        result["assessments"].append({
            **item.model_dump(), "title": doc["title"],
            "quotes": [{"sentence_id": index, "text": doc["sentences"][index]["text"]}
                       for index in item.evidence_sentence_ids],
        })
    return result


def recorded_example(index):
    saved = load_examples()
    example = saved["examples"][index]
    row = example["run"]
    return present_run(row, example["documents"], mode="recorded", source={
        "path": saved["source_path"], "sha256": saved["source_sha256"],
        "claim_id": row["claim_id"], "condition": row["condition"],
        "repetition": row["repetition"],
    })


def run_live(claim, timeout=300):
    claim = claim.strip()
    if not claim or len(claim) > 2000:
        raise ValueError("Enter a claim between 1 and 2,000 characters.")
    missing = live_requirements()
    if missing:
        raise ValueError(" ".join(missing))
    metadata = {}
    start = time.perf_counter()

    async def execute():
        return await asyncio.wait_for(
            run_agent(claim=claim, use_skill=True, run_metadata=metadata, model=MODEL),
            timeout=timeout,
        )

    try:
        report = asyncio.run(execute())
        documents = [read_abstract(doc_id) for doc_id in metadata["opened_document_ids"]]
        row = {"claim": claim, "status": "ok", "report": report.model_dump(),
               "metadata": metadata, "elapsed_seconds": time.perf_counter() - start}
        return present_run(row, documents, mode="live")
    except Exception as error:
        message = (f"The run exceeded its {timeout:g}-second time limit."
                   if isinstance(error, TimeoutError) else f"{type(error).__name__}: {error}")
        # Keep credentials out of a displayed/downloaded SDK error.
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            message = message.replace(key, "[redacted]")
        return present_run({
            "claim": claim, "status": "error", "error": message, "metadata": metadata,
            "elapsed_seconds": time.perf_counter() - start,
        }, [], mode="live")
