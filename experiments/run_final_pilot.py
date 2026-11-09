"""Resume the 24-attempt training-only pilot; never selects or runs dev claims."""

import argparse
import asyncio
import fcntl
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments import agent_comparison as comparison
from experiments.compare_agent_retrieval import load_claims, now
from experiments.final_conditions import CONDITIONS, run_pipeline, summarize_runs, verify_run_config
from experiments.prepare_final_pilot import PLAN, OUTPUT, verify_files, verify_plan

PROJECT_DIR = Path(__file__).resolve().parents[1]


def print_summary(results, output):
    summary = results["summary"]
    display = {"completed_runs": summary["completed_runs"], "planned_runs": summary["planned_runs"],
               "conditions": {}}
    for name, data in summary["conditions"].items():
        metrics = data["metrics_all_attempts"]
        display["conditions"][name] = {
            "attempts": data["attempts"], "failures": data["failures"],
            "evidence_f1": metrics["evidence_sentences"]["f1"] if metrics else None,
            "document_verdict_f1": metrics["document_labels"]["f1"] if metrics else None,
            "nei_false_flags": data["nei_false_flags"],
            "nei_processing_failures": data["nei_processing_failures"],
            "mean_searches": data["mean_searches"], "mean_abstracts_read": data["mean_abstracts_read"],
            "mean_elapsed_seconds": data["mean_elapsed_seconds"],
        }
    display["technical_pilot_passed"] = (summary["complete"] and all(
        data["failures"] == 0 for data in summary["conditions"].values()))
    print("\nFOUR-CONDITION TRAINING PILOT:")
    print(json.dumps(display, indent=2))
    print("Pilot scores are descriptive. This pilot makes no performance-selection decision.")
    print(f"Saved: {output}")


async def run(max_new_runs=None):
    plan_path = PROJECT_DIR / PLAN
    plan = comparison.read_json(plan_path)
    verify_plan(plan)
    plan_hash = comparison.sha256(plan_path)
    _, claims = load_claims(plan["sample"])
    output = PROJECT_DIR / OUTPUT
    if output.exists():
        results = comparison.read_json(output)
        if results["plan_sha256"] != plan_hash or results["plan"] != plan:
            raise ValueError("Checkpoint belongs to a different pilot plan.")
    else:
        results = {"schema_version": 1, "status": "running", "started_at": now(),
                   "split": "train", "role": plan["role"], "plan": plan, "plan_sha256": plan_hash,
                   "sessions": [], "runs": []}
    comparison.validate_checkpoint(results["runs"], plan["schedule"])
    comparison.recover_interrupted(results["runs"], claims)
    results["summary"] = summarize_runs(results["runs"], plan["schedule"])
    if results["summary"]["complete"]:
        results["status"] = "finished"
        comparison.save_json(output, results)
        print_summary(results, output)
        return
    comparison.save_json(output, results)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing from the environment or project .env.")
    from agent import main as run_agent
    from corpus_tools import search_abstracts
    from hybrid_retrieval import HybridRetriever

    setup_start = time.perf_counter()
    retriever = HybridRetriever()
    # Prime the local embedding device with a fixed, non-evaluation query.
    retriever.embed_queries(["scientific research"])
    weight = plan["conditions"]["improved_haiku"]["dense_weight"]
    hybrid_search = partial(retriever.search, dense_weight=weight)
    searchers = {"bm25": search_abstracts, "hybrid": hybrid_search}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                          capture_output=True, text=True, check=False)
    results["sessions"].append({
        "started_at": now(), "platform": platform.platform(),
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "retriever_initialization_seconds": round(time.perf_counter() - setup_start, 6),
        "warmup_query": "scientific research",
        "embedding_cache_file": retriever.cache_path.name,
        "embedding_cache_sha256": comparison.sha256(retriever.cache_path),
    })
    comparison.save_json(output, results)
    consecutive_failures = 0
    new_runs = 0
    with TemporaryDirectory(prefix="scifact-four-condition-") as folder:
        agent_workspace = Path(folder) / "agent"
        skill = agent_workspace / ".claude/skills/scientific-evidence/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(plan["sample"]["skill_text"], encoding="utf-8")
        pipeline_workspace = Path(folder) / "pipeline"
        pipeline_workspace.mkdir()
        for job in plan["schedule"][len(results["runs"]):]:
            if max_new_runs is not None and new_runs >= max_new_runs:
                break
            verify_files(plan["source_sha256"])
            if comparison.sha256(plan_path) != plan_hash:
                raise RuntimeError("Pilot plan changed during the run.")
            if comparison.sha256(skill) != plan["sample"]["skill_sha256"]:
                raise RuntimeError("Frozen Skill workspace changed.")
            config = plan["conditions"][job["condition"]]
            claim = claims[job["claim_id"]]
            row = {**job, "claim": claim["claim"], "gold_evidence": claim["evidence"],
                   "configuration": config, "status": "running", "started_at": now(),
                   "elapsed_seconds": None, "trace_complete": False,
                   "metadata": {"estimated_cost_usd": None, "tool_trace": [], "opened_document_ids": []}}
            results["runs"].append(row)
            comparison.save_json(output, results)
            print(f"\nPILOT {len(results['runs'])}/{len(plan['schedule'])} | claim {job['claim_id']}"
                  f" | {job['condition']} | {config['model']}", flush=True)
            start = time.perf_counter()
            try:
                if config["mode"] == "agent":
                    operation = run_agent(
                        claim=claim["claim"], use_skill=True, run_metadata=row["metadata"],
                        skill_workspace=str(agent_workspace), search_fn=searchers[config["retriever"]],
                        model=config["model"],
                    )
                else:
                    operation = run_pipeline(
                        claim["claim"], config=config, search_fn=hybrid_search,
                        workspace=pipeline_workspace, guidelines=plan["assessment_guidelines"],
                        system_prompt=plan["system_prompts"]["pipeline"], run_metadata=row["metadata"],
                    )
                report = await asyncio.wait_for(operation, timeout=plan["execution"]["timeout_seconds_per_attempt"])
                verify_run_config(row["metadata"], config)
                comparison.finish_row(row, claim, "ok", report.model_dump())
            except asyncio.TimeoutError:
                comparison.finish_row(row, claim, "timeout")
                row["error"] = "Exceeded the frozen per-attempt timeout."
            except (asyncio.CancelledError, KeyboardInterrupt):
                comparison.finish_row(row, claim, "interrupted")
                row["error"] = "Interrupted attempt retained as a processing failure; not retried."
                raise
            except Exception as error:
                comparison.finish_row(row, claim, "error")
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - start, 6)
                row["trace_complete"] = True
                row["finished_at"] = now()
                row["metadata"].setdefault("trace_origin", "Claude-selected MCP tool calls")
                results["summary"] = summarize_runs(results["runs"], plan["schedule"])
                results["status"] = "finished" if results["summary"]["complete"] else "paused"
                comparison.save_json(output, results)
            new_runs += 1
            consecutive_failures = consecutive_failures + 1 if row["status"] != "ok" else 0
            if row["status"] != "ok":
                print(f"PILOT FAILED: {row['error']}", flush=True)
            verify_files(plan["source_sha256"])
            if comparison.sha256(plan_path) != plan_hash:
                raise RuntimeError("Pilot plan changed during the run.")
            if consecutive_failures >= plan["execution"]["pause_after_consecutive_failures"]:
                print("Paused after three consecutive failures. Review the saved errors before continuing.")
                break
    if results["summary"]["complete"]:
        results["finished_at"] = now()
        results["status"] = "finished"
    else:
        results["status"] = "paused"
    comparison.save_json(output, results)
    print_summary(results, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    with (PROJECT_DIR / "data/final_conditions_pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this pilot checkpoint.") from error
        asyncio.run(run(args.max_new_runs))


if __name__ == "__main__":
    main()
