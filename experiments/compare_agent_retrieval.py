"""Prepare and resume the frozen Haiku BM25-versus-hybrid training comparison."""

import argparse
import asyncio
import fcntl
import os
import platform
import time
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.agent_comparison import (
    CONDITIONS, adapt_agent, finish_row, read_json, recover_interrupted,
    save_json, schedule_runs, sha256, summarize, validate_checkpoint,
)
from experiments.prepare_agent_training import baseline_options, select_claims
from experiments.prepare_evaluation import claim_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE = "experiments/agent_training_claims.json"
PLAN = "experiments/agent_retrieval_comparison.json"
OUTPUT = "results/agent_retrieval_training.json"
PACKAGES = ("claude-agent-sdk", "anthropic", "pydantic", "python-dotenv", "rank-bm25",
            "numpy", "sentence-transformers", "torch", "transformers", "huggingface-hub")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_claims(sample):
    import json
    path = PROJECT_DIR / "data/claims_train.jsonl"
    if sha256(path) != sample["dataset_sha256"]:
        raise ValueError("Training data changed after the sample was frozen.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate training claim IDs.")
    ids = sample["claim_ids"]
    if len(ids) != 60 or len(ids) != len(set(ids)) or set(ids) - set(by_id):
        raise ValueError("Expected 60 unique frozen training claims.")
    if sample["split"] != "train" or sample["repetitions_per_condition"] != 2:
        raise ValueError("Expected training claims and two repetitions per condition.")
    for claim_id in ids:
        if claim_label(by_id[claim_id]) != sample["labels"][str(claim_id)]:
            raise ValueError("Claim labels differ from the sample manifest.")
    return records, {claim_id: by_id[claim_id] for claim_id in ids}


def verify_files(fingerprints):
    for name, expected in fingerprints.items():
        if sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")


def prepare():
    output = PROJECT_DIR / PLAN
    if output.exists():
        raise FileExistsError("Comparison plan already exists. Keep it and run the comparison.")
    if (PROJECT_DIR / OUTPUT).exists():
        raise FileExistsError("A comparison checkpoint already exists.")
    sample = read_json(PROJECT_DIR / SAMPLE)
    records, _ = load_claims(sample)
    linked = {
        "data/corpus.jsonl": "corpus_sha256",
        "experiments/retrieval_split.json": "retrieval_split_sha256",
        "experiments/retrieval_config.json": "retrieval_config_sha256",
        "results/retrieval_hybrid_validation.json": "retrieval_validation_sha256",
    }
    for name, key in linked.items():
        if sha256(PROJECT_DIR / name) != sample[key]:
            raise ValueError(f"Frozen input changed: {name}")
    selected, _ = select_claims(
        records, read_json(PROJECT_DIR / "experiments/retrieval_split.json"),
        sample["claims_per_label"], sample["seed"],
    )
    if [claim["id"] for claim in selected] != sample["claim_ids"]:
        raise ValueError("Training sample no longer reproduces from its frozen seed.")
    if sample["first_comparison"]["current_haiku"] != {"retriever": "bm25"}:
        raise ValueError("Unexpected baseline condition.")
    hybrid = sample["first_comparison"]["hybrid_only_haiku"]
    validation = read_json(PROJECT_DIR / "results/retrieval_hybrid_validation.json")
    if (hybrid["retriever"] != "hybrid" or hybrid["dense_weight"] != validation["dense_weight"]
            or validation["decision"]["retrieval_gate_passed"] is not True):
        raise ValueError("Hybrid condition differs from the validated selection.")
    for name, expected in sample["baseline_source_sha256"].items():
        if name != "agent.py" and sha256(PROJECT_DIR / name) != expected:
            raise ValueError(f"Baseline file changed: {name}")
    agent_path = PROJECT_DIR / "agent.py"
    original = agent_path.read_text(encoding="utf-8")
    adapted = adapt_agent(original, sample["baseline_source_sha256"]["agent.py"])
    actual_options = baseline_options(adapted)
    if any(actual_options[key] != sample["shared_agent_options"][key] for key in actual_options):
        raise ValueError("Model or budgets differ from the frozen baseline.")
    if sample["shared_agent_options"]["use_skill"] is not True:
        raise ValueError("Both conditions must use the frozen guidelines.")
    # This adds only a per-run search function; default agent behavior is retained.
    if adapted != original:
        agent_path.write_text(adapted, encoding="utf-8")
    names = sorted(set(sample["baseline_source_sha256"]) | set(linked) | {
        SAMPLE, "data/claims_train.jsonl", "requirements.txt",
        "experiments/agent_comparison.py", "experiments/compare_agent_retrieval.py",
    })
    plan = {
        "schema_version": 1, "created_at": now(), "sample": sample,
        "baseline_agent_sha256": sample["baseline_source_sha256"]["agent.py"],
        "source_sha256": {name: sha256(PROJECT_DIR / name) for name in names},
        "package_versions": {name: version(name) for name in PACKAGES},
        "python_version": platform.python_version(),
        "platform_at_preparation": platform.platform(),
        "schedule": schedule_runs(sample["claim_ids"], sample["repetitions_per_condition"]),
        "execution": {
            "sdk_timeout_seconds_per_run": 300, "concurrency": 1,
            "pause_after_consecutive_failures": 3,
            "condition_order": "Alternate within claim; reverse which condition goes first on repeat 2",
            "workspace": "Temporary project containing only the frozen Skill",
            "latency_scope": "Complete SDK call including tools; retriever setup reported separately",
            "failure_policy": "Keep errors, timeouts and interrupted attempts; no full-run retries",
            "resume_policy": "Skip every saved terminal attempt; unfinished attempt becomes an interruption",
        },
        "scoring_policy": {
            "primary_population": "All scheduled attempts, pooling counts over both repetitions",
            "failed_output": "No positive predictions, gold retained, zero correct-NEI credit",
            "secondary_population": "Same claim/repetition pairs successful in both conditions",
            "false_flag": "Any positive verdict in a successful report for a gold NEI claim",
        },
        "training_keep_rule": [
            "Strictly higher evidence F1 on all attempts",
            "No lower document/verdict F1 on all attempts",
            "No more NEI false flags",
            "No more processing failures",
        ],
        "interpretation": "Exploratory training selection; final performance claims need the frozen dev experiment",
    }
    with output.open("x", encoding="utf-8") as file:
        import json
        json.dump(plan, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print("Added the per-run search adapter to agent.py.")
    print(f"Frozen {len(plan['schedule'])} runs; shared model: {actual_options['model']}")
    print(f"Saved: {output}")


def verify_plan(plan):
    if plan["schema_version"] != 1:
        raise ValueError("Unsupported comparison plan version.")
    verify_files(plan["source_sha256"])
    for name, expected in plan["package_versions"].items():
        if version(name) != expected:
            raise RuntimeError(f"Package version changed: {name}; expected {expected}.")
    if platform.python_version() != plan["python_version"]:
        raise RuntimeError("Python version changed after preparation.")
    if plan["sample"] != read_json(PROJECT_DIR / SAMPLE):
        raise ValueError("Training sample changed.")
    expected = schedule_runs(plan["sample"]["claim_ids"], 2)
    if plan["schedule"] != expected:
        raise ValueError("Comparison schedule differs from the frozen design.")


async def run(max_new_runs):
    plan_path = PROJECT_DIR / PLAN
    plan = read_json(plan_path)
    verify_plan(plan)
    plan_hash = sha256(plan_path)
    sample = plan["sample"]
    _, claims = load_claims(sample)
    output = PROJECT_DIR / OUTPUT
    if output.exists():
        results = read_json(output)
        if results["plan_sha256"] != plan_hash:
            raise ValueError("Checkpoint belongs to a different comparison plan.")
    else:
        results = {"schema_version": 1, "status": "running", "started_at": now(),
                   "split": "train", "plan_sha256": plan_hash, "plan": plan,
                   "sessions": [], "runs": []}
    validate_checkpoint(results["runs"], plan["schedule"])
    recover_interrupted(results["runs"], claims)
    results["summary"] = summarize(results["runs"], plan["schedule"])
    if results["summary"]["complete"]:
        results["status"] = "finished"
        save_json(output, results)
        print_summary(results, output)
        return
    save_json(output, results)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing from the environment or project .env.")
    from agent import main as run_agent
    from corpus_tools import search_abstracts
    from hybrid_retrieval import HybridRetriever

    setup_start = time.perf_counter()
    retriever = HybridRetriever()
    results["sessions"].append({
        "started_at": now(), "platform": platform.platform(),
        "retriever_setup_seconds": round(time.perf_counter() - setup_start, 3),
        "embedding_cache_file": retriever.cache_path.name,
        "embedding_cache_sha256": sha256(retriever.cache_path),
    })
    searchers = {"current_haiku": search_abstracts, "hybrid_only_haiku": partial(
        retriever.search, dense_weight=sample["first_comparison"]["hybrid_only_haiku"]["dense_weight"],
    )}
    save_json(output, results)
    new_runs = 0
    consecutive_failures = 0
    with TemporaryDirectory(prefix="scifact-training-") as folder:
        skill_path = Path(folder) / ".claude/skills/scientific-evidence/SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(sample["skill_text"], encoding="utf-8")
        if sha256(skill_path) != sample["skill_sha256"]:
            raise ValueError("Frozen Skill text and hash disagree.")
        for job in plan["schedule"][len(results["runs"]):]:
            if max_new_runs is not None and new_runs >= max_new_runs:
                break
            verify_files(plan["source_sha256"])
            if sha256(plan_path) != plan_hash:
                raise RuntimeError("Comparison plan changed during the run.")
            if sha256(skill_path) != sample["skill_sha256"]:
                raise RuntimeError("Skill workspace changed during the run.")
            claim = claims[job["claim_id"]]
            row = {**job, "claim": claim["claim"], "gold_evidence": claim["evidence"],
                   "status": "running", "started_at": now(), "elapsed_seconds": None,
                   "trace_complete": False,
                   "metadata": {"estimated_cost_usd": None, "tool_trace": [], "opened_document_ids": []}}
            results["runs"].append(row)
            save_json(output, results)
            print(f"\nRUN {len(results['runs'])}/{len(plan['schedule'])} | claim {job['claim_id']}"
                  f" | repeat {job['repetition']} | {job['condition']}", flush=True)
            start = time.perf_counter()
            try:
                report = await asyncio.wait_for(run_agent(
                    claim=claim["claim"], use_skill=True, run_metadata=row["metadata"],
                    skill_workspace=folder, search_fn=searchers[job["condition"]],
                ), timeout=plan["execution"]["sdk_timeout_seconds_per_run"])
                for key in ("model", "max_turns", "max_abstracts"):
                    if row["metadata"].get(key) != sample["shared_agent_options"][key]:
                        raise RuntimeError(f"Agent returned unexpected configuration: {key}")
                finish_row(row, claim, "ok", report.model_dump())
            except asyncio.TimeoutError:
                finish_row(row, claim, "timeout")
                row["error"] = (
                    f"Exceeded the frozen {plan['execution']['sdk_timeout_seconds_per_run']}"
                    "-second run timeout."
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                finish_row(row, claim, "interrupted")
                row["error"] = "Run interrupted; recorded as a processing failure and not retried."
                raise
            except Exception as error:
                finish_row(row, claim, "error")
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
                row["trace_complete"] = True
                row["finished_at"] = now()
                results["summary"] = summarize(results["runs"], plan["schedule"])
                results["status"] = "finished" if results["summary"]["complete"] else "paused"
                save_json(output, results)
            new_runs += 1
            if row["status"] != "ok":
                print(f"RUN FAILED: {row['error']}", flush=True)
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            verify_files(plan["source_sha256"])
            if sha256(plan_path) != plan_hash:
                raise RuntimeError("Comparison plan changed during the run.")
            if consecutive_failures >= plan["execution"]["pause_after_consecutive_failures"]:
                print("Paused after three consecutive failures. Review the recorded errors before continuing.")
                break
    if results["summary"]["complete"]:
        results["finished_at"] = now()
        results["status"] = "finished"
    else:
        results["status"] = "paused"
    save_json(output, results)
    print_summary(results, output)


def print_summary(results, path):
    import json
    print("\nAGENT TRAINING COMPARISON:")
    summary = results["summary"]
    display = {"completed_runs": summary["completed_runs"],
               "planned_runs": summary["planned_runs"], "conditions": {}}
    for name, condition in summary["conditions"].items():
        metrics = condition["metrics_all_attempts"]
        display["conditions"][name] = {
            "attempts": condition["attempts"],
            "evidence_f1": metrics["evidence_sentences"]["f1"] if metrics else None,
            "document_verdict_f1": metrics["document_labels"]["f1"] if metrics else None,
            "nei_false_flags": condition["nei_false_flags"],
            "nei_correct_abstentions": condition["nei_correct_abstentions"],
            "nei_processing_failures": condition["nei_processing_failures"],
            "failures": condition["failures"],
            "mean_abstracts_read": condition["mean_abstracts_read_observed"],
        }
    display["training_decision"] = summary["training_decision"]
    print(json.dumps(display, indent=2))
    print(f"Saved: {path}")
    if not results["summary"]["complete"]:
        print("Partial checkpoint. Run the same command again to continue; no training decision yet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    if args.prepare and args.max_new_runs is not None:
        parser.error("Use --prepare separately from running the comparison")
    # A process lock prevents two terminals from writing the same checkpoint.
    lock_path = PROJECT_DIR / "data/agent_retrieval_comparison.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this comparison checkpoint.") from error
        if args.prepare:
            prepare()
        else:
            asyncio.run(run(args.max_new_runs))


if __name__ == "__main__":
    main()
