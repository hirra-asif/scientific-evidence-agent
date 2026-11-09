"""Run/resume the frozen final evaluation through the already piloted harness."""

import argparse
import asyncio
import fcntl
import json
from pathlib import Path

from experiments import agent_comparison as comparison
from experiments.final_evaluation_common import PLAN, OUTPUT, require_frozen_commit
from experiments.prepare_final_evaluation import load_dev_claims, verify_plan

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
            "nei_correct_abstentions": data["nei_correct_abstentions"],
            "nei_processing_failures": data["nei_processing_failures"],
            "mean_abstracts_read": data["mean_abstracts_read"],
            "mean_searches": data["mean_searches"],
            "mean_elapsed_seconds": data["mean_elapsed_seconds"],
            "reported_cost_usd": data["reported_cost_usd"],
        }
    print("\nFROZEN DEV EVALUATION:")
    print(json.dumps(display, indent=2))
    print(f"Saved: {output}")
    if summary["complete"]:
        print("Complete. Run: python -m experiments.analyze_final_evaluation")
    else:
        print("Partial checkpoint; continue with the same command. No final inference yet.")


async def run(max_new_runs=None):
    from experiments.run_final_pilot import run as shared_run
    plan_path = PROJECT_DIR / PLAN
    plan = comparison.read_json(plan_path)
    verify_plan(plan)
    commit = require_frozen_commit(PROJECT_DIR, plan, plan_path)
    print(f"Frozen files verified against commit {commit}.")
    await shared_run(max_new_runs, plan_path=plan_path, output_path=PROJECT_DIR / OUTPUT,
                     plan_verifier=verify_plan, claims_loader=load_dev_claims,
                     summary_printer=print_summary, progress_label="FINAL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    with (PROJECT_DIR / "data/final_evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using the final evaluation checkpoint.") from error
        asyncio.run(run(args.max_new_runs))


if __name__ == "__main__":
    main()
