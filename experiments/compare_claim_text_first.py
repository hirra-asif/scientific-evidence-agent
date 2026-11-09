"""Freeze and run candidate 2: a claim-text-first instruction, with BM25 in both arms.

Prepare once, commit the manifest, pilot six attempts, then resume the same file.
The completed retrieval comparison and its results are never overwritten.
"""

import argparse
import asyncio
import fcntl
import hashlib
import importlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments import agent_comparison as comparison
from experiments.compare_agent_retrieval import (
    PACKAGES, load_claims, now, print_summary, verify_files,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_PLAN = "experiments/agent_retrieval_comparison.json"
PARENT_RESULT = "results/agent_retrieval_training.json"
HELPER = "experiments/agent_comparison.py"
SAMPLE = "experiments/agent_training_claims.json"
PLAN = "experiments/agent_claim_text_first_comparison.json"
OUTPUT = "results/agent_claim_text_first_training.json"
CONDITIONS = ("current_haiku", "claim_text_first_haiku")
KEEP_KEY = "keep_claim_text_first_for_combination"
INSTRUCTION = (
    "Search policy for this run:\n"
    "Start by calling search_abstracts with the entire claim, copied verbatim, "
    "as the query.\n"
    "Subsequent searches may use revised queries.\n\n"
)

# Add optional parameters to the shared scoring helper. Its original defaults
# and the completed retrieval comparison's results must remain identical.
HELPER_PATCHES = (
    ("def schedule_runs(claim_ids, repetitions):",
     "def schedule_runs(claim_ids, repetitions, conditions=CONDITIONS):"),
    ("order = CONDITIONS if (index + repetition - 1) % 2 == 0 else CONDITIONS[::-1]",
     "order = conditions if (index + repetition - 1) % 2 == 0 else conditions[::-1]"),
    ("def summarize(runs, schedule):",
     'def summarize(runs, schedule, conditions=CONDITIONS, keep_key="keep_hybrid_for_combination"):'),
    ("for condition in CONDITIONS", "for condition in conditions"),
    ("paired = success[CONDITIONS[0]] & success[CONDITIONS[1]]",
     "paired = success[conditions[0]] & success[conditions[1]]"),
    ('{"ready": complete, "keep_hybrid_for_combination": None}',
     '{"ready": complete, keep_key: None}'),
    ('baseline, hybrid = (summary["conditions"][name] for name in CONDITIONS)',
     'baseline, candidate = (summary["conditions"][name] for name in conditions)'),
    ('hybrid[', 'candidate['),
    ('.update(checks=checks, keep_hybrid_for_combination=all(checks.values()))',
     '.update({"checks": checks, keep_key: all(checks.values())})'),
)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapt_helper(source, frozen_hash):
    """Reject unrelated edits; accept either the frozen or already adapted file."""
    original = source
    if text_hash(original) != frozen_hash:
        for old, new in reversed(HELPER_PATCHES):
            if new not in original:
                raise ValueError("Scoring helper differs from the expected adaptation.")
            original = original.replace(new, old)
        if text_hash(original) != frozen_hash:
            raise ValueError("Scoring helper contains unrelated edits; nothing was overwritten.")
    adapted = original
    for old, new in HELPER_PATCHES:
        if old not in adapted:
            raise ValueError("Cannot locate the expected scoring-helper code.")
        adapted = adapted.replace(old, new)
    return adapted


def candidate_skill(baseline):
    anchor = "For each abstract:\n"
    if baseline.count(anchor) != 1 or baseline.count("$ARGUMENTS") != 1:
        raise ValueError("Unexpected frozen Skill layout.")
    if INSTRUCTION in baseline:
        raise ValueError("The baseline already contains the candidate instruction.")
    return baseline.replace(anchor, INSTRUCTION + anchor, 1)


def first_query_behavior(row):
    events = [event for event in row.get("metadata", {}).get("tool_trace", [])
              if event.get("tool") in {"search_abstracts", "read_abstract"}]
    first_search = next((event for event in events
                         if event["tool"] == "search_abstracts"), None)
    query = first_search.get("arguments", {}).get("query") if first_search else None
    exact = query == row["claim"] if query is not None else None
    first_is_search = events[0]["tool"] == "search_abstracts" if events else None
    return {"first_search_query": query, "first_query_exact_claim": exact,
            "first_corpus_action_is_search": first_is_search,
            "follows_first_query_instruction": (
                bool(exact and first_is_search) if query is not None else None)}


def summarize_results(results):
    summary = comparison.summarize(
        results["runs"], results["plan"]["schedule"], conditions=CONDITIONS,
        keep_key=KEEP_KEY,
    )
    for row in results["runs"]:
        row["first_query_behavior"] = first_query_behavior(row)
    for condition in CONDITIONS:
        rows = [row for row in results["runs"]
                if row["condition"] == condition and row["status"] in comparison.TERMINAL]
        traced = [row for row in rows if row.get("trace_complete", False)]
        with_search = [row for row in traced
                       if row["first_query_behavior"]["first_search_query"] is not None]
        follows = sum(row["first_query_behavior"]["follows_first_query_instruction"]
                      is True for row in with_search)
        summary["conditions"][condition]["first_query_behavior"] = {
            "runs_with_complete_trace": len(traced),
            "runs_without_complete_trace": len(rows) - len(traced),
            "runs_with_search": len(with_search),
            "runs_without_search": len(traced) - len(with_search),
            "runs_following_instruction": follows,
            "fraction_following_instruction_among_runs_with_search": (
                follows / len(with_search) if with_search else None),
        }
    return summary


def prepare():
    plan_path = PROJECT_DIR / PLAN
    if plan_path.exists():
        verify_plan(comparison.read_json(plan_path))
        print(f"Existing frozen plan verified: {plan_path}")
        return
    if (PROJECT_DIR / OUTPUT).exists():
        raise FileExistsError("A checkpoint exists without its plan; restore the original plan.")
    parent = comparison.read_json(PROJECT_DIR / PARENT_RESULT)
    parent_plan = comparison.read_json(PROJECT_DIR / PARENT_PLAN)
    if (parent["plan"] != parent_plan
            or parent["plan_sha256"] != comparison.sha256(PROJECT_DIR / PARENT_PLAN)):
        raise ValueError("The completed result and retrieval plan do not match.")
    comparison.validate_checkpoint(parent["runs"], parent_plan["schedule"])
    old_summary = comparison.summarize(parent["runs"], parent_plan["schedule"])
    if (parent["status"] != "finished" or not old_summary["complete"]
            or old_summary != parent["summary"]):
        raise ValueError("The retrieval comparison must be complete and reproduce exactly.")
    verify_files({name: value for name, value in parent_plan["source_sha256"].items()
                  if name != HELPER})
    sample = parent_plan["sample"]
    if sample != comparison.read_json(PROJECT_DIR / SAMPLE):
        raise ValueError("The frozen training sample changed.")
    load_claims(sample)
    if text_hash(sample["skill_text"]) != sample["skill_sha256"]:
        raise ValueError("Frozen Skill text and hash disagree.")
    packages = {name: version(name) for name in PACKAGES}
    if packages != parent_plan["package_versions"]:
        raise ValueError("Package versions changed since the retrieval comparison.")
    if platform.python_version() != parent_plan["python_version"]:
        raise ValueError("Python version changed since the retrieval comparison.")
    helper_path = PROJECT_DIR / HELPER
    original = helper_path.read_text(encoding="utf-8")
    adapted = adapt_helper(original, parent_plan["source_sha256"][HELPER])
    profiles = {CONDITIONS[0]: sample["skill_text"],
                CONDITIONS[1]: candidate_skill(sample["skill_text"])}
    try:
        if original != adapted:
            helper_path.write_text(adapted, encoding="utf-8")
        importlib.invalidate_caches()
        importlib.reload(comparison)
        if comparison.summarize(parent["runs"], parent_plan["schedule"]) != old_summary:
            raise RuntimeError("Scoring-helper adaptation changed the completed result.")
        names = set(parent_plan["source_sha256"]) | {
            PARENT_PLAN, PARENT_RESULT, "experiments/compare_claim_text_first.py",
        }
        plan = {
            "schema_version": 1, "candidate": "claim_text_first", "created_at": now(),
            "sample": sample, "conditions": list(CONDITIONS),
            "retriever_by_condition": {name: "bm25" for name in CONDITIONS},
            "skill_profiles": {name: {"text": text, "sha256": text_hash(text)}
                               for name, text in profiles.items()},
            "schedule": comparison.schedule_runs(sample["claim_ids"], 2, conditions=CONDITIONS),
            "source_sha256": {name: comparison.sha256(PROJECT_DIR / name)
                              for name in sorted(names)},
            "package_versions": packages, "python_version": platform.python_version(),
            "platform_at_preparation": platform.platform(),
            "execution": parent_plan["execution"],
            "scoring_policy": parent_plan["scoring_policy"],
            "training_keep_rule": parent_plan["training_keep_rule"],
            "instruction_compliance_policy": (
                "Record first corpus action and exact first-query match. "
                "Keep every attempt in scoring, whether or not the instruction is followed. "
                "No search is reported separately; incomplete traces do not establish compliance."
            ),
            "interpretation": parent_plan["interpretation"],
        }
        with plan_path.open("x", encoding="utf-8") as file:
            json.dump(plan, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except BaseException:
        if adapted != original:
            helper_path.write_text(original, encoding="utf-8")
            importlib.invalidate_caches()
            importlib.reload(comparison)
        raise
    print("Completed retrieval result reproduced exactly; original agent and Skill unchanged.")
    print("Frozen 240 runs: current_haiku versus claim_text_first_haiku; both use BM25.")
    print(f"Saved: {plan_path}")


def verify_plan(plan):
    if plan["schema_version"] != 1 or plan["candidate"] != "claim_text_first":
        raise ValueError("Unexpected comparison plan.")
    verify_files(plan["source_sha256"])
    if {name: version(name) for name in PACKAGES} != plan["package_versions"]:
        raise RuntimeError("Package versions changed after preparation.")
    if platform.python_version() != plan["python_version"]:
        raise RuntimeError("Python version changed after preparation.")
    sample = plan["sample"]
    if sample != comparison.read_json(PROJECT_DIR / SAMPLE):
        raise ValueError("Training sample changed.")
    if plan["conditions"] != list(CONDITIONS):
        raise ValueError("Condition names changed.")
    if plan["retriever_by_condition"] != {name: "bm25" for name in CONDITIONS}:
        raise ValueError("Both conditions must use BM25.")
    if plan["schedule"] != comparison.schedule_runs(sample["claim_ids"], 2, conditions=CONDITIONS):
        raise ValueError("Schedule differs from the frozen design.")
    for name, expected in zip(CONDITIONS, (sample["skill_text"], candidate_skill(sample["skill_text"]))):
        if plan["skill_profiles"][name] != {"text": expected, "sha256": text_hash(expected)}:
            raise ValueError("Skill profile differs from the single planned instruction change.")


async def run(max_new_runs):
    plan_path = PROJECT_DIR / PLAN
    plan = comparison.read_json(plan_path)
    verify_plan(plan)
    plan_hash = comparison.sha256(plan_path)
    _, claims = load_claims(plan["sample"])
    output = PROJECT_DIR / OUTPUT
    if output.exists():
        results = comparison.read_json(output)
        if results["plan_sha256"] != plan_hash or results["plan"] != plan:
            raise ValueError("Checkpoint belongs to a different comparison plan.")
    else:
        results = {"schema_version": 1, "status": "running", "started_at": now(),
                   "split": "train", "plan_sha256": plan_hash, "plan": plan,
                   "sessions": [], "runs": []}
    comparison.validate_checkpoint(results["runs"], plan["schedule"])
    comparison.recover_interrupted(results["runs"], claims)
    results["summary"] = summarize_results(results)
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

    results["sessions"].append({"started_at": now(), "platform": platform.platform(),
                               "retriever": "bm25"})
    new_runs = 0
    consecutive_failures = 0
    with TemporaryDirectory(prefix="scifact-claim-first-") as folder:
        workspaces = {}
        for condition, profile in plan["skill_profiles"].items():
            workspace = Path(folder) / condition
            skill = workspace / ".claude/skills/scientific-evidence/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(profile["text"], encoding="utf-8")
            workspaces[condition] = (workspace, skill)
        for job in plan["schedule"][len(results["runs"]):]:
            if max_new_runs is not None and new_runs >= max_new_runs:
                break
            verify_files(plan["source_sha256"])
            if comparison.sha256(plan_path) != plan_hash:
                raise RuntimeError("Comparison plan changed during the run.")
            workspace, skill = workspaces[job["condition"]]
            if comparison.sha256(skill) != plan["skill_profiles"][job["condition"]]["sha256"]:
                raise RuntimeError("Skill workspace changed during the run.")
            claim = claims[job["claim_id"]]
            row = {**job, "claim": claim["claim"], "gold_evidence": claim["evidence"],
                   "status": "running", "started_at": now(), "elapsed_seconds": None,
                   "trace_complete": False,
                   "metadata": {"estimated_cost_usd": None, "tool_trace": [], "opened_document_ids": []}}
            results["runs"].append(row)
            comparison.save_json(output, results)
            print(f"\nRUN {len(results['runs'])}/{len(plan['schedule'])} | claim {job['claim_id']}"
                  f" | repeat {job['repetition']} | {job['condition']}", flush=True)
            start = time.perf_counter()
            try:
                report = await asyncio.wait_for(run_agent(
                    claim=claim["claim"], use_skill=True, run_metadata=row["metadata"],
                    skill_workspace=str(workspace), search_fn=search_abstracts,
                ), timeout=plan["execution"]["sdk_timeout_seconds_per_run"])
                for key in ("model", "max_turns", "max_abstracts"):
                    if row["metadata"].get(key) != plan["sample"]["shared_agent_options"][key]:
                        raise RuntimeError(f"Agent returned unexpected configuration: {key}")
                comparison.finish_row(row, claim, "ok", report.model_dump())
            except asyncio.TimeoutError:
                comparison.finish_row(row, claim, "timeout")
                row["error"] = "Exceeded the frozen per-run timeout."
            except (asyncio.CancelledError, KeyboardInterrupt):
                comparison.finish_row(row, claim, "interrupted")
                row["error"] = "Run interrupted; kept as a processing failure and not retried."
                raise
            except Exception as error:
                comparison.finish_row(row, claim, "error")
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
                row["trace_complete"] = True
                row["finished_at"] = now()
                results["summary"] = summarize_results(results)
                results["status"] = "finished" if results["summary"]["complete"] else "paused"
                comparison.save_json(output, results)
            new_runs += 1
            if row["status"] != "ok":
                print(f"RUN FAILED: {row['error']}", flush=True)
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            verify_files(plan["source_sha256"])
            if comparison.sha256(plan_path) != plan_hash:
                raise RuntimeError("Comparison plan changed during the run.")
            if consecutive_failures >= plan["execution"]["pause_after_consecutive_failures"]:
                print("Paused after three consecutive failures. Review errors before continuing.")
                break
    if results["summary"]["complete"]:
        results["finished_at"] = now()
        results["status"] = "finished"
    else:
        results["status"] = "paused"
    comparison.save_json(output, results)
    print_summary(results, output)
    print("\nFIRST-QUERY BEHAVIOR (descriptive; every attempt remains in the scores):")
    print(json.dumps({name: results["summary"]["conditions"][name]["first_query_behavior"]
                      for name in CONDITIONS}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    if args.prepare and args.max_new_runs is not None:
        parser.error("Use --prepare separately from running")
    with (PROJECT_DIR / "data/agent_claim_text_first_comparison.lock").open("a") as lock:
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
