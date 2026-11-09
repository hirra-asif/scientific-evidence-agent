"""Record development decisions and freeze a training-only four-condition pilot."""

import copy
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from experiments import agent_comparison as comparison
from experiments.compare_agent_retrieval import PACKAGES, load_claims, now
from experiments.compare_claim_text_first import summarize_results as prompt_summary
from experiments.final_conditions import (
    CONDITIONS, HAIKU, SONNET, adapt_agent, agent_system_prompt, assessment_guidelines,
    choose_pilot_claims, configurations, digest, pipeline_system_prompt, schedule_runs,
)
from experiments.prepare_agent_training import baseline_options

PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_PLAN = "experiments/agent_claim_text_first_comparison.json"
PARENT_RESULT = "results/agent_claim_text_first_training.json"
HYBRID_RESULT = "results/agent_retrieval_training.json"
SAMPLE = "experiments/agent_training_claims.json"
PLAN = "experiments/final_conditions_pilot.json"
OUTPUT = "results/final_conditions_pilot.json"
NEW_FILES = ("experiments/final_conditions.py", "experiments/prepare_final_pilot.py",
             "experiments/run_final_pilot.py")


def verify_files(fingerprints):
    for name, expected in fingerprints.items():
        if comparison.sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")


def development_decisions(hybrid, claim_first):
    if (hybrid["summary"]["training_decision"]["keep_hybrid_for_combination"] is not True
            or claim_first["summary"]["training_decision"]["keep_claim_text_first_for_combination"] is not False):
        raise ValueError("Training decisions differ from the selected four-condition design.")
    found_unread = 0
    hybrid_success_missing_gold = 0
    for result in (hybrid, claim_first):
        for row in result["runs"]:
            gold = set(row["gold_evidence"])
            opened = set(row["metadata"].get("opened_document_ids", []))
            found = {str(hit["document_id"])
                     for event in row["metadata"].get("tool_trace", [])
                     if event["tool"] == "search_abstracts" and event.get("status") == "ok"
                     for hit in event.get("results", [])}
            found_unread += len((gold & found) - opened)
            if row["condition"] == "hybrid_only_haiku" and row["status"] == "ok":
                hybrid_success_missing_gold += len(gold - opened)
    if found_unread != 0 or hybrid_success_missing_gold != 0:
        raise ValueError("Read-when-unsure diagnostic differs from the recorded skip rationale.")
    return {
        "hybrid_retrieval": {"decision": "retain", "reason": "Passed its frozen training keep rule"},
        "claim_text_first": {"decision": "exclude",
                             "reason": "F1 increased, but NEI false flags rose from 8 to 9; failed its frozen rule"},
        "read_when_unsure": {
            "decision": "skip_without_prompt_trial",
            "reason": "Its targeted retrieval failure was absent in the available training traces",
            "gold_papers_found_but_unread_across_both_comparisons": found_unread,
            "gold_papers_unread_on_successful_hybrid_attempts": hybrid_success_missing_gold,
            "limit": "Diagnostic on development data; no claim that the instruction could never help",
        },
        "combination": {"decision": "not_required", "reason": "Hybrid is the sole retained change"},
    }


def prepare():
    output = PROJECT_DIR / PLAN
    if output.exists():
        verify_plan(comparison.read_json(output))
        print(f"Existing pilot plan verified: {output}")
        return
    if (PROJECT_DIR / OUTPUT).exists():
        raise FileExistsError("Pilot result exists without its plan; restore the original plan.")
    parent = comparison.read_json(PROJECT_DIR / PARENT_RESULT)
    parent_plan = comparison.read_json(PROJECT_DIR / PARENT_PLAN)
    if (parent["plan"] != parent_plan
            or parent["plan_sha256"] != comparison.sha256(PROJECT_DIR / PARENT_PLAN)):
        raise ValueError("Claim-text-first result and plan do not match.")
    comparison.validate_checkpoint(parent["runs"], parent_plan["schedule"])
    if (parent["status"] != "finished" or not parent["summary"]["complete"]
            or prompt_summary(copy.deepcopy(parent)) != parent["summary"]):
        raise ValueError("Completed claim-text-first result does not reproduce.")
    verify_files({name: value for name, value in parent_plan["source_sha256"].items()
                  if name != "agent.py"})
    hybrid = comparison.read_json(PROJECT_DIR / HYBRID_RESULT)
    comparison.validate_checkpoint(hybrid["runs"], hybrid["plan"]["schedule"])
    if (hybrid["status"] != "finished" or not hybrid["summary"]["complete"]
            or comparison.summarize(hybrid["runs"], hybrid["plan"]["schedule"]) != hybrid["summary"]):
        raise ValueError("Completed hybrid result does not reproduce.")
    decisions = development_decisions(hybrid, parent)
    sample = parent_plan["sample"]
    if sample != comparison.read_json(PROJECT_DIR / SAMPLE):
        raise ValueError("Frozen training sample changed.")
    load_claims(sample)
    packages = {name: version(name) for name in PACKAGES}
    if packages != parent_plan["package_versions"] or platform.python_version() != parent_plan["python_version"]:
        raise ValueError("Python or dependency versions changed since the training comparisons.")
    if sample["shared_agent_options"]["model"] != HAIKU:
        raise ValueError("The baseline model differs from the selected Haiku ID.")
    if digest(sample["skill_text"]) != sample["skill_sha256"]:
        raise ValueError("Frozen Skill text and hash disagree.")
    agent_path = PROJECT_DIR / "agent.py"
    original = agent_path.read_text(encoding="utf-8")
    adapted = adapt_agent(original, parent_plan["source_sha256"]["agent.py"])
    defaults = baseline_options(adapted)
    if any(defaults[key] != sample["shared_agent_options"][key] for key in defaults):
        raise ValueError("Adapter changed the baseline defaults.")
    guidelines = assessment_guidelines(sample["skill_text"])
    agent_prompt = agent_system_prompt(adapted)
    pipeline_prompt = pipeline_system_prompt(agent_prompt)
    selected = choose_pilot_claims(sample)
    weight = sample["first_comparison"]["hybrid_only_haiku"]["dense_weight"]
    try:
        if original != adapted:
            agent_path.write_text(adapted, encoding="utf-8")
        names = set(parent_plan["source_sha256"]) | set(NEW_FILES) | {PARENT_PLAN, PARENT_RESULT}
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                              capture_output=True, text=True, check=False)
        plan = {
            "schema_version": 1, "experiment": "four_condition_training_pilot",
            "created_at": now(), "split": "train", "role": "Technical pilot, not final model selection",
            "sample": sample, "claim_ids": selected, "repetitions": 1,
            "selection_rule": "First two claims of each class in the frozen training sample order",
            "conditions": configurations(weight), "schedule": schedule_runs(selected),
            "development_decisions": decisions,
            "assessment_guidelines": guidelines, "assessment_guidelines_sha256": digest(guidelines),
            "system_prompts": {"agent": agent_prompt, "pipeline": pipeline_prompt},
            "source_sha256": {name: comparison.sha256(PROJECT_DIR / name) for name in sorted(names)},
            "package_versions": packages, "python_version": platform.python_version(),
            "platform_at_preparation": platform.platform(),
            "git_head_before_pilot_commit": head.stdout.strip() if head.returncode == 0 else None,
            "execution": {
                "timeout_seconds_per_attempt": 300, "concurrency": 1,
                "pause_after_consecutive_failures": 3,
                "condition_order": "Rotate four conditions by claim index",
                "latency_scope": "Full attempt, including fixed retrieval or agent tools and the SDK call",
                "setup_scope": "Shared retriever initialization recorded separately; excluded from per-attempt latency",
                "failure_policy": "Keep errors, timeouts and interrupted attempts; no full-attempt retries",
                "model_effort_policy": "SDK/model defaults; no explicit effort or thinking override",
                "model_fallback": "None",
            },
            "harness": {
                "all_conditions": "claude_agent_sdk.query with JSON Schema and agent.validate_report",
                "agent_guidelines": "Unchanged frozen scientific-evidence Skill",
                "pipeline_guidelines": "Exact assessment section from that Skill, supplied in the prompt",
                "pipeline_retrieval": "One complete-claim hybrid search; read the top three abstracts in order",
                "pipeline_tools": "No built-in or MCP action tools; SDK StructuredOutput may carry the response",
                "pipeline_workspace": "Empty temporary project; no Skill tool",
                "agent_workspace": "Temporary project containing only the frozen Skill",
                "max_abstracts": 3,
            },
            "scoring_policy": parent_plan["scoring_policy"],
            "pilot_rule": "All 24 attempts must complete with validated reports before freezing dev; inspect failures without tuning to pilot F1",
            "next_evaluation": {
                "split": "dev", "unused_claims": 60, "per_class": 20,
                "repetitions_per_condition": 2, "total_attempts": 480,
                "status": "Not selected or run by this pilot",
            },
            "model_id_reference": "https://platform.claude.com/docs/en/models/overview",
        }
        with output.open("x", encoding="utf-8") as file:
            json.dump(plan, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except BaseException:
        if original != adapted:
            agent_path.write_text(original, encoding="utf-8")
        raise
    print(json.dumps({"pilot_claim_ids": selected, "split": "train", "attempts": 24,
                      "conditions": plan["conditions"], "development_decisions": decisions}, indent=2))
    print("Added per-run model selection and usage logging; baseline defaults are unchanged.")
    print("No dev claims were selected or read. Commit the pilot code and plan before running.")
    print(f"Saved: {output}")


def verify_plan(plan):
    if (plan["schema_version"] != 1 or plan["experiment"] != "four_condition_training_pilot"
            or plan["split"] != "train" or plan["repetitions"] != 1):
        raise ValueError("Expected the training-only pilot plan.")
    verify_files(plan["source_sha256"])
    if {name: version(name) for name in PACKAGES} != plan["package_versions"]:
        raise RuntimeError("Dependency versions changed after preparation.")
    if platform.python_version() != plan["python_version"]:
        raise RuntimeError("Python version changed after preparation.")
    sample = plan["sample"]
    if sample != comparison.read_json(PROJECT_DIR / SAMPLE):
        raise ValueError("Frozen training sample changed.")
    if plan["claim_ids"] != choose_pilot_claims(sample) or plan["schedule"] != schedule_runs(plan["claim_ids"]):
        raise ValueError("Pilot claims or schedule changed.")
    weight = sample["first_comparison"]["hybrid_only_haiku"]["dense_weight"]
    if plan["conditions"] != configurations(weight):
        raise ValueError("Condition configurations changed.")
    guidelines = assessment_guidelines(sample["skill_text"])
    if plan["assessment_guidelines"] != guidelines or plan["assessment_guidelines_sha256"] != digest(guidelines):
        raise ValueError("Assessment guidelines differ between conditions.")
    agent_prompt = agent_system_prompt((PROJECT_DIR / "agent.py").read_text(encoding="utf-8"))
    if plan["system_prompts"] != {"agent": agent_prompt, "pipeline": pipeline_system_prompt(agent_prompt)}:
        raise ValueError("System prompts changed.")


if __name__ == "__main__":
    prepare()
