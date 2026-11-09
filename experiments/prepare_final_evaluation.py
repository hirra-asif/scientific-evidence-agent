"""Freeze unused dev claims, the unchanged four conditions and analysis rules."""

import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from experiments import agent_comparison as comparison
from experiments.compare_agent_retrieval import PACKAGES, now
from experiments.final_conditions import schedule_runs, summarize_runs, verify_run_config
from experiments.final_evaluation_common import (
    PLAN, OUTPUT, PILOT_PLAN, PILOT_RESULT, SEED, PER_LABEL, REPETITIONS,
    adapt_runner, analysis_policy, audit_history, read_claims, select_claims,
)
from experiments.prepare_evaluation import LABELS, claim_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNNER = "experiments/run_final_pilot.py"
NEW_FILES = ("experiments/final_evaluation_common.py", "experiments/prepare_final_evaluation.py",
             "experiments/run_final_evaluation.py", "experiments/analyze_final_evaluation.py")


def verify_files(fingerprints):
    for name, expected in fingerprints.items():
        if comparison.sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")


def load_dev_claims(sample):
    path = PROJECT_DIR / "data/claims_dev.jsonl"
    if comparison.sha256(path) != sample["dataset_sha256"]:
        raise ValueError("Dev dataset changed after freezing.")
    records = read_claims(path)
    by_id = {row["id"]: row for row in records}
    ids = sample["claim_ids"]
    if sample["split"] != "dev" or sample["repetitions_per_condition"] != REPETITIONS:
        raise ValueError("Expected dev claims and two repetitions per condition.")
    if len(ids) != 60 or len(set(ids)) != 60 or set(ids) - set(by_id):
        raise ValueError("Expected 60 unique known dev claims.")
    if set(ids) & set(sample["excluded_claim_ids"]):
        raise ValueError("Final claims overlap prior recorded/reserved claims.")
    for claim_id in ids:
        if sample["labels"][str(claim_id)] != claim_label(by_id[claim_id]):
            raise ValueError("Frozen labels differ from the dataset.")
    if Counter(sample["labels"].values()) != {label: PER_LABEL for label in LABELS}:
        raise ValueError("Expected exactly 20 claims per class.")
    selected, _ = select_claims(records, set(sample["excluded_claim_ids"]), sample["seed"], PER_LABEL)
    if [row["id"] for row in selected] != ids:
        raise ValueError("Frozen selection does not reproduce from its seed and exclusions.")
    return records, {claim_id: by_id[claim_id] for claim_id in ids}


def verify_plan(plan):
    if (plan["schema_version"] != 1 or plan["experiment"] != "final_dev_evaluation"
            or plan["split"] != "dev"):
        raise ValueError("Expected the frozen final dev evaluation.")
    verify_files(plan["source_sha256"])
    if {name: version(name) for name in PACKAGES} != plan["package_versions"]:
        raise RuntimeError("Dependency versions changed after freezing.")
    if platform.python_version() != plan["python_version"]:
        raise RuntimeError("Python version changed after freezing.")
    pilot = comparison.read_json(PROJECT_DIR / PILOT_PLAN)
    for key in ("conditions", "assessment_guidelines", "assessment_guidelines_sha256",
                "system_prompts", "harness", "scoring_policy", "development_decisions"):
        if plan[key] != pilot[key]:
            raise ValueError(f"Configuration differs from the completed pilot: {key}")
    if plan["execution"] != pilot["execution"]:
        raise ValueError("Execution settings differ from the completed pilot.")
    if plan["analysis_policy"] != analysis_policy():
        raise ValueError("Analysis policy differs from the frozen implementation.")
    if plan["schedule"] != schedule_runs(plan["sample"]["claim_ids"], REPETITIONS):
        raise ValueError("Final schedule changed.")
    load_dev_claims(plan["sample"])


def prepare():
    output = PROJECT_DIR / PLAN
    if output.exists():
        verify_plan(comparison.read_json(output))
        print(f"Existing final freeze verified; no claims reselected: {output}")
        return
    if (PROJECT_DIR / OUTPUT).exists():
        raise FileExistsError("Final results exist without their plan; restore the original plan.")
    pilot = comparison.read_json(PROJECT_DIR / PILOT_RESULT)
    pilot_plan = comparison.read_json(PROJECT_DIR / PILOT_PLAN)
    if (pilot["plan"] != pilot_plan
            or pilot["plan_sha256"] != comparison.sha256(PROJECT_DIR / PILOT_PLAN)):
        raise ValueError("Pilot result and plan do not match.")
    comparison.validate_checkpoint(pilot["runs"], pilot_plan["schedule"])
    if (pilot["status"] != "finished" or len(pilot["runs"]) != 24
            or any(row["status"] != "ok" for row in pilot["runs"])
            or summarize_runs(pilot["runs"], pilot_plan["schedule"]) != pilot["summary"]):
        raise ValueError("The 24-attempt pilot must finish without failures and reproduce.")
    for row in pilot["runs"]:
        verify_run_config(row["metadata"], pilot_plan["conditions"][row["condition"]])
    verify_files({name: value for name, value in pilot_plan["source_sha256"].items() if name != RUNNER})
    packages = {name: version(name) for name in PACKAGES}
    if packages != pilot_plan["package_versions"] or platform.python_version() != pilot_plan["python_version"]:
        raise ValueError("Python or dependency versions changed since the pilot.")
    dev_path = PROJECT_DIR / "data/claims_dev.jsonl"
    dev = read_claims(dev_path)
    train = read_claims(PROJECT_DIR / "data/claims_train.jsonl")
    if {row["id"] for row in dev} & {row["id"] for row in train}:
        raise ValueError("Training and dev claim IDs overlap.")
    dev_hash = comparison.sha256(dev_path)
    for filename in ("experiments/evaluation_claims.json", "experiments/confirmation_claims.json"):
        previous = comparison.read_json(PROJECT_DIR / filename)
        if previous["split"] != "dev" or previous["dataset_sha256"] != dev_hash:
            raise ValueError(f"Dev data differs from the previous evaluation: {filename}")
    audit = audit_history(PROJECT_DIR, dev)
    selected, pools = select_claims(dev, set(audit["excluded_claim_ids"]))
    sample = {
        "split": "dev", "seed": SEED, "claims_per_label": PER_LABEL,
        "repetitions_per_condition": REPETITIONS, "dataset_sha256": dev_hash,
        "claim_ids": [row["id"] for row in selected],
        "labels": {str(row["id"]): claim_label(row) for row in selected},
        "excluded_claim_ids": audit["excluded_claim_ids"],
        "skill_text": pilot_plan["sample"]["skill_text"],
        "skill_sha256": pilot_plan["sample"]["skill_sha256"],
    }
    gold_papers = Counter(doc for row in selected for doc in row["evidence"])
    runner_path = PROJECT_DIR / RUNNER
    original = runner_path.read_text(encoding="utf-8")
    adapted = adapt_runner(original, pilot_plan["source_sha256"][RUNNER])
    try:
        if original != adapted:
            runner_path.write_text(adapted, encoding="utf-8")
        names = set(pilot_plan["source_sha256"]) | set(NEW_FILES) | {
            PILOT_PLAN, PILOT_RESULT, "data/claims_dev.jsonl",
            "experiments/evaluation_claims.json", "experiments/confirmation_claims.json",
        }
        plan = {
            "schema_version": 1, "experiment": "final_dev_evaluation", "created_at": now(),
            "split": "dev", "role": "Frozen final evaluation on previously unused dev claims",
            "sample": sample, "schedule": schedule_runs(sample["claim_ids"], REPETITIONS),
            "schedule_policy": "Rotate conditions within each claim; reverse the base condition order for repetition 2; run every condition 120 times",
            "history_audit": audit,
            "selection_summary": {
                "dev_claims": len(dev), "previously_recorded_or_reserved": len(audit["excluded_claim_ids"]),
                "eligible_pool_counts": pools, "selected_counts": dict(Counter(sample["labels"].values())),
                "previous_claim_overlap": 0, "training_claim_overlap": 0,
                "unique_annotated_papers": len(gold_papers),
                "annotated_papers_shared_by_selected_claims": sum(count > 1 for count in gold_papers.values()),
            },
            "conditions": pilot_plan["conditions"], "system_prompts": pilot_plan["system_prompts"],
            "assessment_guidelines": pilot_plan["assessment_guidelines"],
            "assessment_guidelines_sha256": pilot_plan["assessment_guidelines_sha256"],
            "development_decisions": pilot_plan["development_decisions"],
            "harness": pilot_plan["harness"], "execution": pilot_plan["execution"],
            "scoring_policy": pilot_plan["scoring_policy"], "analysis_policy": analysis_policy(),
            "source_sha256": {name: comparison.sha256(PROJECT_DIR / name) for name in sorted(names)},
            "package_versions": packages, "python_version": platform.python_version(),
            "platform_at_preparation": platform.platform(),
            "commit_policy": "Manifest, source and prior results must match committed HEAD before the first attempt or resume",
        }
        with output.open("x", encoding="utf-8") as file:
            json.dump(plan, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except BaseException:
        if original != adapted:
            runner_path.write_text(original, encoding="utf-8")
        raise
    print(json.dumps(plan["selection_summary"], indent=2))
    print("Frozen: 60 dev claims, two repetitions, four unchanged conditions, 480 attempts.")
    print("Primary comparisons: improvements; pipeline vs improved agent; Sonnet vs improved Haiku.")
    print("Analysis uses paired claim-level bootstrap intervals with both repetitions kept together.")
    print("No model requests were made. Commit the code, pilot result and manifest before running.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    prepare()
