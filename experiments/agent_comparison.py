"""Scoring, checkpoints and the small per-run search adapter used in comparisons."""

import hashlib
import json
from pathlib import Path

from scoring import score_claim, summarize_scores

CONDITIONS = ("current_haiku", "hybrid_only_haiku")
TERMINAL = {"ok", "error", "timeout", "interrupted"}
PATCHES = (
    (
        "    def __init__(self):\n        self.opened_documents = {}\n",
        "    def __init__(self, search_fn=None):\n"
        "        self.search_fn = search_abstracts if search_fn is None else search_fn\n"
        "        self.opened_documents = {}\n",
    ),
    (
        '            results = search_abstracts(args["query"])\n',
        '            results = self.search_fn(args["query"])\n',
    ),
    (
        "    skill_workspace=None,\n):\n    state = RunState()\n",
        "    skill_workspace=None,\n    search_fn=None,\n):\n"
        "    state = RunState(search_fn=search_fn)\n",
    ),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def adapt_agent(source, frozen_hash):
    """Permit only the three documented edits to the frozen baseline source."""
    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    original = source
    if digest(source) != frozen_hash:
        for old, new in PATCHES:
            if original.count(new) != 1:
                raise ValueError("agent.py differs from the frozen baseline and expected adapter.")
            original = original.replace(new, old, 1)
        if digest(original) != frozen_hash:
            raise ValueError("agent.py contains changes beyond the per-run search adapter.")
    adapted = original
    for old, new in PATCHES:
        if adapted.count(old) != 1:
            raise ValueError("Could not locate the expected agent code for the search adapter.")
        adapted = adapted.replace(old, new, 1)
    return adapted


def schedule_runs(claim_ids, repetitions):
    schedule = []
    for repetition in range(1, repetitions + 1):
        for index, claim_id in enumerate(claim_ids):
            order = CONDITIONS if (index + repetition - 1) % 2 == 0 else CONDITIONS[::-1]
            for condition in order:
                schedule.append({"claim_id": claim_id, "repetition": repetition,
                                 "condition": condition})
    return schedule


def run_key(row):
    return row["claim_id"], row["repetition"], row["condition"]


def failed_scores(claim):
    counts = dict(score_claim(claim, {"assessments": []}))
    # Missing output supplies no positive prediction, but earns no NEI credit.
    counts["correct_nei"] = 0
    return counts


def finish_row(row, claim, status, report=None):
    row["status"] = status
    row["gold_is_nei"] = not bool(claim["evidence"])
    row["nei_outcome"] = None
    if status == "ok":
        row["report"] = report
        row["score_counts"] = dict(score_claim(claim, report))
        positives = any(item["verdict"] != "NOT_ENOUGH_INFO" for item in report["assessments"])
        row["abstained"] = not positives
        if row["gold_is_nei"]:
            row["nei_outcome"] = "false_flag" if positives else "correct_abstention"
    else:
        row["score_counts"] = failed_scores(claim)
        row["abstained"] = False
        if row["gold_is_nei"]:
            row["nei_outcome"] = "processing_failure"


def validate_checkpoint(runs, schedule):
    if len(runs) > len(schedule):
        raise ValueError("Checkpoint has more attempts than the frozen schedule.")
    for index, row in enumerate(runs):
        if run_key(row) != run_key(schedule[index]):
            raise ValueError("Checkpoint does not follow the frozen schedule.")
        if row["status"] not in TERMINAL | {"running"}:
            raise ValueError("Unknown checkpoint run status.")
        if row["status"] == "running" and index != len(runs) - 1:
            raise ValueError("Only the last checkpoint row may be unfinished.")


def recover_interrupted(runs, claims_by_id):
    for row in runs:
        if row["status"] == "running":
            finish_row(row, claims_by_id[row["claim_id"]], "interrupted")
            row["error"] = "Previous process stopped before this outcome was saved; not retried."
            row["elapsed_seconds"] = None
            row["trace_complete"] = False


def summarize(runs, schedule):
    terminal = [row for row in runs if row["status"] in TERMINAL]
    success = {
        condition: {(r["claim_id"], r["repetition"]) for r in terminal
                    if r["condition"] == condition and r["status"] == "ok"}
        for condition in CONDITIONS
    }
    paired = success[CONDITIONS[0]] & success[CONDITIONS[1]]
    complete = len(terminal) == len(schedule)
    summary = {"planned_runs": len(schedule), "completed_runs": len(terminal),
               "complete": complete, "successful_pairs": len(paired), "conditions": {}}

    def aggregate(rows):
        metrics = summarize_scores([row["score_counts"] for row in rows]) if rows else None
        nei = [row for row in rows if row["gold_is_nei"]]
        false_flags = sum(row["nei_outcome"] == "false_flag" for row in nei)
        nei_failures = sum(row["nei_outcome"] == "processing_failure" for row in nei)
        successful_nei = len(nei) - nei_failures
        return {
            "attempts": len(rows), "metrics_all_attempts": metrics,
            "failures": sum(row["status"] != "ok" for row in rows),
            "nei_attempts": len(nei), "nei_false_flags": false_flags,
            "nei_correct_abstentions": sum(r["nei_outcome"] == "correct_abstention" for r in nei),
            "nei_processing_failures": nei_failures,
            "nei_false_flag_rate_successful_runs": false_flags / successful_nei if successful_nei else None,
        }

    for condition in CONDITIONS:
        rows = [row for row in terminal if row["condition"] == condition]
        data = aggregate(rows)
        matched = [r for r in rows if (r["claim_id"], r["repetition"]) in paired]
        data["metrics_on_successful_pairs"] = (
            summarize_scores([r["score_counts"] for r in matched]) if matched else None
        )
        data["per_repetition"] = {
            str(repetition): aggregate([r for r in rows if r["repetition"] == repetition])
            for repetition in sorted({item["repetition"] for item in schedule})
        }
        elapsed = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None]
        costs = [r["metadata"]["estimated_cost_usd"] for r in rows
                 if r["metadata"].get("estimated_cost_usd") is not None]
        traced = [r for r in rows if r.get("trace_complete", False)]
        data.update({
            "mean_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
            "runs_without_elapsed_time": len(rows) - len(elapsed),
            "reported_cost_usd": sum(costs) if costs else None,
            "runs_without_cost": len(rows) - len(costs),
            "mean_searches_observed": sum(
                sum(e["tool"] == "search_abstracts" for e in r["metadata"].get("tool_trace", []))
                for r in traced) / len(traced) if traced else None,
            "mean_abstracts_read_observed": sum(
                len(r["metadata"].get("opened_document_ids", [])) for r in traced
            ) / len(traced) if traced else None,
            "runs_with_complete_trace": len(traced),
        })
        summary["conditions"][condition] = data
    summary["training_decision"] = {"ready": complete, "keep_hybrid_for_combination": None}
    if complete:
        baseline, hybrid = (summary["conditions"][name] for name in CONDITIONS)
        a, b = baseline["metrics_all_attempts"], hybrid["metrics_all_attempts"]
        checks = {
            "higher_evidence_f1": b["evidence_sentences"]["f1"] > a["evidence_sentences"]["f1"],
            "no_lower_document_verdict_f1": b["document_labels"]["f1"] >= a["document_labels"]["f1"],
            "no_more_nei_false_flags": hybrid["nei_false_flags"] <= baseline["nei_false_flags"],
            "no_more_processing_failures": hybrid["failures"] <= baseline["failures"],
        }
        summary["training_decision"].update(checks=checks, keep_hybrid_for_combination=all(checks.values()))
    return summary
