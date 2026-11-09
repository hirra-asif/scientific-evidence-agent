"""Validate completed results and apply the analysis frozen before the dev run."""

import copy
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from experiments import agent_comparison as comparison
from experiments.compare_agent_retrieval import now
from experiments.final_conditions import (
    CONDITIONS, build_pipeline_prompt, digest, summarize_runs, verify_run_config,
)
from experiments.final_evaluation_common import PLAN, OUTPUT, ANALYSIS, REPORT, bootstrap_intervals
from experiments.prepare_final_evaluation import load_dev_claims, verify_plan
from scoring import summarize_scores

PROJECT_DIR = Path(__file__).resolve().parents[1]


def audit_results(results, plan, claims):
    from agent import AgentReport, validate_report
    from corpus_tools import read_abstract
    comparison.validate_checkpoint(results["runs"], plan["schedule"])
    if results["status"] != "finished" or len(results["runs"]) != len(plan["schedule"]):
        raise ValueError("Analysis requires the complete frozen evaluation, including failures.")
    for row in results["runs"]:
        claim = claims[row["claim_id"]]
        if row["claim"] != claim["claim"] or row["gold_evidence"] != claim["evidence"]:
            raise ValueError("Saved claim or gold evidence differs from the frozen dataset.")
        config = plan["conditions"][row["condition"]]
        if row["configuration"] != config:
            raise ValueError("A run used a different condition configuration.")
        recalculated = copy.deepcopy(row)
        comparison.finish_row(recalculated, claim, row["status"], row.get("report"))
        for key in ("score_counts", "abstained", "nei_outcome", "gold_is_nei"):
            if recalculated[key] != row[key]:
                raise ValueError(f"Saved scoring/outcome mismatch: {comparison.run_key(row)}, {key}")
        ids = row["metadata"].get("opened_document_ids", [])
        if len(ids) > 3 or len(ids) != len(set(ids)):
            raise ValueError("Invalid read budget or duplicate opened documents.")
        if row["status"] == "ok":
            verify_run_config(row["metadata"], config)
            documents = {doc_id: read_abstract(doc_id) for doc_id in ids}
            validate_report(AgentReport.model_validate(row["report"]), documents)
            if config["mode"] == "pipeline":
                metadata = row["metadata"]
                trace = metadata["tool_trace"]
                if ([event["tool"] for event in trace] != ["search_abstracts"] + ["read_abstract"] * 3
                        or trace[0]["arguments"]["query"] != claim["claim"]):
                    raise ValueError("Pipeline did not follow its single-query fixed retrieval policy.")
                selected = [str(hit["document_id"]) for hit in trace[0]["results"][:3]]
                if selected != ids or selected != metadata["selected_document_ids"]:
                    raise ValueError("Pipeline did not supply its top three documents.")
                if metadata["supplied_documents"] != [documents[doc_id] for doc_id in ids]:
                    raise ValueError("Pipeline text differs from the corpus.")
                prompt = build_pipeline_prompt(claim["claim"], metadata["supplied_documents"], plan["assessment_guidelines"])
                if digest(prompt) != metadata["assessment_prompt_sha256"]:
                    raise ValueError("Pipeline assessment prompt differs from the frozen rules and evidence.")
    summary = summarize_runs(results["runs"], plan["schedule"])
    if summary != results["summary"] or not summary["complete"]:
        raise ValueError("Saved aggregate scores do not reproduce.")
    return summary


def descriptive_details(runs, summary):
    details = {}
    for name in CONDITIONS:
        rows = [row for row in runs if row["condition"] == name]
        data = summary["conditions"][name]
        timed = [row["elapsed_seconds"] for row in rows if row.get("elapsed_seconds") is not None]
        successful_times = [row["elapsed_seconds"] for row in rows
                            if row["status"] == "ok" and row.get("elapsed_seconds") is not None]
        nei = data["nei_attempts"]
        successful_nei = nei - data["nei_processing_failures"]
        failures = Counter(f"{row['status']} / {row['metadata'].get('stop_reason') or 'unspecified'}"
                           for row in rows if row["status"] != "ok")
        details[name] = {
            "failure_rate": data["failures"] / len(rows), "failure_breakdown": dict(failures),
            "nei_false_flag_rate_all_attempts": data["nei_false_flags"] / nei if nei else None,
            "nei_false_flag_rate_successful_attempts": data["nei_false_flags"] / successful_nei if successful_nei else None,
            "correct_nei_abstention_rate_all_attempts": data["nei_correct_abstentions"] / nei if nei else None,
            "median_elapsed_seconds_all_attempts": statistics.median(timed) if timed else None,
            "p95_elapsed_seconds_all_attempts": float(np.quantile(timed, 0.95)) if timed else None,
            "mean_elapsed_seconds_successful_attempts": statistics.mean(successful_times) if successful_times else None,
            "cost_note": "Sum of available SDK estimates; inspect runs_without_cost before interpreting the total",
        }
    return details


def matched_successful_pairs(runs, contrasts):
    result = {}
    for contrast in contrasts:
        pairs = {}
        for name in (contrast["candidate"], contrast["reference"]):
            pairs[name] = {(row["claim_id"], row["repetition"]): row for row in runs
                           if row["condition"] == name and row["status"] == "ok"}
        keys = set(pairs[contrast["candidate"]]) & set(pairs[contrast["reference"]])
        result[contrast["name"]] = {"successful_pairs": len(keys), "conditions": {
            name: summarize_scores([pairs[name][key]["score_counts"] for key in sorted(keys)]) if keys else None
            for name in pairs}}
    return result


def markdown_report(analysis):
    def number(value, digits=3):
        return f"{value:.{digits}f}" if value is not None else "unavailable"
    lines = [
        "# Frozen dev evaluation", "",
        "60 previously unused dev claims, 20 per class; two repetitions per condition (480 attempts).",
        "Counts are pooled before calculating precision, recall and F1. Failed attempts retain their gold evidence, contribute no positive predictions and earn no correct-abstention credit.",
        "", "## Performance", "",
        "| Condition | Document/verdict P | R | F1 | Evidence P | R | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in analysis["summary"]["conditions"].items():
        metrics = data["metrics_all_attempts"]
        document, evidence = metrics["document_labels"], metrics["evidence_sentences"]
        values = [document[k] for k in ("precision", "recall", "f1")] + [evidence[k] for k in ("precision", "recall", "f1")]
        lines.append("| " + name + " | " + " | ".join(number(v) for v in values) + " |")
    lines += ["", "## Reliability and resource use", "",
              "| Condition | NEI false flags | Correct NEI abstentions | NEI failures | All failures | Mean searches | Mean abstracts | Mean seconds | Reported USD |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in analysis["summary"]["conditions"].items():
        values = [f"{data['nei_false_flags']}/{data['nei_attempts']}",
                  f"{data['nei_correct_abstentions']}/{data['nei_attempts']}", str(data["nei_processing_failures"]),
                  f"{data['failures']}/{data['attempts']}", number(data["mean_searches"], 2),
                  number(data["mean_abstracts_read"], 2), number(data["mean_elapsed_seconds"], 2),
                  number(data["reported_cost_usd"], 4)]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    missing = {name: data["runs_without_cost"] for name, data in analysis["summary"]["conditions"].items()}
    lines += ["", "A false flag is any SUPPORT or CONTRADICT assessment on a gold NEI claim. Failed NEI attempts are counted separately. Latency includes retrieval and the SDK call; shared retriever setup is separate.",
              "", "Attempts without SDK cost estimates: " + ", ".join(f"{name}: {count}" for name, count in missing.items()) + ".",
              "", "## Planned paired comparisons", "",
              "Differences are candidate minus reference. Both repetitions and all four conditions stay together when resampling a claim. The bootstrap resamples within each gold class (50,000 draws; seed 47).",
              "", "| Comparison | Metric | Difference | 95% interval | Adjusted interval |",
              "| --- | --- | ---: | --- | --- |"]
    for contrast in analysis["bootstrap"]["comparisons"].values():
        label = f"{contrast['candidate']} − {contrast['reference']}"
        for metric, values in contrast["metrics"].items():
            intervals = [f"[{number(values[key]['low'])}, {number(values[key]['high'])}]" for key in ("ci95", "adjusted_ci")]
            lines.append(f"| {label} | {metric} | {values['difference']:+.3f} | {intervals[0]} | {intervals[1]} |")
    lines += ["", "The adjusted intervals use 99.167% confidence per endpoint (Bonferroni across three comparisons and two primary metrics). The frozen rule requires both adjusted intervals above zero to claim higher F1 on both primary metrics. Both below zero favors the reference; other outcomes are inconclusive or mixed.", ""]
    for contrast in analysis["bootstrap"]["comparisons"].values():
        lines.append(f"- **{contrast['name']}:** {contrast['conclusion'].replace('_', ' ')}.")
    lines += ["", "These conclusions concern the primary F1 metrics. NEI outcomes, failures and resource use remain descriptive and must also inform any deployment choice.",
              "", "## Limits and reproducibility", ""]
    lines += ["- " + item + "." for item in analysis["analysis_policy"]["limitations"]]
    lines += ["", f"- Result SHA-256: `{analysis['result_sha256']}`",
              f"- Frozen manifest SHA-256: `{analysis['plan_sha256']}`",
              "", "[Frozen manifest](../experiments/final_evaluation.json) · [Raw results](../results/final_evaluation.json) · [Analysis, per-repeat scores and matched-success diagnostics](../results/final_evaluation_analysis.json)", ""]
    return "\n".join(lines)


def main():
    plan_path = PROJECT_DIR / PLAN
    plan = comparison.read_json(plan_path)
    verify_plan(plan)
    result_path = PROJECT_DIR / OUTPUT
    raw = result_path.read_bytes()
    results = json.loads(raw)
    if results["plan"] != plan or results["plan_sha256"] != comparison.sha256(plan_path):
        raise ValueError("Results do not belong to the frozen plan.")
    _, claims = load_dev_claims(plan["sample"])
    summary = audit_results(results, plan, claims)
    print("All 480 attempts and scores verified. Computing frozen paired intervals...", flush=True)
    analysis = {
        "schema_version": 1, "created_at": now(), "split": "dev",
        "plan_sha256": results["plan_sha256"], "result_sha256": hashlib.sha256(raw).hexdigest(),
        "analysis_policy": plan["analysis_policy"], "summary": summary,
        "descriptive_details": descriptive_details(results["runs"], summary),
        "per_repetition": {str(rep): summarize_runs(
            [row for row in results["runs"] if row["repetition"] == rep],
            [job for job in plan["schedule"] if job["repetition"] == rep]) for rep in (1, 2)},
        "matched_successful_pairs": matched_successful_pairs(results["runs"], plan["analysis_policy"]["planned_comparisons"]),
        "bootstrap": bootstrap_intervals(results["runs"], plan["sample"], plan["analysis_policy"]),
    }
    comparison.save_json(PROJECT_DIR / ANALYSIS, analysis)
    report_path = PROJECT_DIR / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(analysis), encoding="utf-8")
    print(json.dumps(analysis["bootstrap"]["comparisons"], indent=2))
    print(f"Saved: {PROJECT_DIR / ANALYSIS}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
