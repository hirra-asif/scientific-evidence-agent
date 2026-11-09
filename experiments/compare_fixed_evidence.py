import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from anthropic import Anthropic

from agent import AgentReport, validate_report
from experiments.compare_skills import load_claims, save_results
from corpus_tools import read_abstract
from experiments.evidence_test import ClaimCheck, MODEL, SYSTEM_PROMPT
from scoring import score_claim, summarize_scores

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Previously inspected development cases: two per verdict class.
# One fixed abstract per claim; this is a diagnostic, not a fresh benchmark.
CASES = {
    137: "26016929",
    148: "1084345",
    216: "21366394",
    692: "24088502",
    800: "4679264",
    1014: "6277638",
}


def make_prompts():
    path = PROJECT_DIR / ".claude/skills/scientific-evidence/SKILL.md"
    raw = path.read_text(encoding="utf-8")
    current = raw.split("---", 2)[2].split("Claim to assess:")[0].strip()

    # Adapt tool instructions identically for both guideline conditions.
    current = current.replace(
        "Use the corpus search and abstract-reading tools to assess the claim.",
        "Use only the supplied abstract to assess the claim.",
    ).replace(
        "abstracts you actually read.",
        "the supplied abstract.",
    )

    old_rule = (
        "Select sufficient evidence sentences, including relevant qualifications."
    )
    new_rule = (
        "Cite the smallest set of sentences that together justify the verdict.\n"
        "   Include a qualifying sentence only if leaving it out would change\n"
        "   or overstate the verdict."
    )
    if current.count(old_rule) != 1:
        raise ValueError("The current evidence rule has changed; review it first.")

    return {
        "off": SYSTEM_PROMPT,
        "current": SYSTEM_PROMPT + "\n\n" + current,
        "minimal": SYSTEM_PROMPT + "\n\n" + current.replace(old_rule, new_rule),
    }


def summarize(runs):
    conditions = ("off", "current", "minimal")
    successful = {
        condition: {
            (r["claim_id"], r["repeat"])
            for r in runs
            if r["condition"] == condition and r["status"] == "ok"
        }
        for condition in conditions
    }
    common = set.intersection(*successful.values())
    summary = {
        "complete_case_repetitions": len(common),
        "scoring_population": "Case-repetitions successful in all conditions",
        "conditions": {},
    }

    for condition in conditions:
        rows = [r for r in runs if r["condition"] == condition]
        scored = [
            r for r in rows
            if (r["claim_id"], r["repeat"]) in common
        ]
        summary["conditions"][condition] = {
            "attempted": len(rows),
            "failures": sum(r["status"] == "error" for r in rows),
            "verdict_accuracy_all_attempts": (
                sum(r.get("verdict_matches", False) for r in rows) / len(rows)
                if rows else None
            ),
            "metrics_on_complete_cases": (
                summarize_scores([r["score_counts"] for r in scored])
                if scored else None
            ),
            "mean_cited_sentences_on_complete_cases": (
                sum(r["citation_count"] for r in scored) / len(scored)
                if scored else None
            ),
        }
    return summary


def main(repeats):
    prompts = make_prompts()
    claims = load_claims("dev", list(CASES))
    cases = []

    for claim in claims:
        doc_id = CASES[claim["id"]]
        if claim["evidence"] and set(claim["evidence"]) != {doc_id}:
            raise ValueError("This diagnostic requires one gold document per claim.")

        document = read_abstract(doc_id)
        abstract = "\n".join(
            f"[{s['sentence_id']}] {s['text']}"
            for s in document["sentences"]
        )
        expected = (
            claim["evidence"][doc_id][0]["label"]
            if claim["evidence"] else "NOT_ENOUGH_INFO"
        )
        cases.append({
            "claim": claim,
            "document": document,
            "expected_verdict": expected,
            "prompt": f"CLAIM:\n{claim['claim']}\n\nABSTRACT:\n{abstract}",
        })

    started = datetime.now(timezone.utc)
    output = PROJECT_DIR / "results" / (
        f"fixed_evidence_{started:%Y%m%d_%H%M%S_%f}.json"
    )
    output.parent.mkdir(exist_ok=True)
    source_files = [
        "experiments/compare_fixed_evidence.py", "agent.py", "experiments/compare_skills.py",
        "scoring.py", "corpus_tools.py", "experiments/evidence_test.py",
        ".claude/skills/scientific-evidence/SKILL.md",
    ]

    results = {
        "status": "running",
        "experiment": "Fixed-abstract guideline-text comparison; no agent tools",
        "started_at": started.isoformat(),
        "model": MODEL,
        "repeats": repeats,
        "max_tokens": 2048,
        "max_retries": 2,
        "timeout_seconds": 120,
        "system_prompts": prompts,
        "cases": cases,
        "package_versions": {
            name: version(name)
            for name in ("anthropic", "claude-agent-sdk", "pydantic")
        },
        "source_sha256": {
            name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "runs": [],
    }
    save_results(output, results)
    print(f"Saving results to: {output}", flush=True)

    with Anthropic(max_retries=2, timeout=120.0) as client:
        for repeat in range(repeats):
            for index, case in enumerate(cases):
                order = list(prompts)
                offset = (repeat + index) % len(order)
                order = order[offset:] + order[:offset]

                for condition in order:
                    claim = case["claim"]
                    document = case["document"]
                    row = {
                        "claim_id": claim["id"],
                        "repeat": repeat + 1,
                        "condition": condition,
                    }
                    print(
                        f"Claim {claim['id']} | repeat {repeat + 1}/{repeats}"
                        f" | guidelines {condition}",
                        flush=True,
                    )
                    start = time.perf_counter()

                    try:
                        response = client.messages.parse(
                            model=MODEL,
                            max_tokens=2048,
                            output_format=ClaimCheck,
                            system=prompts[condition],
                            messages=[{"role": "user", "content": case["prompt"]}],
                        )
                        row["raw_response"] = response.model_dump(mode="json")
                        answer = response.parsed_output
                        if answer is None:
                            raise RuntimeError(
                                f"No structured answer: {response.stop_reason}"
                            )

                        report = AgentReport.model_validate({
                            "assessments": [{
                                "document_id": document["document_id"],
                                **answer.model_dump(),
                            }]
                        })
                        validate_report(report, {document["document_id"]: document})
                        row["report"] = report.model_dump()
                        row["score_counts"] = dict(score_claim(claim, row["report"]))
                        row["citation_count"] = len(answer.evidence_sentence_ids)
                        row["verdict_matches"] = (
                            answer.verdict == case["expected_verdict"]
                        )
                        row["status"] = "ok"

                    except Exception as error:
                        row["status"] = "error"
                        row["error"] = f"{type(error).__name__}: {error}"
                        print(f"RUN FAILED: {row['error']}", flush=True)

                    row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
                    results["runs"].append(row)
                    results["summary"] = summarize(results["runs"])
                    save_results(output, results)

    results["status"] = "finished"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(output, results)
    print("\nFIXED-EVIDENCE SUMMARY:")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSaved results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    main(args.repeats)