import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from agent import main as check_claim

PROJECT_DIR = Path(__file__).resolve().parent
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
PRACTICE_CLAIM = "1 in 5 million in UK have abnormal PrP positivity."


def dataset_label(claim):
    labels = {
        evidence["label"]
        for evidence_sets in claim["evidence"].values()
        for evidence in evidence_sets
    }
    if not labels:
        return "NOT_ENOUGH_INFO"
    if len(labels) == 1:
        return next(iter(labels))
    return "MIXED"


def select_claims(claims):
    rng = random.Random(42)
    selected = []

    for label in LABELS:
        pool = [
            claim for claim in claims
            if dataset_label(claim) == label
            and claim["claim"] != PRACTICE_CLAIM
        ]
        selected.extend(rng.sample(pool, 2))

    return selected


def compare_report(claim, report):
    gold = claim["evidence"]

    ids = [item.document_id for item in report.assessments]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document assessments.")

    # NEI assessments make no positive evidence claim.
    predictions = {
        item.document_id: item
        for item in report.assessments
        if item.verdict != "NOT_ENOUGH_INFO"
    }

    # Require all annotated papers, correct labels,
    # and no extra positive papers.
    papers_and_labels_match = set(predictions) == set(gold) and all(
        any(
            predictions[doc_id].verdict == evidence["label"]
            for evidence in evidence_sets
        )
        for doc_id, evidence_sets in gold.items()
    )

    # Accept any alternative annotated evidence set for each paper.
    exact_evidence_match = papers_and_labels_match and all(
        any(
            predictions[doc_id].verdict == evidence["label"]
            and set(predictions[doc_id].evidence_sentence_ids)
            == set(evidence["sentences"])
            for evidence in evidence_sets
        )
        for doc_id, evidence_sets in gold.items()
    )

    return {
        "papers_and_labels_match": papers_and_labels_match,
        "exact_evidence_match": exact_evidence_match,
    }


async def evaluate():
    claims_path = PROJECT_DIR / "data" / "claims_train.jsonl"

    with claims_path.open(encoding="utf-8") as file:
        claims = [json.loads(line) for line in file if line.strip()]

    selected = select_claims(claims)
    results = []

    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"basic_eval_{timestamp}.json"

    for number, claim in enumerate(selected, start=1):
        print(f"\n===== CLAIM {number}/{len(selected)} =====", flush=True)
        print(claim["claim"], flush=True)

        row = {
            "claim_id": claim["id"],
            "claim": claim["claim"],
            "dataset_label": dataset_label(claim),
            "gold_evidence": claim["evidence"],
        }

        try:
            # Only claim text goes to Claude, not the dataset answers.
            report = await check_claim(claim["claim"])

            row.update(compare_report(claim, report))
            row["report"] = report.model_dump()
            row["status"] = "success"

        except Exception as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"RUN FAILED: {row['error']}", flush=True)

        results.append(row)

        # Save after each claim so completed results are retained.
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    total = len(results)

    correct_labels = sum(
        row.get("papers_and_labels_match", False)
        for row in results
    )
    exact_evidence = sum(
        row.get("exact_evidence_match", False)
        for row in results
    )
    failed = sum(
        row["status"] == "error"
        for row in results
    )

    print("\nBASIC EVALUATION SUMMARY")
    print(f"Correct papers and labels: {correct_labels}/{total}")
    print(f"Exact evidence match too: {exact_evidence}/{total}")
    print(f"Processing failures: {failed}/{total}")
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    asyncio.run(evaluate())