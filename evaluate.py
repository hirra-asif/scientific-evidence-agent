import asyncio
import json
from pathlib import Path

from agent import main as check_claim

PROJECT_DIR = Path(__file__).resolve().parent


async def evaluate_one():
    # Load the dataset answers for comparison.
    claims_path = PROJECT_DIR / "data" / "claims_train.jsonl"

    with claims_path.open(encoding="utf-8") as file:
        claims = [json.loads(line) for line in file if line.strip()]

    # Start with the first claim that has annotated evidence.
    claim = next(item for item in claims if item["evidence"])

    print("EVALUATING CLAIM:")
    print(claim["claim"])

    # Only the claim text goes to the agent.
    report = await check_claim(claim["claim"])

    predictions = {
        item.document_id: item
        for item in report.assessments
    }

    print("\nEVALUATION CHECK:")

    for document_id, gold_sets in claim["evidence"].items():
        expected_labels = sorted({
            evidence["label"] for evidence in gold_sets
        })
        expected_sentences = [
            evidence["sentences"] for evidence in gold_sets
        ]

        print(f"\nDocument: {document_id}")
        print(f"Dataset labels: {expected_labels}")
        print(f"Dataset evidence sets: {expected_sentences}")

        prediction = predictions.get(document_id)

        if prediction is None:
            print("MISSED: agent did not return this document.")
            continue

        label_matches = any(
            prediction.verdict == evidence["label"]
            for evidence in gold_sets
        )

        exact_match = any(
            prediction.verdict == evidence["label"]
            and set(prediction.evidence_sentence_ids)
            == set(evidence["sentences"])
            for evidence in gold_sets
        )

        print(f"Agent label: {prediction.verdict}")
        print(f"Agent sentences: {prediction.evidence_sentence_ids}")
        print(f"Label matches: {label_matches}")
        print(f"Label and exact evidence match: {exact_match}")

    extra_documents = [
        item.document_id
        for item in report.assessments
        if item.verdict != "NOT_ENOUGH_INFO"
        and item.document_id not in claim["evidence"]
    ]
    print(f"\nExtra positive documents outside dataset evidence: {extra_documents}")


if __name__ == "__main__":
    asyncio.run(evaluate_one())