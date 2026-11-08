"""SciFact scoring for validated reports.

Reference:
https://github.com/allenai/scifact-evaluator/blob/master/evaluator/eval.py
"""

from collections import Counter


def precision_recall_f1(correct, predicted, gold):
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = (
        2 * correct / (predicted + gold)
        if predicted + gold else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_claim(claim, report):
    """Accept a dataset claim and a report.model_dump() dictionary."""
    gold = claim["evidence"]
    assessments = report["assessments"]

    doc_ids = [item["document_id"] for item in assessments]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate document assessments.")

    for item in assessments:
        ids = item["evidence_sentence_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence sentence IDs.")

    predictions = {
        item["document_id"]: item
        for item in assessments
        if item["verdict"] != "NOT_ENOUGH_INFO"
    }

    counts = Counter({
        "claims": 1,
        "gold_documents": len(gold),
        "predicted_documents": len(predictions),
        "correct_document_labels": 0,
        "correct_abstracts_with_evidence": 0,
        "gold_sentences": sum(
            len(evidence["sentences"])
            for evidence_sets in gold.values()
            for evidence in evidence_sets
        ),
        "predicted_sentences": sum(
            len(item["evidence_sentence_ids"])
            for item in predictions.values()
        ),
        "correct_sentences": 0,
        "nei_claims": int(not gold),
        "correct_nei": int(not gold and not predictions),
    })

    for doc_id, prediction in predictions.items():
        matching_sets = [
            set(evidence["sentences"])
            for evidence in gold.get(doc_id, [])
            if evidence["label"] == prediction["verdict"]
        ]

        if not matching_sets:
            continue

        counts["correct_document_labels"] += 1
        sentence_ids = prediction["evidence_sentence_ids"]

        # Abstract scoring normally considers the first three sentences.
        # The reference scorer increases this limit when even the
        # shortest gold rationale requires more than three sentences.
        limit = max(3, min(len(s) for s in matching_sets))
        first_sentences = set(sentence_ids[:limit])

        if any(
            s.issubset(first_sentences)
            for s in matching_sets
        ):
            counts["correct_abstracts_with_evidence"] += 1

        # Sentence scoring considers every predicted sentence.
        # Credit requires a complete rationale and the correct label.
        predicted_set = set(sentence_ids)
        credited = set()

        for evidence_set in matching_sets:
            if evidence_set.issubset(predicted_set):
                credited.update(evidence_set)

        counts["correct_sentences"] += len(credited)

    return counts


def summarize_scores(scores):
    """Pool counts across claims before calculating the metrics."""
    total = Counter()

    for score in scores:
        total.update(score)

    return {
        "document_labels": precision_recall_f1(
            total["correct_document_labels"],
            total["predicted_documents"],
            total["gold_documents"],
        ),
        "abstract_with_evidence": precision_recall_f1(
            total["correct_abstracts_with_evidence"],
            total["predicted_documents"],
            total["gold_documents"],
        ),
        "evidence_sentences": precision_recall_f1(
            total["correct_sentences"],
            total["predicted_sentences"],
            total["gold_sentences"],
        ),
        "nei_abstention_rate": (
            total["correct_nei"] / total["nei_claims"]
            if total["nei_claims"] else None
        ),
        "counts": dict(total),
    }


if __name__ == "__main__":
    # Local examples using the sentence selections you just observed.
    # This does not call Claude.
    example_claim = {
        "evidence": {
            "13734012": [
                {"label": "CONTRADICT", "sentences": [4]}
            ]
        }
    }

    for name, sentence_ids in [
        ("Skill off example", [4]),
        ("Skill on example", [3, 4]),
    ]:
        example_report = {
            "assessments": [{
                "document_id": "13734012",
                "verdict": "CONTRADICT",
                "evidence_sentence_ids": sentence_ids,
            }]
        }

        metrics = summarize_scores([
            score_claim(example_claim, example_report)
        ])
        evidence = metrics["evidence_sentences"]

        print(
            f"{name}: "
            f"precision={evidence['precision']:.3f}, "
            f"recall={evidence['recall']:.3f}, "
            f"F1={evidence['f1']:.3f}"
        )