import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.prepare_evaluation import claim_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED = 44


def paper_groups(claims):
    """Keep claims linked by annotated papers in the same group."""
    by_id = {claim["id"]: claim for claim in claims}
    paper_to_claims = defaultdict(set)

    for claim in claims:
        for document_id in claim["evidence"]:
            paper_to_claims[document_id].add(claim["id"])

    remaining = set(by_id)
    groups = []

    while remaining:
        first = min(remaining)
        remaining.remove(first)
        pending = [first]
        group = []

        while pending:
            claim_id = pending.pop()
            group.append(claim_id)

            for document_id in by_id[claim_id]["evidence"]:
                for neighbour in paper_to_claims[document_id]:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        pending.append(neighbour)

        groups.append(sorted(group))

    return groups


def main():
    output = PROJECT_DIR / "experiments/retrieval_split.json"

    if output.exists():
        raise FileExistsError(
            "retrieval_split.json already exists. Keep the frozen split."
        )

    raw_claims = (PROJECT_DIR / "data/claims_train.jsonl").read_bytes()
    raw_corpus = (PROJECT_DIR / "data/corpus.jsonl").read_bytes()

    claims = [
        json.loads(line)
        for line in raw_claims.splitlines()
        if line.strip()
    ]
    corpus_ids = {
        str(json.loads(line)["doc_id"])
        for line in raw_corpus.splitlines()
        if line.strip()
    }

    if len({claim["id"] for claim in claims}) != len(claims):
        raise ValueError("Duplicate training claim IDs.")

    eligible = [claim for claim in claims if claim["evidence"]]
    by_id = {claim["id"]: claim for claim in eligible}
    gold_ids = {
        doc for claim in eligible for doc in claim["evidence"]
    }

    missing = gold_ids - corpus_ids
    if missing:
        raise ValueError(
            f"Annotated papers missing from corpus: {sorted(missing)}"
        )

    groups = paper_groups(eligible)
    if len(groups) < 2:
        raise ValueError(
            "Need at least two independent paper groups to split."
        )

    rng = random.Random(SEED)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    tuning, validation = [], []

    for group in groups:
        if len(tuning) == len(validation):
            destination = rng.choice([tuning, validation])
        else:
            destination = (
                tuning if len(tuning) < len(validation) else validation
            )
        destination.extend(group)

    tuning.sort()
    validation.sort()

    def papers(ids):
        return {
            doc
            for claim_id in ids
            for doc in by_id[claim_id]["evidence"]
        }

    if set(tuning) & set(validation):
        raise ValueError("Claim overlap between halves.")

    if set(tuning) | set(validation) != set(by_id):
        raise ValueError(
            "The split does not cover every annotated training claim."
        )

    if papers(tuning) & papers(validation):
        raise ValueError("Annotated paper overlap between halves.")

    def describe(ids):
        labels = Counter(claim_label(by_id[i]) for i in ids)
        return {
            "claims": len(ids),
            "unique_annotated_papers": len(papers(ids)),
            "claim_paper_pairs": sum(
                len(by_id[i]["evidence"]) for i in ids
            ),
            "labels": dict(sorted(labels.items())),
        }

    summary = {
        "total_training_claims": len(claims),
        "annotated_training_claims": len(eligible),
        "excluded_without_annotated_evidence": len(claims) - len(eligible),
        "paper_groups": len(groups),
        "largest_group_claims": max(map(len, groups)),
        "tuning": describe(tuning),
        "validation": describe(validation),
        "shared_annotated_papers": 0,
    }

    manifest = {
        "schema_version": 1,
        "experiment": "Offline BM25 versus hybrid retrieval comparison",
        "split": "train",
        "seed": SEED,
        "selection": "All training claims with annotated evidence",
        "grouping": "Connected components of claims sharing annotated papers",
        "assignment": (
            "Seeded group shuffle, largest groups first, balance claim counts"
        ),
        "dataset_sha256": hashlib.sha256(raw_claims).hexdigest(),
        "corpus_sha256": hashlib.sha256(raw_corpus).hexdigest(),
        "tuning_claim_ids": tuning,
        "validation_claim_ids": validation,
        "summary": summary,
    }

    with output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()