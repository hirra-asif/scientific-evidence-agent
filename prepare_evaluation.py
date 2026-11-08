import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SEED = 42
PER_LABEL = 10
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


def claim_label(claim):
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


def main():
    output_path = PROJECT_DIR / "evaluation_claims.json"

    if output_path.exists():
        raise FileExistsError(
            "evaluation_claims.json already exists. Keep the frozen sample."
        )

    dataset_path = PROJECT_DIR / "data" / "claims_dev.jsonl"
    raw = dataset_path.read_bytes()

    claims = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(SEED)
    selected = []

    for label in LABELS:
        pool = sorted(
            [claim for claim in claims if claim_label(claim) == label],
            key=lambda claim: claim["id"],
        )

        if len(pool) < PER_LABEL:
            raise ValueError(f"Not enough claims for {label}.")

        selected.extend(rng.sample(pool, PER_LABEL))

    rng.shuffle(selected)
    claim_ids = [claim["id"] for claim in selected]

    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("Selected claim IDs are not unique.")

    manifest = {
        "split": "dev",
        "seed": SEED,
        "claims_per_label": PER_LABEL,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "claim_ids": claim_ids,
        "labels": {
            str(claim["id"]): claim_label(claim)
            for claim in selected
        },
    }

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    counts = Counter(claim_label(claim) for claim in selected)

    for label in LABELS:
        print(f"{label}: {counts[label]}")

    print(f"Total: {len(selected)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()