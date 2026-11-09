import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from experiments.prepare_evaluation import claim_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED = 43
PER_LABEL = 10
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


def main():
    output = PROJECT_DIR / "experiments/confirmation_claims.json"
    if output.exists():
        raise FileExistsError(
            "confirmation_claims.json already exists. Keep the frozen sample."
        )

    baseline = json.loads(
        (PROJECT_DIR / "experiments/evaluation_claims.json").read_text(encoding="utf-8")
    )
    raw = (PROJECT_DIR / "data/claims_dev.jsonl").read_bytes()
    dataset_hash = hashlib.sha256(raw).hexdigest()

    if baseline["split"] != "dev":
        raise ValueError("Expected the original development-set manifest.")
    if dataset_hash != baseline["dataset_sha256"]:
        raise ValueError("The development dataset differs from the baseline.")

    excluded = set(baseline["claim_ids"])

    # Also exclude development claims from additional saved comparisons.
    for path in sorted((PROJECT_DIR / "results").glob("skill_comparison_*.json")):
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("split") == "dev":
            excluded.update(previous["claim_ids"])

    for path in sorted((PROJECT_DIR / "results").glob("fixed_evidence_*.json")):
        previous = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(case["claim"]["id"] for case in previous["cases"])

    claims = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(SEED)
    selected = []

    for label in LABELS:
        pool = sorted(
            [
                claim for claim in claims
                if claim["id"] not in excluded and claim_label(claim) == label
            ],
            key=lambda claim: claim["id"],
        )
        if len(pool) < PER_LABEL:
            raise ValueError(f"Not enough unused claims for {label}.")
        selected.extend(rng.sample(pool, PER_LABEL))

    rng.shuffle(selected)

    skill_path = PROJECT_DIR / ".claude/skills/scientific-evidence/SKILL.md"
    current = skill_path.read_text(encoding="utf-8")

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

    manifest = {
        "experiment": "Fresh-claim comparison of current and minimal Skills",
        "split": "dev",
        "seed": SEED,
        "claims_per_label": PER_LABEL,
        "dataset_sha256": dataset_hash,
        "corpus_sha256": hashlib.sha256(
            (PROJECT_DIR / "data/corpus.jsonl").read_bytes()
        ).hexdigest(),
        "excluded_claim_ids": sorted(excluded),
        "claim_ids": [claim["id"] for claim in selected],
        "labels": {
            str(claim["id"]): claim_label(claim)
            for claim in selected
        },
        "skill_texts": {
            "current": current,
            "minimal": current.replace(old_rule, new_rule),
        },
        "adoption_checks": [
            "Higher evidence F1 on the same successfully completed claims.",
            "No lower evidence recall or document-label F1 on those claims.",
            "No increase in processing failures.",
            "Review changed answers for lost necessary qualifications.",
        ],
    }

    with output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    counts = Counter(claim_label(claim) for claim in selected)

    for label in LABELS:
        print(f"{label}: {counts[label]}")

    overlap = set(manifest["claim_ids"]) & excluded
    print(f"Total: {len(selected)}")
    print(f"Overlap with previous claims: {len(overlap)}")
    print("Both rule versions saved. No API calls made.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()