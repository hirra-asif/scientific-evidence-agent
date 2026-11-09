import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TOP_K = (3, 5)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_retrieval(rows):
    if not rows:
        raise ValueError("No retrieval results to score.")

    gold_sets = [set(row["gold_document_ids"]) for row in rows]
    if any(not gold for gold in gold_sets):
        raise ValueError("Retrieval recall requires annotated evidence.")

    ranked_ids = [
        [item["document_id"] for item in row["results"]]
        for row in rows
    ]
    if any(len(ids) != len(set(ids)) for ids in ranked_ids):
        raise ValueError("Duplicate retrieved document IDs.")

    gold_pairs = sum(len(gold) for gold in gold_sets)
    summary = {
        "claims": len(rows),
        "gold_claim_paper_pairs": gold_pairs,
    }

    for k in TOP_K:
        hits = [
            len(gold & set(ids[:k]))
            for gold, ids in zip(gold_sets, ranked_ids)
        ]
        summary[f"at_{k}"] = {
            "retrieved_gold_pairs": sum(hits),
            "recall": sum(hits) / gold_pairs,
            "mean_claim_recall": sum(
                hit / len(gold)
                for hit, gold in zip(hits, gold_sets)
            ) / len(rows),
            "claims_with_any_gold": sum(hit > 0 for hit in hits),
            "claims_with_all_gold": sum(
                hit == len(gold)
                for hit, gold in zip(hits, gold_sets)
            ),
        }

    return summary


def main():
    manifest_path = PROJECT_DIR / "experiments/retrieval_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["schema_version"] != 1 or manifest["split"] != "train":
        raise ValueError("Expected the frozen training retrieval split.")

    dataset_path = PROJECT_DIR / "data/claims_train.jsonl"
    corpus_path = PROJECT_DIR / "data/corpus.jsonl"

    if sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Training data differs from the frozen split.")
    if sha256(corpus_path) != manifest["corpus_sha256"]:
        raise ValueError("Corpus differs from the frozen split.")

    claim_ids = manifest["tuning_claim_ids"]

    if not claim_ids or len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Expected nonempty, unique tuning claim IDs.")
    if set(claim_ids) & set(manifest["validation_claim_ids"]):
        raise ValueError("Tuning and validation claim IDs overlap.")

    claims = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {claim["id"]: claim for claim in claims}

    if len(by_id) != len(claims):
        raise ValueError("Duplicate training claim IDs.")
    if set(claim_ids) - set(by_id):
        raise ValueError("The manifest contains unknown training claim IDs.")
    if any(not by_id[i]["evidence"] for i in claim_ids):
        raise ValueError("Every tuning claim must have annotated evidence.")

    source_files = [
        "corpus_tools.py",
        "experiments/evaluate_bm25.py",
        "experiments/retrieval_split.json",
    ]
    fingerprints = {
        name: sha256(PROJECT_DIR / name)
        for name in source_files
    }

    started = datetime.now(timezone.utc)
    timer = time.perf_counter()

    # Use the project's actual retriever after checking the frozen inputs.
    from corpus_tools import search_abstracts

    rows = []

    for index, claim_id in enumerate(claim_ids, start=1):
        claim = by_id[claim_id]
        retrieved = search_abstracts(
            claim["claim"], top_k=max(TOP_K)
        )
        rows.append({
            "claim_id": claim_id,
            "query": claim["claim"],
            "gold_document_ids": sorted(claim["evidence"]),
            "results": retrieved,
        })

        if index % 50 == 0 or index == len(claim_ids):
            print(
                f"Searched {index}/{len(claim_ids)} tuning claims",
                flush=True,
            )

    elapsed = time.perf_counter() - timer
    summary = summarize_retrieval(rows)

    for name, expected in fingerprints.items():
        if sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"{name} changed during the run.")

    results = {
        "status": "finished",
        "experiment": "Claim-text BM25 retrieval baseline",
        "dataset_split": "train",
        "partition": "tuning",
        "query_policy": "Unmodified claim text; one search per claim",
        "top_k": list(TOP_K),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "timing_scope": (
            "Retriever import/index initialization and claim searches"
        ),
        "dataset_sha256": manifest["dataset_sha256"],
        "corpus_sha256": manifest["corpus_sha256"],
        "source_sha256": fingerprints,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name)
            for name in ("rank-bm25", "numpy")
        },
        "claim_ids": claim_ids,
        "summary": summary,
        "rows": rows,
    }

    output = PROJECT_DIR / "results" / (
        f"retrieval_bm25_tuning_{started:%Y%m%d_%H%M%S_%f}.json"
    )
    output.parent.mkdir(exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()