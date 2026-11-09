"""Evaluate only the frozen tuning claims; leave validation untouched."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.evaluate_bm25 import summarize_retrieval
from hybrid_retrieval import HybridRetriever, choose_weight, sha256

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main(baseline_path):
    started = datetime.now(timezone.utc)
    timer = time.perf_counter()
    config_path = PROJECT_DIR / "experiments/retrieval_config.json"
    split_path = PROJECT_DIR / "experiments/retrieval_split.json"
    config = json.loads(config_path.read_text())
    split = json.loads(split_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    train_path = PROJECT_DIR / "data/claims_train.jsonl"
    corpus_path = PROJECT_DIR / "data/corpus.jsonl"

    if split["schema_version"] != 1 or split["split"] != "train":
        raise ValueError("Expected the frozen training retrieval split.")
    if config["retrieval_split_sha256"] != sha256(split_path):
        raise ValueError("Retrieval split changed after configuration was frozen.")
    for path, key in ((train_path, "dataset_sha256"), (corpus_path, "corpus_sha256")):
        if sha256(path) != split[key] or split[key] != config[key]:
            raise ValueError(f"Frozen input changed: {path.name}")
        if baseline[key] != config[key]:
            raise ValueError("Baseline used different data.")
    if (baseline["status"] != "finished" or baseline["partition"] != "tuning"
            or baseline["dataset_split"] != "train"):
        raise ValueError("Expected the completed training/tuning BM25 baseline.")
    if baseline["source_sha256"]["corpus_tools.py"] != config["bm25_source_sha256"]:
        raise ValueError("Baseline used different BM25 code.")
    for name, expected in baseline["package_versions"].items():
        if config["environment"]["packages"][name] != expected:
            raise ValueError(f"Baseline used a different version of {name}.")

    claim_ids = split["tuning_claim_ids"]
    if not claim_ids or len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Expected unique, nonempty tuning IDs.")
    if set(claim_ids) & set(split["validation_claim_ids"]):
        raise ValueError("Tuning and validation overlap.")
    if baseline["claim_ids"] != claim_ids:
        raise ValueError("Baseline did not use the same ordered tuning claims.")
    if [row["claim_id"] for row in baseline["rows"]] != claim_ids:
        raise ValueError("Baseline rows do not match its claim manifest.")
    if summarize_retrieval(baseline["rows"]) != baseline["summary"]:
        raise ValueError("Stored baseline summary does not match its rows.")
    claims = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    by_id = {claim["id"]: claim for claim in claims}
    if len(by_id) != len(claims):
        raise ValueError("Duplicate training claim IDs.")
    selected = [by_id[claim_id] for claim_id in claim_ids]
    if any(not claim["evidence"] for claim in selected):
        raise ValueError("Tuning claims must have annotated evidence.")
    for claim, saved in zip(selected, baseline["rows"]):
        if (saved["query"] != claim["claim"]
                or saved["gold_document_ids"] != sorted(claim["evidence"])):
            raise ValueError("Baseline claim text or gold documents differ.")

    names = [
        "hybrid_retrieval.py", "corpus_tools.py", "experiments/evaluate_hybrid.py",
        "experiments/evaluate_bm25.py", "experiments/retrieval_config.json",
        "experiments/retrieval_split.json", "data/claims_train.jsonl", "data/corpus.jsonl",
    ]
    fingerprints = {name: sha256(PROJECT_DIR / name) for name in names}
    baseline_hash = sha256(baseline_path)
    retriever = HybridRetriever()
    from corpus_tools import search_abstracts

    queries, query_token_lengths = retriever.embed_queries(
        [claim["claim"] for claim in selected], progress=True,
    )
    weights = config["fusion"]["dense_weights"]
    rows = {weight: [] for weight in weights}
    bm25_rows = []
    ranking_rows = []
    for i, (claim, embedding, saved) in enumerate(zip(selected, queries, baseline["rows"]), 1):
        bm25_order, dense_order = retriever.rankings(claim["claim"], embedding)
        bm25_results = search_abstracts(claim["claim"], top_k=5)
        expected_ids = [item["document_id"] for item in saved["results"]]
        actual_ids = [item["document_id"] for item in bm25_results]
        rank_ids = [retriever.document_ids[index] for index in bm25_order[:5]]
        if actual_ids != expected_ids or rank_ids != expected_ids:
            raise ValueError(f"BM25 baseline mismatch for claim {claim['id']}.")
        common = {
            "claim_id": claim["id"], "query": claim["claim"],
            "gold_document_ids": sorted(claim["evidence"]),
        }
        bm25_rows.append({**common, "results": bm25_results})
        for weight in weights:
            rows[weight].append({
                **common, "results": retriever.results(bm25_order, dense_order, weight),
            })
        ranking_rows.append({
            "claim_id": claim["id"],
            "bm25_candidate_ids": [retriever.document_ids[j] for j in bm25_order],
            "dense_candidate_ids": [retriever.document_ids[j] for j in dense_order],
        })
        if i % 50 == 0 or i == len(selected):
            print(f"Compared {i}/{len(selected)} tuning claims", flush=True)

    summaries = {weight: summarize_retrieval(rows[weight]) for weight in weights}
    bm25_summary = summarize_retrieval(bm25_rows)
    if bm25_summary != baseline["summary"]:
        raise ValueError("Recomputed BM25 summary differs from the saved baseline.")
    winner = choose_weight(summaries)
    tuning_passed = summaries[winner]["at_3"]["recall"] > bm25_summary["at_3"]["recall"]
    for name, expected in fingerprints.items():
        if sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"File changed during evaluation: {name}")
    if sha256(baseline_path) != baseline_hash:
        raise RuntimeError("Baseline file changed during evaluation.")

    report = {
        "status": "finished", "experiment": "Frozen BM25 versus hybrid retrieval",
        "dataset_split": "train", "partition": "tuning", "claim_ids": claim_ids,
        "query_policy": "Unmodified claim text; one search per claim",
        "top_k": [3, 5], "configuration": config,
        "source_sha256": fingerprints, "baseline_sha256": baseline_hash,
        "baseline_file": baseline_path.name, "bm25_baseline_reproduced": True,
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - timer, 3),
        "timing_scope": "Entire offline comparison, including loading/encoding and checks",
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cache": {"file": retriever.cache_path.name, "hit": retriever.cache_hit,
                  "sha256": sha256(retriever.cache_path)},
        "corpus_token_lengths": retriever.corpus_token_lengths,
        "query_token_lengths": query_token_lengths,
        "bm25": {"summary": bm25_summary, "rows": bm25_rows},
        "hybrids": [
            {"dense_weight": weight, "summary": summaries[weight], "rows": rows[weight]}
            for weight in weights
        ],
        "candidate_rankings": ranking_rows,
        "selection": {
            "best_dense_weight": winner, "tuning_passed": tuning_passed,
            "validation_evaluated": False, "adopted": False,
        },
    }
    output = PROJECT_DIR / "results" / f"retrieval_hybrid_tuning_{started:%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print("\nMethod                  Recall@3  Recall@5")
    print(f"BM25                    {bm25_summary['at_3']['recall']:.4f}    {bm25_summary['at_5']['recall']:.4f}")
    for weight in weights:
        summary = summaries[weight]
        print(f"Hybrid (dense={weight:.2f})     {summary['at_3']['recall']:.4f}    {summary['at_5']['recall']:.4f}")
    print(f"Best tuning weight: {winner}; beats BM25 recall@3: {tuning_passed}")
    print("Validation has not been evaluated. No agent behavior has changed.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    main(args.baseline)
