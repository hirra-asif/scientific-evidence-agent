"""Confirm the tuning-selected hybrid once on the frozen validation claims."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.evaluate_bm25 import summarize_retrieval
from hybrid_retrieval import HybridRetriever, choose_weight, sha256

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "results/retrieval_hybrid_validation.json"
FROZEN_FILES = (
    "hybrid_retrieval.py",
    "corpus_tools.py",
    "experiments/evaluate_hybrid.py",
    "experiments/evaluate_bm25.py",
    "experiments/retrieval_config.json",
    "experiments/retrieval_split.json",
    "data/claims_train.jsonl",
    "data/corpus.jsonl",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_weight(tuning, config, split, by_id):
    """Verify the recorded winner using tuning rows, without scoring validation."""
    if (tuning["status"] != "finished" or tuning["partition"] != "tuning"
            or tuning["dataset_split"] != "train"):
        raise ValueError("Expected a completed training/tuning comparison.")
    if tuning["configuration"] != config:
        raise ValueError("Configuration differs from the tuning run.")
    if tuning["bm25_baseline_reproduced"] is not True:
        raise ValueError("The tuning run did not reproduce its BM25 baseline.")
    if tuning["selection"]["validation_evaluated"] is not False:
        raise ValueError("Expected a selection made before validation.")
    claim_ids = split["tuning_claim_ids"]
    if tuning["claim_ids"] != claim_ids:
        raise ValueError("Tuning claim IDs differ from the frozen split.")

    def verified_summary(condition):
        rows = condition["rows"]
        if [row["claim_id"] for row in rows] != claim_ids:
            raise ValueError("Tuning rows do not match the frozen claim IDs.")
        for row in rows:
            claim = by_id[row["claim_id"]]
            if (row["query"] != claim["claim"]
                    or row["gold_document_ids"] != sorted(claim["evidence"])):
                raise ValueError("Tuning claim text or gold documents differ.")
        summary = summarize_retrieval(rows)
        if summary != condition["summary"]:
            raise ValueError("Tuning summary does not match its saved rows.")
        return summary

    baseline = verified_summary(tuning["bm25"])
    weights = [condition["dense_weight"] for condition in tuning["hybrids"]]
    if weights != config["fusion"]["dense_weights"] or len(weights) != len(set(weights)):
        raise ValueError("Tuning candidates differ from the frozen weights.")
    summaries = {
        condition["dense_weight"]: verified_summary(condition)
        for condition in tuning["hybrids"]
    }
    winner = choose_weight(summaries)
    if winner != tuning["selection"]["best_dense_weight"]:
        raise ValueError("Recorded selection does not match the tuning results.")
    if (summaries[winner]["at_3"]["recall"] <= baseline["at_3"]["recall"]
            or tuning["selection"]["tuning_passed"] is not True):
        raise ValueError("The selected hybrid did not pass the tuning rule.")
    return winner


def validation_claims(split, by_id):
    if split["schema_version"] != 1 or split["split"] != "train":
        raise ValueError("Expected the frozen training retrieval split.")
    tuning_ids = split["tuning_claim_ids"]
    validation_ids = split["validation_claim_ids"]
    for ids in (tuning_ids, validation_ids):
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(by_id):
            raise ValueError("Expected known, unique, nonempty claim IDs.")
    tuning_set, validation_set = set(tuning_ids), set(validation_ids)
    if tuning_set & validation_set:
        raise ValueError("Tuning and validation claim IDs overlap.")
    annotated = {key for key, claim in by_id.items() if claim["evidence"]}
    if tuning_set | validation_set != annotated:
        raise ValueError("Split must cover every annotated training claim exactly once.")
    tuning_papers = {doc for key in tuning_ids for doc in by_id[key]["evidence"]}
    validation_papers = {doc for key in validation_ids for doc in by_id[key]["evidence"]}
    if tuning_papers & validation_papers:
        raise ValueError("Annotated papers overlap between tuning and validation.")
    return [by_id[key] for key in validation_ids]


def compare(retriever, claims, weight, bm25_search):
    queries, token_lengths = retriever.embed_queries(
        [claim["claim"] for claim in claims], progress=True,
    )
    if len(queries) != len(claims):
        raise ValueError("Embedding count does not match validation claims.")
    bm25_rows, hybrid_rows = [], []
    for i, (claim, embedding) in enumerate(zip(claims, queries), 1):
        bm25_order, dense_order = retriever.rankings(claim["claim"], embedding)
        bm25_results = bm25_search(claim["claim"], top_k=5)
        if ([item["document_id"] for item in bm25_results]
                != [retriever.document_ids[j] for j in bm25_order[:5]]):
            raise ValueError(f"BM25 implementations disagree for claim {claim['id']}.")
        common = {
            "claim_id": claim["id"], "query": claim["claim"],
            "gold_document_ids": sorted(claim["evidence"]),
        }
        bm25_rows.append({**common, "results": bm25_results})
        hybrid_rows.append({
            **common, "results": retriever.results(bm25_order, dense_order, weight),
        })
        if i % 50 == 0 or i == len(claims):
            print(f"Compared {i}/{len(claims)} validation claims", flush=True)
    return bm25_rows, hybrid_rows, token_lengths


def main(tuning_path):
    output = PROJECT_DIR / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"Validation already has a saved result: {output}")
    started = datetime.now(timezone.utc)
    timer = time.perf_counter()
    tuning_hash = sha256(tuning_path)
    tuning = read_json(tuning_path)
    config = read_json(PROJECT_DIR / "experiments/retrieval_config.json")
    split_path = PROJECT_DIR / "experiments/retrieval_split.json"
    split = read_json(split_path)
    fingerprints = {name: sha256(PROJECT_DIR / name) for name in FROZEN_FILES}
    for name, actual in fingerprints.items():
        if tuning["source_sha256"].get(name) != actual:
            raise ValueError(f"File differs from the tuning run: {name}")
    script_name = "experiments/validate_hybrid.py"
    fingerprints[script_name] = sha256(PROJECT_DIR / script_name)
    if sha256(split_path) != config["retrieval_split_sha256"]:
        raise ValueError("Retrieval split differs from the frozen configuration.")
    for name, key in (("data/claims_train.jsonl", "dataset_sha256"),
                      ("data/corpus.jsonl", "corpus_sha256")):
        if fingerprints[name] != config[key] or config[key] != split[key]:
            raise ValueError(f"Frozen data changed: {name}")

    train_path = PROJECT_DIR / "data/claims_train.jsonl"
    claims = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    by_id = {claim["id"]: claim for claim in claims}
    if len(by_id) != len(claims):
        raise ValueError("Duplicate training claim IDs.")
    held_out = validation_claims(split, by_id)
    weight = selected_weight(tuning, config, split, by_id)
    print(f"Frozen tuning selection: dense weight {weight}", flush=True)
    print(f"Evaluating {len(held_out)} validation claims...", flush=True)

    retriever = HybridRetriever()
    from corpus_tools import search_abstracts

    cache_hash = sha256(retriever.cache_path)
    if retriever.cache_path.name != tuning["cache"]["file"]:
        raise ValueError("Embedding cache identity differs from tuning.")
    if retriever.cache_hit and cache_hash != tuning["cache"]["sha256"]:
        raise ValueError("Existing embedding cache changed since tuning.")
    bm25_rows, hybrid_rows, query_lengths = compare(
        retriever, held_out, weight, search_abstracts,
    )
    bm25_summary = summarize_retrieval(bm25_rows)
    hybrid_summary = summarize_retrieval(hybrid_rows)
    differences = {
        f"recall_at_{k}": hybrid_summary[f"at_{k}"]["recall"] - bm25_summary[f"at_{k}"]["recall"]
        for k in (3, 5)
    }
    passed = differences["recall_at_3"] > 0
    for name, expected in fingerprints.items():
        if sha256(PROJECT_DIR / name) != expected:
            raise RuntimeError(f"File changed during validation: {name}")
    if sha256(tuning_path) != tuning_hash or sha256(retriever.cache_path) != cache_hash:
        raise RuntimeError("Tuning result or embedding cache changed during validation.")

    report = {
        "status": "finished", "experiment": "Confirm tuning-selected hybrid retrieval",
        "dataset_split": "train", "partition": "validation",
        "claim_ids": split["validation_claim_ids"], "top_k": [3, 5],
        "query_policy": "Unmodified claim text; one search per claim",
        "configuration": config, "dense_weight": weight,
        "tuning_result_file": Path(tuning_path).name, "tuning_result_sha256": tuning_hash,
        "source_sha256": fingerprints,
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - timer, 3),
        "timing_scope": "Complete validation run, including loading/encoding and checks",
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cache": {"file": retriever.cache_path.name, "hit": retriever.cache_hit,
                  "sha256": cache_hash},
        "corpus_token_lengths": retriever.corpus_token_lengths,
        "query_token_lengths": query_lengths,
        "bm25": {"summary": bm25_summary, "rows": bm25_rows},
        "hybrid": {"summary": hybrid_summary, "rows": hybrid_rows},
        "differences": differences,
        "decision": {
            "rule": config["selection"]["keep_rule"],
            "tuning_passed": True, "validation_passed": passed,
            "retrieval_gate_passed": passed,
            "recommended_retriever": "hybrid" if passed else "bm25",
            "agent_behavior_changed": False,
        },
    }
    output.parent.mkdir(exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print("\nMethod                  Recall@3  Recall@5")
    print(f"BM25                    {bm25_summary['at_3']['recall']:.4f}    {bm25_summary['at_5']['recall']:.4f}")
    print(f"Hybrid (dense={weight:.2f})     {hybrid_summary['at_3']['recall']:.4f}    {hybrid_summary['at_5']['recall']:.4f}")
    print(f"Recall@3 difference: {100 * differences['recall_at_3']:+.2f} percentage points")
    print(f"Passed the frozen retrieval rule: {passed}")
    print("Agent verdict and evidence performance still need evaluation.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-result", required=True, type=Path)
    args = parser.parse_args()
    main(args.tuning_result)
