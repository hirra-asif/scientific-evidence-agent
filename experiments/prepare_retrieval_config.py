import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "intfloat/e5-small-v2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = PROJECT_DIR / "experiments/retrieval_config.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen config: {output}")

    split_path = PROJECT_DIR / "experiments/retrieval_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["schema_version"] != 1 or split["split"] != "train":
        raise ValueError("Expected the frozen training retrieval split.")

    train_path = PROJECT_DIR / "data/claims_train.jsonl"
    corpus_path = PROJECT_DIR / "data/corpus.jsonl"
    for path, key in (
        (train_path, "dataset_sha256"),
        (corpus_path, "corpus_sha256"),
    ):
        if sha256(path) != split[key]:
            raise ValueError(f"Frozen input changed: {path.name}")

    with train_path.open(encoding="utf-8") as file:
        claims = [json.loads(line) for line in file if line.strip()]
    first_id = split["tuning_claim_ids"][0]
    claim = next(item for item in claims if item["id"] == first_id)
    with corpus_path.open(encoding="utf-8") as file:
        document = next(json.loads(line) for line in file if line.strip())

    revision = model_info(MODEL_ID, token=False).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Could not resolve an immutable model revision.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_ID} at {revision} on {device}...", flush=True)
    model = SentenceTransformer(
        MODEL_ID, revision=revision, device=device, trust_remote_code=False,
    )
    model.float()
    model.max_seq_length = 512
    texts = [
        "query: " + claim["claim"],
        "passage: " + document["title"] + " " + " ".join(document["abstract"]),
    ]
    embeddings = model.encode(
        texts, prompt="", batch_size=32, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    if embeddings.shape != (2, 384) or not np.isfinite(embeddings).all():
        raise RuntimeError("Unexpected embedding shape or non-finite values.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Embeddings are not normalized.")

    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_split_sha256": sha256(split_path),
        "dataset_sha256": split["dataset_sha256"],
        "corpus_sha256": split["corpus_sha256"],
        "bm25_source_sha256": sha256(PROJECT_DIR / "corpus_tools.py"),
        "embedding": {
            "model_id": MODEL_ID,
            "revision": revision,
            "device": device,
            "dtype": "float32",
            "dimensions": 384,
            "max_seq_length": 512,
            "batch_size": 32,
            "normalize_embeddings": True,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "document_text": "title + ' ' + ' '.join(abstract)",
            "similarity": "dot product of normalized embeddings",
        },
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rank_start": 1,
            "rank_constant": 60,
            "candidate_depth_per_retriever": 100,
            "dense_weights": [0.25, 0.5, 0.75],
            "bm25_weight": "1 - dense_weight",
            "missing_rank_contribution": 0,
            "bm25_candidates": "Only documents with a positive BM25 score",
            "tie_break": "Original corpus order",
        },
        "selection": {
            "partition": "tuning",
            "primary_metric": "at_3.recall",
            "tie_breakers": ["Higher at_5.recall", "Lower dense weight"],
            "keep_rule": (
                "Select one hybrid on tuning. Keep it only if recall@3 exceeds "
                "BM25 on both tuning and the subsequent validation comparison."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in (
                    "sentence-transformers", "torch", "transformers",
                    "huggingface-hub", "numpy", "rank-bm25",
                )
            },
        },
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    print(f"Embedding check passed: {embeddings.shape}")
    print(f"Dense weights: {config['fusion']['dense_weights']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()