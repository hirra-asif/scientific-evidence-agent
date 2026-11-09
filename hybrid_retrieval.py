"""Frozen E5 embeddings and weighted rank fusion with the existing BM25 index."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked_indices(scores, limit, positive_only=False):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite, one-dimensional retrieval scores.")
    if limit < 1:
        raise ValueError("The ranking limit must be positive.")
    order = np.argsort(-scores, kind="stable")
    if positive_only:
        order = order[scores[order] > 0]
    return order[:limit].tolist()


def fuse_ranks(bm25_order, dense_order, dense_weight, rank_constant=60):
    """Inputs contain corpus indices; ties follow original corpus order."""
    if not 0 <= dense_weight <= 1 or not rank_constant > 0:
        raise ValueError("Invalid fusion weight or rank constant.")
    scores = {}
    for order, weight in ((bm25_order, 1 - dense_weight),
                          (dense_order, dense_weight)):
        if len(order) != len(set(order)):
            raise ValueError("Duplicate document in a retrieval ranking.")
        if weight == 0:
            continue
        for rank, index in enumerate(order, start=1):
            scores[index] = scores.get(index, 0.0) + weight / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def choose_weight(summaries):
    """Maximize recall@3, then recall@5, then prefer less dense weight."""
    return min(summaries, key=lambda weight: (
        -summaries[weight]["at_3"]["recall"],
        -summaries[weight]["at_5"]["recall"],
        weight,
    ))


def check_embeddings(values, count, dimensions):
    if values.shape != (count, dimensions) or values.dtype != np.float32:
        raise ValueError("Unexpected embedding shape or dtype.")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite embedding values.")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embeddings are not normalized.")


def read_cache(path, identity, document_ids, dimensions):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata["identity"] != identity:
            raise ValueError("Embedding cache does not match the frozen settings.")
        if metadata["document_ids"] != document_ids:
            raise ValueError("Embedding cache has a different document order.")
        values = saved["embeddings"]
    check_embeddings(values, len(document_ids), dimensions)
    return values, metadata["token_lengths"]


class HybridRetriever:
    def __init__(self):
        self.config = json.loads(
            (PROJECT_DIR / "experiments/retrieval_config.json").read_text()
        )
        config = self.config
        self.settings = config["embedding"]
        self.fusion = config["fusion"]
        if config["schema_version"] != 1:
            raise ValueError("Unsupported retrieval configuration version.")
        # These fields describe behavior implemented below, not tunable options.
        required_embedding = {
            "dtype": "float32",
            "normalize_embeddings": True,
            "document_text": "title + ' ' + ' '.join(abstract)",
            "similarity": "dot product of normalized embeddings",
        }
        required_fusion = {
            "method": "weighted_reciprocal_rank_fusion",
            "rank_start": 1,
            "bm25_weight": "1 - dense_weight",
            "missing_rank_contribution": 0,
            "bm25_candidates": "Only documents with a positive BM25 score",
            "tie_break": "Original corpus order",
        }
        for actual, required in ((self.settings, required_embedding),
                                 (self.fusion, required_fusion)):
            if any(actual[key] != value for key, value in required.items()):
                raise ValueError("Configuration describes unsupported behavior.")
        for name, expected in config["environment"]["packages"].items():
            if version(name) != expected:
                raise ValueError(f"Package version changed: {name}; expected {expected}.")
        for name, key in (("data/corpus.jsonl", "corpus_sha256"),
                          ("corpus_tools.py", "bm25_source_sha256")):
            if sha256(PROJECT_DIR / name) != config[key]:
                raise ValueError(f"Frozen input changed: {name}")

        # Reuse the exact production corpus order, tokenizer and BM25 index.
        from corpus_tools import documents, index, tokenize
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.document_ids = [str(doc["doc_id"]) for doc in documents]
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ValueError("Duplicate corpus document IDs.")
        self.bm25 = index
        self.tokenize = tokenize
        print("Loading the pinned E5 model...", flush=True)
        self.model = SentenceTransformer(
            self.settings["model_id"], revision=self.settings["revision"],
            device=self.settings["device"], trust_remote_code=False,
        )
        self.model.float()
        self.model.max_seq_length = self.settings["max_seq_length"]

        identity = {
            "corpus_sha256": config["corpus_sha256"],
            "embedding": self.settings,
            "environment": config["environment"],
            "implementation_sha256": sha256(Path(__file__)),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_dir = PROJECT_DIR / "data/embedding_cache"
        cache_dir.mkdir(exist_ok=True)
        self.cache_path = cache_dir / f"e5_{key}.npz"
        self.cache_hit = self.cache_path.exists()
        if self.cache_hit:
            print("Loading cached corpus embeddings...", flush=True)
            self.embeddings, self.corpus_token_lengths = read_cache(
                self.cache_path, identity, self.document_ids,
                self.settings["dimensions"],
            )
        else:
            texts = [
                self.settings["passage_prefix"] + doc["title"] + " "
                + " ".join(doc["abstract"]) for doc in documents
            ]
            print(f"Embedding {len(texts)} corpus documents...", flush=True)
            self.embeddings, self.corpus_token_lengths = self._encode(texts, True)
            metadata = {
                "identity": identity,
                "document_ids": self.document_ids,
                "token_lengths": self.corpus_token_lengths,
            }
            # Complete the cache before making it visible to another run.
            with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as file:
                temporary = Path(file.name)
                try:
                    np.savez(file, embeddings=self.embeddings,
                             metadata=json.dumps(metadata, sort_keys=True))
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(self.cache_path)
        print(f"Corpus token lengths: {self.corpus_token_lengths}", flush=True)

    def _encode(self, texts, progress=False):
        # Count actual tokenizer lengths, including prefixes and special tokens.
        lengths = []
        for start in range(0, len(texts), self.settings["batch_size"]):
            batch = texts[start:start + self.settings["batch_size"]]
            tokens = self.model.tokenizer(
                batch, padding=False, truncation=False, add_special_tokens=True,
                return_attention_mask=False, return_token_type_ids=False,
                verbose=False,
            )
            lengths.extend(len(ids) for ids in tokens["input_ids"])
        values = self.model.encode(
            texts, prompt="", batch_size=self.settings["batch_size"],
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=progress,
        )
        check_embeddings(values, len(texts), self.settings["dimensions"])
        return values, {
            "texts": len(texts),
            "over_token_limit": sum(n > self.model.max_seq_length for n in lengths),
            "longest_tokens": max(lengths, default=0),
            "token_limit": self.model.max_seq_length,
        }

    def embed_queries(self, queries, progress=False):
        return self._encode(
            [self.settings["query_prefix"] + query for query in queries], progress,
        )

    def rankings(self, query, query_embedding):
        depth = self.fusion["candidate_depth_per_retriever"]
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        dense_scores = self.embeddings @ query_embedding
        return (
            ranked_indices(bm25_scores, depth, positive_only=True),
            ranked_indices(dense_scores, depth),
        )

    def results(self, bm25_order, dense_order, dense_weight, top_k=5):
        if dense_weight not in self.fusion["dense_weights"]:
            raise ValueError("Weight is not one of the frozen candidates.")
        if not 1 <= top_k <= self.fusion["candidate_depth_per_retriever"]:
            raise ValueError("Invalid number of results.")
        ranked = fuse_ranks(
            bm25_order, dense_order, dense_weight, self.fusion["rank_constant"],
        )
        return [
            {"document_id": self.document_ids[index],
             "title": self.documents[index]["title"], "score": float(score)}
            for index, score in ranked[:top_k]
        ]

    def search(self, query, dense_weight, top_k=5):
        embeddings, _ = self.embed_queries([query])
        bm25_order, dense_order = self.rankings(query, embeddings[0])
        return self.results(bm25_order, dense_order, dense_weight, top_k)
