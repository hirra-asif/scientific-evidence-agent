import json
import re
from functools import lru_cache
from pathlib import Path
from threading import RLock

from rank_bm25 import BM25Okapi

DATA_PATH = Path(__file__).resolve().parent / "data" / "corpus.jsonl"

_LOAD_LOCK = RLock()


@lru_cache(maxsize=1)
def _load_documents(path):
    """Read once on demand; importing this module never needs the dataset."""
    try:
        with open(path, encoding="utf-8") as file:
            documents = [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"SciFact corpus not found at {path}. Download the dataset using the README Setup instructions."
        ) from error
    if not documents:
        raise ValueError("The SciFact corpus is empty.")
    by_id = {str(document["doc_id"]): document for document in documents}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate document IDs in the SciFact corpus.")
    return documents, by_id


def tokenize(text):
    """Convert text into lowercase words and numbers."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def _load_index(path):
    documents, _ = _load_documents(path)
    return BM25Okapi([
        tokenize(document["title"] + " " + " ".join(document["abstract"]))
        for document in documents
    ])


def __getattr__(name):
    """Preserve the existing documents/index imports used by retrieval scripts."""
    if name not in {"documents", "documents_by_id", "index"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _LOAD_LOCK:
        if name == "index":
            return _load_index(DATA_PATH)
        documents, by_id = _load_documents(DATA_PATH)
        return documents if name == "documents" else by_id


def search_abstracts(query: str, top_k: int = 5) -> list[dict]:
    """Return the highest-ranked documents for a search query."""
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    with _LOAD_LOCK:
        documents, _ = _load_documents(DATA_PATH)
        index = _load_index(DATA_PATH)
    scores = index.get_scores(tokenize(query))
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "document_id": str(documents[i]["doc_id"]),
            "title": documents[i]["title"],
            "score": round(float(scores[i]), 3),
        }
        for i in ranked_indices
        if scores[i] > 0
    ]


def read_abstract(document_id: str) -> dict:
    """Return an abstract with numbered sentences."""
    with _LOAD_LOCK:
        _, documents_by_id = _load_documents(DATA_PATH)
    document = documents_by_id[str(document_id)]

    return {
        "document_id": str(document["doc_id"]),
        "title": document["title"],
        "sentences": [
            {"sentence_id": i, "text": sentence}
            for i, sentence in enumerate(document["abstract"])
        ],
    }


if __name__ == "__main__":
    query = "1 in 5 million in UK have abnormal PrP positivity."
    results = search_abstracts(query)
    print(json.dumps(results, indent=2))
