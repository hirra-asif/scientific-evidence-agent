import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_PATH = Path(__file__).resolve().parent / "data" / "corpus.jsonl"

with open(DATA_PATH, encoding="utf-8") as file:
    documents = [json.loads(line) for line in file if line.strip()]

documents_by_id = {
    str(document["doc_id"]): document
    for document in documents
}


def tokenize(text):
    """Convert text into lowercase words and numbers."""
    return re.findall(r"\w+", text.lower())


# Build the search index using titles and abstracts.
index = BM25Okapi([
    tokenize(document["title"] + " " + " ".join(document["abstract"]))
    for document in documents
])


def search_abstracts(query: str, top_k: int = 5) -> list[dict]:
    """Return the highest-ranked documents for a search query."""
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

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