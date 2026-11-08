import json
from pathlib import Path

# Find the data folder relative to this script.
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_jsonl(filename):
    """Read a file containing one JSON record per line."""
    with open(DATA_DIR / filename, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# Load the abstracts and training claims.
documents = load_jsonl("corpus.jsonl")
claims = load_jsonl("claims_train.jsonl")

print(f"Number of abstracts: {len(documents)}")
print(f"Number of training claims: {len(claims)}")

# Index abstracts by document ID so we can look them up.
corpus = {str(doc["doc_id"]): doc for doc in documents}

# Choose a claim that has labelled evidence.
claim = next(item for item in claims if item["evidence"])

print("\nCLAIM:")
print(claim["claim"])

# Get the first evidence document associated with this claim.
doc_id, evidence_sets = next(iter(claim["evidence"].items()))
document = corpus[doc_id]

print("\nSOURCE:")
print(document["title"])

# Each evidence set contains a label and supporting sentence IDs.
evidence_ids = set()

print("\nDATASET ANSWERS:")
for evidence in evidence_sets:
    print(f"Label: {evidence['label']}")
    print(f"Evidence sentence IDs: {evidence['sentences']}")
    evidence_ids.update(evidence["sentences"])

# Display the complete abstract and mark its evidence sentences.
print("\nABSTRACT:")
for sentence_id, sentence in enumerate(document["abstract"]):
    marker = " <-- EVIDENCE" if sentence_id in evidence_ids else ""
    print(f"[{sentence_id}] {sentence}{marker}")