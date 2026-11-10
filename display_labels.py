"""Reader-facing names; saved experiment identifiers remain unchanged."""

CONDITION_LABELS = {
    "current_haiku": "Haiku · BM25",
    "improved_haiku": "Haiku · hybrid",
    "improved_sonnet": "Sonnet · hybrid",
    "fixed_pipeline": "Fixed pipeline · hybrid",
}


def condition_label(identifier):
    return CONDITION_LABELS.get(identifier, identifier)
