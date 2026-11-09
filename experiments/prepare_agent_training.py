"""Freeze training claims and the baseline for bounded agent development."""

import ast
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.prepare_evaluation import claim_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED = 45
PER_LABEL = 20
REPEATS = 2
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_options(source):
    """Read explicit defaults without importing the SDK or loading a model."""
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MAX_ABSTRACTS":
                    values[target.id] = ast.literal_eval(node.value)
    main = next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "main")
    positional = main.args.posonlyargs + main.args.args
    defaults = dict(zip([arg.arg for arg in positional[-len(main.args.defaults):]],
                        main.args.defaults)) if main.args.defaults else {}
    defaults.update({arg.arg: value for arg, value in
                     zip(main.args.kwonlyargs, main.args.kw_defaults) if value is not None})
    calls = [node for node in ast.walk(main)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "ClaudeAgentOptions"]
    if len(calls) != 1:
        raise ValueError("Expected one ClaudeAgentOptions configuration in agent.main.")
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}

    def literal(node):
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            if node.id in defaults:
                return ast.literal_eval(defaults[node.id])
        return ast.literal_eval(node)

    result = {
        "model": literal(keywords["model"]),
        "max_turns": literal(keywords["max_turns"]),
        "max_abstracts": values["MAX_ABSTRACTS"],
    }
    if not isinstance(result["model"], str) or not result["model"]:
        raise ValueError("Expected an explicit baseline model ID.")
    if result["max_turns"] < 1 or result["max_abstracts"] != 3:
        raise ValueError("Expected a positive turn limit and the three-abstract budget.")
    return result


def select_claims(claims, split, per_label=PER_LABEL, seed=SEED):
    by_id = {claim["id"]: claim for claim in claims}
    if len(by_id) != len(claims):
        raise ValueError("Duplicate training claim IDs.")
    tuning = split["tuning_claim_ids"]
    validation = split["validation_claim_ids"]
    tuning_ids = set(tuning)
    validation_ids = set(validation)
    for ids in (tuning, validation):
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(by_id):
            raise ValueError("Expected known, unique, nonempty retrieval split IDs.")
    if tuning_ids & validation_ids:
        raise ValueError("Retrieval split claim IDs overlap.")
    annotated = {claim["id"] for claim in claims if claim["evidence"]}
    if tuning_ids | validation_ids != annotated:
        raise ValueError("Retrieval split must cover every annotated training claim.")
    tuning_papers = {doc for key in tuning for doc in by_id[key]["evidence"]}
    validation_papers = {doc for key in validation for doc in by_id[key]["evidence"]}
    if tuning_papers & validation_papers:
        raise ValueError("Annotated papers overlap between retrieval halves.")

    pools = {label: [] for label in LABELS}
    excluded_nei = 0
    nei_without_cited_ids = 0
    for claim in claims:
        label = claim_label(claim)
        if claim["evidence"]:
            if label not in ("SUPPORT", "CONTRADICT"):
                raise ValueError(f"Unexpected annotated label for claim {claim['id']}.")
            if claim["id"] in tuning_ids:
                pools[label].append(claim)
        else:
            cited_ids = {str(doc) for doc in (claim.get("cited_doc_ids") or [])}
            # NEI has no gold evidence: use citation metadata only for this exclusion.
            if cited_ids & validation_papers:
                excluded_nei += 1
                continue
            nei_without_cited_ids += int(not cited_ids)
            pools["NOT_ENOUGH_INFO"].append(claim)

    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        pool = sorted(pools[label], key=lambda claim: claim["id"])
        if len(pool) < per_label:
            raise ValueError(f"Need {per_label} eligible {label} claims; found {len(pool)}.")
        selected.extend(rng.sample(pool, per_label))
    rng.shuffle(selected)
    return selected, {
        "eligible_pool_counts": {label: len(pools[label]) for label in LABELS},
        "selected_counts": dict(Counter(claim_label(claim) for claim in selected)),
        "selected_claims": len(selected),
        "nei_excluded_for_validation_paper_links": excluded_nei,
        "eligible_nei_without_citation_metadata": nei_without_cited_ids,
        "retrieval_validation_claim_overlap": len(
            {claim["id"] for claim in selected} & validation_ids
        ),
    }


def main():
    output = PROJECT_DIR / "experiments/agent_training_claims.json"
    if output.exists():
        raise FileExistsError(f"Keep the frozen training sample: {output}")
    split_path = PROJECT_DIR / "experiments/retrieval_split.json"
    config_path = PROJECT_DIR / "experiments/retrieval_config.json"
    result_path = PROJECT_DIR / "results/retrieval_hybrid_validation.json"
    train_path = PROJECT_DIR / "data/claims_train.jsonl"
    corpus_path = PROJECT_DIR / "data/corpus.jsonl"
    split, config, validation = map(read_json, (split_path, config_path, result_path))
    if split["schema_version"] != 1 or split["split"] != "train":
        raise ValueError("Expected the frozen training retrieval split.")
    if (validation["status"] != "finished" or validation["partition"] != "validation"
            or validation["dataset_split"] != "train"):
        raise ValueError("Expected the completed training retrieval validation.")
    if (validation["configuration"] != config
            or validation["claim_ids"] != split["validation_claim_ids"]
            or config["retrieval_split_sha256"] != sha256(split_path)):
        raise ValueError("Retrieval validation does not match the frozen setup.")
    if (validation["decision"]["retrieval_gate_passed"] is not True
            or validation["differences"]["recall_at_3"] <= 0):
        raise ValueError("Hybrid retrieval did not pass its validation rule.")
    weight = validation["dense_weight"]
    if weight not in config["fusion"]["dense_weights"]:
        raise ValueError("Validated weight is not a frozen candidate.")
    for path, key in ((train_path, "dataset_sha256"), (corpus_path, "corpus_sha256")):
        if sha256(path) != config[key] or config[key] != split[key]:
            raise ValueError(f"Frozen input changed: {path.name}")
    for name in ("hybrid_retrieval.py", "corpus_tools.py"):
        if sha256(PROJECT_DIR / name) != validation["source_sha256"][name]:
            raise ValueError(f"Retriever changed since validation: {name}")

    claims = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    selected, summary = select_claims(claims, split)
    skill_path = PROJECT_DIR / ".claude/skills/scientific-evidence/SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")
    if "name: scientific-evidence" not in skill_text or "$ARGUMENTS" not in skill_text:
        raise ValueError("Expected the scientific-evidence Skill with its claim placeholder.")
    options = baseline_options((PROJECT_DIR / "agent.py").read_text(encoding="utf-8"))
    baseline_files = (
        "agent.py", "corpus_tools.py", "scoring.py", "hybrid_retrieval.py",
        ".claude/skills/scientific-evidence/SKILL.md",
        "experiments/prepare_evaluation.py", "experiments/prepare_agent_training.py",
    )
    manifest = {
        "schema_version": 1,
        "experiment": "Bounded agent development on a fixed training sample",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "train", "sample_role": "Development, not a held-out performance estimate",
        "seed": SEED, "claims_per_label": PER_LABEL,
        "dataset_sha256": split["dataset_sha256"], "corpus_sha256": split["corpus_sha256"],
        "retrieval_split_sha256": sha256(split_path),
        "retrieval_config_sha256": sha256(config_path),
        "retrieval_validation_sha256": sha256(result_path),
        "selection_policy": {
            "annotated": "Seeded sample from the retrieval-tuning half",
            "nei": "Training NEI without known cited-paper links to retrieval-validation evidence",
            "previous_training_use": "Not excluded; this is a development sample",
        },
        "claim_ids": [claim["id"] for claim in selected],
        "labels": {str(claim["id"]): claim_label(claim) for claim in selected},
        "summary": summary,
        "repetitions_per_condition": REPEATS,
        "first_comparison": {
            "current_haiku": {"retriever": "bm25"},
            "hybrid_only_haiku": {"retriever": "hybrid", "dense_weight": weight},
        },
        "shared_agent_options": {**options, "use_skill": True, "search_results_per_call": 5},
        "skill_name": "scientific-evidence", "skill_text": skill_text,
        "skill_sha256": sha256(skill_path),
        "baseline_source_sha256": {
            name: sha256(PROJECT_DIR / name) for name in baseline_files
        },
        "planned_candidates": ["hybrid_retrieval", "claim_text_first", "read_when_unsure"],
        "scope_limit": "Test these candidates individually, then at most one retained combination",
        "metrics": [
            "evidence_precision_recall_f1", "document_verdict_precision_recall_f1",
            "nei_false_flags", "correct_nei_abstentions", "processing_failures",
            "searches", "abstracts_read", "latency", "cost",
        ],
        "interpretation": "Training comparisons guide development; final claims use frozen dev evaluation",
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Shared model: {options['model']}")
    print(f"Hybrid dense weight: {weight}")
    print(f"Repetitions per condition: {REPEATS}")
    print(f"Planned runs for first comparison: {len(selected) * REPEATS * 2}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
