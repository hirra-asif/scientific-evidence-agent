"""Frozen selection, provenance checks and paired analysis for the final dev run."""

import hashlib
import json
import random
import subprocess
from pathlib import Path

from experiments import agent_comparison as comparison
from experiments.final_conditions import CONDITIONS
from experiments.prepare_evaluation import LABELS, claim_label

PLAN = "experiments/final_evaluation.json"
OUTPUT = "results/final_evaluation.json"
ANALYSIS = "results/final_evaluation_analysis.json"
REPORT = "docs/final_evaluation.md"
PILOT_PLAN = "experiments/final_conditions_pilot.json"
PILOT_RESULT = "results/final_conditions_pilot.json"
SEED = 46
PER_LABEL = 20
REPETITIONS = 2
PRIMARY_METRICS = ("evidence_f1", "document_verdict_f1")
COMPARISONS = (
    {"name": "improvements", "candidate": "improved_haiku", "reference": "current_haiku"},
    {"name": "pipeline_vs_agent", "candidate": "fixed_pipeline", "reference": "improved_haiku"},
    {"name": "sonnet_vs_haiku", "candidate": "improved_sonnet", "reference": "improved_haiku"},
)
COUNT_FIELDS = ("correct_sentences", "predicted_sentences", "gold_sentences",
                "correct_document_labels", "predicted_documents", "gold_documents")
RUNNER_PATCHES = (
    ("async def run(max_new_runs=None):\n    plan_path = PROJECT_DIR / PLAN\n",
     "async def run(max_new_runs=None, *, plan_path=None, output_path=None,\n"
     "              plan_verifier=None, claims_loader=None, summary_printer=None,\n"
     '              progress_label="PILOT"):\n'
     "    plan_path = PROJECT_DIR / PLAN if plan_path is None else Path(plan_path)\n"),
    ("    verify_plan(plan)\n", "    (verify_plan if plan_verifier is None else plan_verifier)(plan)\n"),
    ('    _, claims = load_claims(plan["sample"])\n',
     '    _, claims = (load_claims if claims_loader is None else claims_loader)(plan["sample"])\n'),
    ("    output = PROJECT_DIR / OUTPUT\n",
     "    output = PROJECT_DIR / OUTPUT if output_path is None else Path(output_path)\n"),
    ('"split": "train", "role": plan["role"]',
     '"split": plan.get("split", "train"), "role": plan["role"]'),
    ("print_summary(results, output)\n",
     "(print_summary if summary_printer is None else summary_printer)(results, output)\n"),
    ('print(f"\\nPILOT {len(results[\'runs\'])}',
     'print(f"\\n{progress_label} {len(results[\'runs\'])}'),
    ('print(f"PILOT FAILED: {row[\'error\']}"',
     'print(f"{progress_label} FAILED: {row[\'error\']}"'),
)


def adapt_runner(source, frozen_hash):
    """Reuse the tested execution loop; add explicit inputs, not a second harness."""
    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    original = source
    if digest(original) != frozen_hash:
        for old, new in reversed(RUNNER_PATCHES):
            if new not in original:
                raise ValueError("Pilot runner differs from its frozen source and expected adapter.")
            original = original.replace(new, old)
        if digest(original) != frozen_hash:
            raise ValueError("Pilot runner has unrelated edits; nothing was overwritten.")
    adapted = original
    for old, new in RUNNER_PATCHES:
        if old not in adapted:
            raise ValueError("Could not locate the expected runner code.")
        adapted = adapted.replace(old, new)
    return adapted


def read_claims(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError(f"Duplicate claim IDs in {path}")
    if any(claim_label(r) not in LABELS for r in records):
        raise ValueError("Unexpected claim label.")
    return records


def normalize(text):
    return " ".join(text.casefold().split())


def recorded_claims(value):
    """Extract claim IDs/text from old and new result formats, never document IDs."""
    ids, texts = set(), set()
    def integer(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None
    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get("claim"), str):
                texts.add(normalize(node["claim"]))
                candidate = integer(node.get("id"))
                if candidate is not None:
                    ids.add(candidate)
            for key, val in node.items():
                if key == "claim_id":
                    candidate = integer(val)
                    if candidate is not None:
                        ids.add(candidate)
                elif (key == "claim_ids" or key.endswith("_claim_ids")) and isinstance(val, list):
                    ids.update(v for item in val if (v := integer(item)) is not None)
                elif key == "labels" and isinstance(val, dict):
                    ids.update(v for item, label in val.items()
                               if label in LABELS and (v := integer(item)) is not None)
                visit(val)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(value)
    return ids, texts


def audit_history(root, dev):
    """Exclude recorded/reserved dev IDs and exact claim text already in the repo."""
    root = Path(root)
    dev_ids = {claim["id"] for claim in dev}
    text_ids = {}
    for claim in dev:
        text_ids.setdefault(normalize(claim["claim"]), set()).add(claim["id"])
    excluded, sources = set(), {}
    skipped = {PLAN, OUTPUT, ANALYSIS, REPORT}
    paths = set()
    for folder in ("experiments", "results"):
        paths.update((root / folder).rglob("*.json"))
    paths.update(root.glob("*.py"))
    paths.update(root.glob("*.md"))
    for folder in ("experiments", "docs", "tests", ".claude/skills"):
        for suffix in ("*.py", "*.md"):
            paths.update((root / folder).rglob(suffix))
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        if name in skipped:
            continue
        text = path.read_text(encoding="utf-8")
        matches = set()
        if path.suffix == ".json":
            ids, seen_text = recorded_claims(json.loads(text))
            matches |= ids & dev_ids
            for item in seen_text:
                matches |= text_ids.get(item, set())
        else:
            normalized = normalize(text)
            for claim_text, ids in text_ids.items():
                if claim_text in normalized:
                    matches |= ids
        excluded |= matches
        sources[name] = {"sha256": comparison.sha256(path), "matched_dev_claim_ids": sorted(matches)}
    return {"excluded_claim_ids": sorted(excluded), "sources": sources,
            "scope": "Recorded/reserved IDs in project JSON plus exact claim text in project code/docs; off-repository use is not observable"}


def select_claims(dev, excluded, seed=SEED, per_label=PER_LABEL):
    rng = random.Random(seed)
    selected, pools = [], {}
    for label in LABELS:
        pool = sorted((r for r in dev if r["id"] not in excluded and claim_label(r) == label),
                      key=lambda row: row["id"])
        pools[label] = len(pool)
        if len(pool) < per_label:
            raise ValueError(f"Need {per_label} unused {label} claims; found {len(pool)}.")
        selected.extend(rng.sample(pool, per_label))
    rng.shuffle(selected)
    return selected, pools


def analysis_policy():
    return {
        "primary_metrics": list(PRIMARY_METRICS), "planned_comparisons": list(COMPARISONS),
        "point_estimate": "Micro F1 from pooled counts over all attempts, including failures",
        "bootstrap": {
            "method": "Paired percentile bootstrap, stratified by gold claim class",
            "unit": "Whole claim; both repetitions and all four conditions travel together",
            "draws": 50000, "seed": 47, "batch_size": 500,
            "descriptive_confidence_level": 0.95, "family_alpha": 0.05,
            "multiplicity": "Bonferroni across 3 planned comparisons x 2 primary metrics",
            "family_size": 6, "adjusted_confidence_level": 1 - 0.05 / 6,
        },
        "decision_rule": "Evidence of higher F1 on both primary metrics requires both adjusted difference intervals strictly above zero; both below zero favors the reference; otherwise inconclusive or mixed",
        "other_metrics": "Descriptive: precision, recall, NEI outcomes, failures, searches, abstracts, latency and SDK-estimated cost",
        "no_automatic_winner": "The primary-metric conclusion does not replace review of NEI false flags and processing failures",
        "limitations": [
            "Balanced subset of 60 unused dev claims, not a full SciFact benchmark result",
            "Unused refers to recorded project history; papers/topics can overlap development data",
            "Bootstrap clusters repetitions by claim; dependence between different claims about the same paper is not modeled",
            "Intervals are approximate; an interval containing zero does not establish equivalence",
        ],
    }


def require_frozen_commit(root, plan, plan_path):
    """Require the manifest and executable sources/results to be committed locally."""
    root = Path(root)
    names = {Path(plan_path).relative_to(root).as_posix()} | set(plan["source_sha256"])
    for name in sorted(names):
        if name.startswith("data/"):
            continue
        result = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root, capture_output=True)
        if result.returncode != 0 or result.stdout != (root / name).read_bytes():
            raise RuntimeError(f"Commit the frozen file before running: {name}")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def build_count_cube(runs, claim_ids, repetitions):
    import numpy as np
    positions = {claim_id: index for index, claim_id in enumerate(claim_ids)}
    condition_index = {name: index for index, name in enumerate(CONDITIONS)}
    cube = np.zeros((len(claim_ids), len(CONDITIONS), len(COUNT_FIELDS)), dtype=np.int64)
    seen = set()
    for row in runs:
        key = comparison.run_key(row)
        if key in seen or row["status"] not in comparison.TERMINAL:
            raise ValueError("Duplicate or unfinished attempt in bootstrap input.")
        if row["repetition"] not in range(1, repetitions + 1):
            raise ValueError("Unexpected repetition in bootstrap input.")
        seen.add(key)
        cube[positions[row["claim_id"]], condition_index[row["condition"]]] += [
            row["score_counts"][field] for field in COUNT_FIELDS]
    if len(seen) != len(claim_ids) * len(CONDITIONS) * repetitions:
        raise ValueError("Bootstrap requires every scheduled attempt, including failures.")
    return cube


def f1_from_counts(totals):
    import numpy as np
    counts = np.asarray(totals).reshape(*np.asarray(totals).shape[:-1], 2, 3)
    denominator = counts[..., 1] + counts[..., 2]
    return np.divide(2 * counts[..., 0], denominator,
                     out=np.zeros(denominator.shape, dtype=float), where=denominator != 0)


def bootstrap_intervals(runs, sample, policy):
    import numpy as np
    ids = sample["claim_ids"]
    cube = build_count_cube(runs, ids, sample["repetitions_per_condition"])
    settings = policy["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    groups = [np.asarray([i for i, claim_id in enumerate(ids)
                          if sample["labels"][str(claim_id)] == label]) for label in LABELS]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Each bootstrap stratum must contain claims.")
    draws = settings["draws"]
    distribution = np.empty((draws, len(CONDITIONS), 2), dtype=float)
    for start in range(0, draws, settings["batch_size"]):
        count = min(settings["batch_size"], draws - start)
        indices = np.concatenate([group[rng.integers(0, len(group), size=(count, len(group)))]
                                  for group in groups], axis=1)
        distribution[start:start + count] = f1_from_counts(cube[indices].sum(axis=1))
    point = f1_from_counts(cube.sum(axis=0))
    def interval(values, level):
        tail = (1 - level) / 2
        low, high = np.quantile(values, [tail, 1 - tail])
        return {"low": float(low), "high": float(high), "confidence_level": level}
    result = {"settings": settings, "conditions": {}, "comparisons": {}}
    for index, name in enumerate(CONDITIONS):
        result["conditions"][name] = {
            metric: {"point": float(point[index, m]),
                     "ci95": interval(distribution[:, index, m], settings["descriptive_confidence_level"])}
            for m, metric in enumerate(PRIMARY_METRICS)}
    for contrast in policy["planned_comparisons"]:
        candidate, reference = CONDITIONS.index(contrast["candidate"]), CONDITIONS.index(contrast["reference"])
        differences = distribution[:, candidate] - distribution[:, reference]
        entry = {**contrast, "metrics": {}}
        for m, metric in enumerate(PRIMARY_METRICS):
            entry["metrics"][metric] = {
                "difference": float(point[candidate, m] - point[reference, m]),
                "ci95": interval(differences[:, m], settings["descriptive_confidence_level"]),
                "adjusted_ci": interval(differences[:, m], settings["adjusted_confidence_level"]),
            }
        bounds = [entry["metrics"][metric]["adjusted_ci"] for metric in PRIMARY_METRICS]
        entry["conclusion"] = ("candidate_higher_on_both_primary_metrics" if all(x["low"] > 0 for x in bounds)
                               else "reference_higher_on_both_primary_metrics" if all(x["high"] < 0 for x in bounds)
                               else "inconclusive_or_mixed")
        result["comparisons"][contrast["name"]] = entry
    return result
