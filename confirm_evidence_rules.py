import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import main as run_agent
from compare_skills import load_claims, save_results
from scoring import score_claim, summarize_scores

PROJECT_DIR = Path(__file__).resolve().parent
CONDITIONS = ("current", "minimal")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_document_label_match(claim, report):
    expected = {
        (doc_id, evidence["label"])
        for doc_id, evidence_sets in claim["evidence"].items()
        for evidence in evidence_sets
    }
    predicted = {
        (item["document_id"], item["verdict"])
        for item in report["assessments"]
        if item["verdict"] != "NOT_ENOUGH_INFO"
    }
    return predicted == expected


def summarize(runs, requested_pairs):
    successful = {
        condition: {
            row["claim_id"] for row in runs
            if row["condition"] == condition and row["status"] == "ok"
        }
        for condition in CONDITIONS
    }
    paired = successful["current"] & successful["minimal"]
    summary = {
        "requested_pairs": requested_pairs,
        "successful_pairs": len(paired),
        "scoring_population": "Claims successful in both conditions",
        "conditions": {},
    }

    for condition in CONDITIONS:
        rows = [r for r in runs if r["condition"] == condition]
        scored = [r for r in rows if r["claim_id"] in paired]
        nei_rows = [r for r in rows if r["gold_is_nei"]]
        costs = [
            r["metadata"]["estimated_cost_usd"] for r in rows
            if r["metadata"].get("estimated_cost_usd") is not None
        ]

        def correct(row):
            return row["status"] == "ok" and row["exact_document_label_match"]

        summary["conditions"][condition] = {
            "attempted": len(rows),
            "failures": sum(r["status"] == "error" for r in rows),
            "metrics_on_successful_pairs": (
                summarize_scores([r["score_counts"] for r in scored])
                if scored else None
            ),
            "exact_document_label_match_rate_all_attempts": (
                sum(correct(r) for r in rows) / len(rows) if rows else None
            ),
            "nei_abstention_rate_all_attempts": (
                sum(correct(r) for r in nei_rows) / len(nei_rows)
                if nei_rows else None
            ),
            "mean_elapsed_seconds": (
                round(sum(r["elapsed_seconds"] for r in rows) / len(rows), 3)
                if rows else None
            ),
            "reported_estimated_cost_usd": sum(costs) if costs else None,
            "runs_without_cost": len(rows) - len(costs),
        }
    return summary


async def main():
    manifest_path = PROJECT_DIR / "confirmation_claims.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["split"] != "dev":
        raise ValueError("This comparison requires development claims.")
    if sha256(PROJECT_DIR / "data/claims_dev.jsonl") != manifest["dataset_sha256"]:
        raise ValueError("The development dataset has changed.")
    if sha256(PROJECT_DIR / "data/corpus.jsonl") != manifest["corpus_sha256"]:
        raise ValueError("The abstract corpus has changed.")

    claim_ids = manifest["claim_ids"]
    if len(claim_ids) != 30 or len(set(claim_ids)) != 30:
        raise ValueError("Expected 30 unique frozen claims.")
    if set(claim_ids) & set(manifest["excluded_claim_ids"]):
        raise ValueError("The sample contains previously used claims.")

    skill_path = PROJECT_DIR / ".claude/skills/scientific-evidence/SKILL.md"
    if skill_path.read_text(encoding="utf-8") != manifest["skill_texts"]["current"]:
        raise ValueError("Your current Skill differs from the frozen version.")

    claims = load_claims("dev", claim_ids)
    started = datetime.now(timezone.utc)
    output = PROJECT_DIR / "results" / (
        f"confirmation_{started:%Y%m%d_%H%M%S_%f}.json"
    )
    output.parent.mkdir(exist_ok=True)

    source_files = [
        "agent.py", "corpus_tools.py", "scoring.py",
        "compare_skills.py", "confirm_evidence_rules.py",
        "confirmation_claims.json",
    ]
    fingerprints = {name: sha256(PROJECT_DIR / name) for name in source_files}

    results = {
        "status": "running",
        "experiment": "Agent SDK comparison: current versus minimal evidence Skill",
        "started_at": started.isoformat(),
        "manifest": manifest,
        "condition_order": "current/minimal for first claim, then alternating",
        "source_sha256": fingerprints,
        "package_versions": {
            name: version(name)
            for name in ("claude-agent-sdk", "anthropic", "pydantic")
        },
        "claims": claims,
        "runs": [],
    }
    save_results(output, results)
    print(f"Saving results to: {output}", flush=True)

    for index, claim in enumerate(claims):
        order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]

        for condition in order:
            # Stop if code or the frozen experiment is edited during the run.
            for name, expected_hash in fingerprints.items():
                if sha256(PROJECT_DIR / name) != expected_hash:
                    raise RuntimeError(f"{name} changed during the comparison.")

            metadata = {"estimated_cost_usd": None}
            row = {
                "claim_id": claim["id"],
                "condition": condition,
                "gold_is_nei": not claim["evidence"],
                "metadata": metadata,
            }
            print(
                f"\nCLAIM {index + 1}/{len(claims)} (ID {claim['id']})"
                f" | Skill: {condition}",
                flush=True,
            )
            start = time.perf_counter()

            try:
                # Both versions use the same Skill name in a clean workspace.
                # Corpus access still goes through our existing Python tools.
                with TemporaryDirectory(prefix="scifact-confirmation-") as folder:
                    path = (
                        Path(folder) / ".claude/skills"
                        / "scientific-evidence/SKILL.md"
                    )
                    path.parent.mkdir(parents=True)
                    path.write_text(
                        manifest["skill_texts"][condition], encoding="utf-8"
                    )

                    report = await run_agent(
                        claim=claim["claim"],
                        use_skill=True,
                        run_metadata=metadata,
                        skill_workspace=folder,
                    )

                row["report"] = report.model_dump()
                row["score_counts"] = dict(score_claim(claim, row["report"]))
                row["exact_document_label_match"] = exact_document_label_match(
                    claim, row["report"]
                )
                row["status"] = "ok"

            except Exception as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"RUN FAILED: {row['error']}", flush=True)

            row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            results["runs"].append(row)
            results["summary"] = summarize(results["runs"], len(claims))
            save_results(output, results)

    results["status"] = "finished"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(output, results)

    print("\nCONFIRMATION SUMMARY:")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSaved results: {output}")


if __name__ == "__main__":
    asyncio.run(main())