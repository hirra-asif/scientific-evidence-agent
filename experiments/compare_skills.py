import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from agent import main as run_agent
from scoring import score_claim, summarize_scores

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_claims(split, claim_ids):
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Claim IDs must not contain duplicates.")

    path = PROJECT_DIR / "data" / f"claims_{split}.jsonl"
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    by_id = {item["id"]: item for item in records}
    missing = set(claim_ids) - set(by_id)
    if missing:
        raise ValueError(f"Unknown claim IDs: {sorted(missing)}")

    return [by_id[claim_id] for claim_id in claim_ids]


def build_summary(runs, requested_pairs):
    successful_ids = {
        mode: {
            row["claim_id"]
            for row in runs
            if row["skill"] == mode and row["status"] == "ok"
        }
        for mode in ("off", "on")
    }

    # Compare scores on exactly the same successfully completed claims.
    paired_ids = successful_ids["off"] & successful_ids["on"]

    summary = {
        "requested_pairs": requested_pairs,
        "successful_pairs": len(paired_ids),
        "scoring_population": "Claims successful in both modes",
        "modes": {},
    }

    for mode in ("off", "on"):
        rows = [row for row in runs if row["skill"] == mode]
        scored = [row for row in rows if row["claim_id"] in paired_ids]

        costs = [
            row["metadata"]["estimated_cost_usd"]
            for row in rows
            if row["metadata"].get("estimated_cost_usd") is not None
        ]

        summary["modes"][mode] = {
            "attempted": len(rows),
            "failures": sum(row["status"] == "error" for row in rows),
            "metrics_on_successful_pairs": (
                summarize_scores([row["score_counts"] for row in scored])
                if scored else None
            ),
            "mean_elapsed_seconds": (
                round(
                    sum(row["elapsed_seconds"] for row in rows) / len(rows),
                    3,
                )
                if rows else None
            ),
            "reported_estimated_cost_usd": (
                sum(costs) if costs else None
            ),
            "runs_without_cost": len(rows) - len(costs),
        }

    return summary


def save_results(path, results):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


async def main(split, claim_ids):
    claims = load_claims(split, claim_ids)
    started_at = datetime.now(timezone.utc)

    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (
        f"skill_comparison_{started_at:%Y%m%d_%H%M%S_%f}.json"
    )

    # Fingerprints identify the code and Skill used for this experiment.
    source_files = [
        "agent.py",
        "corpus_tools.py",
        "scoring.py",
        "experiments/compare_skills.py",
        ".claude/skills/scientific-evidence/SKILL.md",
    ]

    results = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "split": split,
        "claim_ids": claim_ids,
        "mode_order": "off/on for first claim, then alternating",
        "package_versions": {
            name: version(name)
            for name in ("claude-agent-sdk", "anthropic", "pydantic")
        },
        "source_sha256": {
            name: hashlib.sha256(
                (PROJECT_DIR / name).read_bytes()
            ).hexdigest()
            for name in source_files
        },
        "claims": [
            {
                "id": claim["id"],
                "claim": claim["claim"],
                "gold_evidence": claim["evidence"],
            }
            for claim in claims
        ],
        "runs": [],
    }

    save_results(output_path, results)
    print(f"Saving results to: {output_path}", flush=True)

    for index, claim in enumerate(claims):
        # Alternate the order to avoid always running one mode first.
        modes = ("off", "on") if index % 2 == 0 else ("on", "off")

        for mode in modes:
            print(
                f"\nCLAIM {index + 1}/{len(claims)} "
                f"(ID {claim['id']}) | Skill {mode}",
                flush=True,
            )

            metadata = {"estimated_cost_usd": None}
            row = {
                "claim_id": claim["id"],
                "skill": mode,
                "metadata": metadata,
            }
            start = time.perf_counter()

            try:
                # The agent receives only the claim text.
                # Gold labels and evidence stay in the evaluator.
                report = await run_agent(
                    claim=claim["claim"],
                    use_skill=(mode == "on"),
                    run_metadata=metadata,
                )

                row["report"] = report.model_dump()
                row["score_counts"] = dict(
                    score_claim(claim, row["report"])
                )
                row["status"] = "ok"

            except Exception as error:
                # A processing failure is never treated as an abstention.
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"RUN FAILED: {row['error']}", flush=True)

            row["elapsed_seconds"] = round(
                time.perf_counter() - start, 3
            )

            results["runs"].append(row)
            results["summary"] = build_summary(
                results["runs"], len(claims)
            )

            # Preserve completed runs after every attempt.
            save_results(output_path, results)

    results["status"] = "finished"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(output_path, results)

    print("\nCOMPARISON SUMMARY:")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["train", "dev"], default="train"
    )
    parser.add_argument(
        "--claim-ids", type=int, nargs="+", required=True
    )
    args = parser.parse_args()

    asyncio.run(main(args.split, args.claim_ids))