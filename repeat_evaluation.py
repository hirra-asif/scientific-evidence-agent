import argparse
import asyncio
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def verify_setup(baseline, manifest):
    if baseline.get("status") != "finished":
        raise ValueError("The baseline batch is not finished.")

    if (
        baseline["split"] != manifest["split"]
        or baseline["claim_ids"] != manifest["claim_ids"]
    ):
        raise ValueError(
            "The saved claim selection differs from the baseline."
        )

    dataset = (
        PROJECT_DIR / "data" / f"claims_{manifest['split']}.jsonl"
    )
    actual_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()

    if actual_hash != manifest["dataset_sha256"]:
        raise ValueError("The development dataset has changed.")

    for filename, expected_hash in baseline["source_sha256"].items():
        path = PROJECT_DIR / filename
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"{filename} has changed since the baseline. "
                "Stop here so we can check the difference."
            )

    for package, expected_version in baseline["package_versions"].items():
        installed_version = version(package)

        if installed_version != expected_version:
            raise ValueError(
                f"{package} version changed: "
                f"{expected_version} -> {installed_version}."
            )


async def main(baseline_path, repeats):
    baseline = json.loads(
        (PROJECT_DIR / baseline_path).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (PROJECT_DIR / "evaluation_claims.json").read_text(
            encoding="utf-8"
        )
    )

    verify_setup(baseline, manifest)

    # Import the existing evaluator after the setup checks.
    from compare_skills import main as run_comparison

    for index in range(repeats):
        verify_setup(baseline, manifest)

        print(
            f"\nADDITIONAL BATCH {index + 1}/{repeats}",
            flush=True,
        )

        # Run sequentially: the agent keeps per-run state in memory.
        await run_comparison(
            manifest["split"],
            manifest["claim_ids"],
        )

    print("\nAll additional batches finished.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    asyncio.run(main(args.baseline, args.repeats))