"""Offline checks for unused-claim selection, frozen execution and paired statistics."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from experiments import agent_comparison as comparison
from experiments import final_evaluation_common as common
from experiments.final_conditions import CONDITIONS, schedule_runs, summarize_runs


def synthetic_claim(claim_id, label):
    return {"id": claim_id, "claim": f"Synthetic claim number {claim_id}.",
            "evidence": {} if label == "NOT_ENOUGH_INFO" else {
                str(claim_id + 1000): [{"label": label, "sentences": [0]}]}}


def scored_fixture(variable=False):
    ids = list(range(1, 7))
    labels = {str(i): common.LABELS[(i - 1) // 2] for i in ids}
    sample = {"claim_ids": ids, "labels": labels, "repetitions_per_condition": 2, "split": "dev"}
    rows = []
    for job in schedule_runs(ids, 2):
        claim = synthetic_claim(job["claim_id"], labels[str(job["claim_id"])])
        positive = bool(claim["evidence"])
        correct = positive and (job["claim_id"] % 2 == 0 or job["repetition"] == 1)
        if variable and job["condition"] == "improved_haiku":
            correct = positive
        assessments = [{"document_id": str(claim["id"] + 1000),
                        "verdict": labels[str(claim["id"])] if correct else "NOT_ENOUGH_INFO",
                        "evidence_sentence_ids": [0] if correct else [], "explanation": "Test"}]
        row = {**job, "claim": claim["claim"], "gold_evidence": claim["evidence"],
               "elapsed_seconds": 1.0, "trace_complete": True, "metadata": {}}
        comparison.finish_row(row, claim, "ok", {"assessments": assessments})
        rows.append(row)
    return sample, rows


class TestFinalSelection(unittest.TestCase):
    def test_recorded_ids_cover_legacy_formats_without_treating_documents_as_claims(self):
        value = {"claim_ids": [1, "2"], "excluded_claim_ids": [3], "runs": [{"claim_id": 4}],
                 "cases": [{"claim": {"id": 5, "claim": "A known claim."}}],
                 "labels": {"6": "SUPPORT"}, "paper": {"id": 999, "document_id": 888}}
        ids, texts = common.recorded_claims(value)
        self.assertEqual(ids, {1, 2, 3, 4, 5, 6})
        self.assertEqual(texts, {"a known claim."})

    def test_history_audit_catches_id_and_text_exposure_and_excludes_its_own_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            (root / "experiments").mkdir()
            dev = [synthetic_claim(i, "SUPPORT") for i in (1, 2, 3)]
            (root / "results/old.json").write_text(json.dumps({"claim_id": 1, "document_id": "3"}))
            (root / "README.md").write_text(dev[1]["claim"].upper())
            (root / common.OUTPUT).write_text(json.dumps({"claim_id": 3}))
            audit = common.audit_history(root, dev)
            self.assertEqual(audit["excluded_claim_ids"], [1, 2])
            self.assertNotIn(common.OUTPUT, audit["sources"])

    def test_balanced_selection_is_reproducible_and_never_reuses_excluded_ids(self):
        records = [synthetic_claim(i, common.LABELS[(i - 1) // 30]) for i in range(1, 91)]
        excluded = {1, 31, 61}
        selected, pools = common.select_claims(records, excluded)
        repeated, _ = common.select_claims(list(reversed(records)), excluded)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 60)
        self.assertFalse({r["id"] for r in selected} & excluded)
        self.assertEqual(pools, {label: 29 for label in common.LABELS})

    def test_insufficient_unused_class_is_rejected_without_relaxing_the_rule(self):
        records = [synthetic_claim(i, "SUPPORT") for i in range(30)]
        with self.assertRaises(ValueError):
            common.select_claims(records, set())

    def test_shared_runner_adapter_is_idempotent_and_rejects_unrelated_edits(self):
        path = Path(__file__).resolve().parents[1] / "experiments/run_final_pilot.py"
        source = path.read_text()
        frozen = "867f516d5e1c11132197d4a2cd380dc0f083ae1153479496541fc37fc52131ef"
        adapted = common.adapt_runner(source, frozen)
        self.assertEqual(common.adapt_runner(adapted, frozen), adapted)
        with self.assertRaises(ValueError):
            common.adapt_runner(adapted + "\n# unrelated change\n", frozen)

    def test_final_schedule_has_480_unique_attempts_and_balances_each_position(self):
        jobs = schedule_runs(list(range(60)), 2)
        self.assertEqual(len(jobs), 480)
        self.assertEqual(len({comparison.run_key(job) for job in jobs}), 480)
        for repetition in (1, 2):
            rows = [job for job in jobs if job["repetition"] == repetition]
            for position in range(4):
                self.assertCountEqual([r["condition"] for r in rows[position::4]], list(CONDITIONS) * 15)
        self.assertEqual([j["condition"] for j in jobs[:4]], list(CONDITIONS))
        self.assertEqual([j["condition"] for j in jobs[240:244]], list(CONDITIONS[::-1]))


class TestPairedStatistics(unittest.TestCase):
    def setUp(self):
        self.policy = common.analysis_policy()
        self.policy["bootstrap"].update(draws=300, batch_size=50)

    def test_micro_f1_uses_pooled_counts_not_mean_per_claim_f1(self):
        counts = np.array([[1, 1, 1, 1, 1, 1], [0, 100, 1, 0, 100, 1]])
        value = common.f1_from_counts(counts.sum(axis=0))
        np.testing.assert_allclose(value, [2 / 103, 2 / 103])
        self.assertNotAlmostEqual(float(value[0]), 0.5)

    def test_both_repetitions_are_aggregated_into_one_claim_cluster(self):
        sample, rows = scored_fixture()
        cube = common.build_count_cube(rows, sample["claim_ids"], 2)
        self.assertEqual(cube.shape, (6, 4, 6))
        self.assertEqual(cube[0, 0].tolist(), [1, 1, 2, 1, 1, 2])

    def test_identical_paired_conditions_have_zero_difference_in_every_interval(self):
        sample, rows = scored_fixture()
        result = common.bootstrap_intervals(rows, sample, self.policy)
        for contrast in result["comparisons"].values():
            self.assertEqual(contrast["conclusion"], "inconclusive_or_mixed")
            for metric in contrast["metrics"].values():
                self.assertEqual(metric["difference"], 0)
                for interval in ("ci95", "adjusted_ci"):
                    self.assertEqual(metric[interval]["low"], 0)
                    self.assertEqual(metric[interval]["high"], 0)

    def test_seeded_intervals_reproduce_and_adjusted_intervals_are_no_narrower(self):
        sample, rows = scored_fixture(variable=True)
        result = common.bootstrap_intervals(rows, sample, self.policy)
        self.assertEqual(result, common.bootstrap_intervals(rows, sample, self.policy))
        for contrast in result["comparisons"].values():
            for metric in contrast["metrics"].values():
                self.assertLessEqual(metric["adjusted_ci"]["low"], metric["ci95"]["low"])
                self.assertGreaterEqual(metric["adjusted_ci"]["high"], metric["ci95"]["high"])
        self.assertEqual(self.policy["bootstrap"]["family_size"], 6)
        self.assertAlmostEqual(self.policy["bootstrap"]["adjusted_confidence_level"], 1 - 0.05 / 6)

    def test_missing_or_duplicate_attempts_are_rejected(self):
        sample, rows = scored_fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                common.build_count_cube(bad, sample["claim_ids"], 2)

    def test_bootstrap_point_estimates_match_the_existing_scorer(self):
        sample, rows = scored_fixture(variable=True)
        result = common.bootstrap_intervals(rows, sample, self.policy)
        summary = summarize_runs(rows, schedule_runs(sample["claim_ids"], 2))
        for name in CONDITIONS:
            scores = summary["conditions"][name]["metrics_all_attempts"]
            self.assertAlmostEqual(result["conditions"][name]["evidence_f1"]["point"], scores["evidence_sentences"]["f1"])
            self.assertAlmostEqual(result["conditions"][name]["document_verdict_f1"]["point"], scores["document_labels"]["f1"])


class TestFrozenCommit(unittest.TestCase):
    def test_uncommitted_files_block_execution_and_committed_files_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "experiments").mkdir()
            (root / "source.py").write_text("value = 1\n")
            plan = {"source_sha256": {"source.py": comparison.sha256(root / "source.py")}}
            path = root / common.PLAN
            path.write_text(json.dumps(plan))
            def git(*args):
                return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)
            git("init", "-q")
            git("config", "user.name", "Offline test")
            git("config", "user.email", "offline-test@example.invalid")
            with self.assertRaises(RuntimeError):
                common.require_frozen_commit(root, plan, path)
            git("add", "source.py", common.PLAN)
            git("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Test fixture")
            self.assertEqual(len(common.require_frozen_commit(root, plan, path)), 40)
            (root / "source.py").write_text("value = 2\n")
            with self.assertRaises(RuntimeError):
                common.require_frozen_commit(root, plan, path)


class TestFinalHarnessReuse(unittest.IsolatedAsyncioTestCase):
    async def test_final_plan_loader_output_and_printer_override_pilot_defaults(self):
        from experiments import run_final_pilot as runner
        sample, rows = scored_fixture()
        plan = {"split": "dev", "sample": sample, "schedule": schedule_runs(sample["claim_ids"], 2),
                "role": "Final dev evaluation"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, output_path = root / "dev-plan.json", root / "dev-results.json"
            comparison.save_json(plan_path, plan)
            results = {"plan": plan, "plan_sha256": comparison.sha256(plan_path), "runs": rows,
                       "status": "finished", "split": "dev"}
            comparison.save_json(output_path, results)
            verifier, loader, printer = Mock(), Mock(return_value=([], {})), Mock()
            with patch.object(runner, "verify_plan", side_effect=AssertionError("Pilot verifier used")):
                with patch.object(runner, "load_claims", side_effect=AssertionError("Training loader used")):
                    await runner.run(plan_path=plan_path, output_path=output_path,
                                     plan_verifier=verifier, claims_loader=loader,
                                     summary_printer=printer, progress_label="FINAL")
            verifier.assert_called_once_with(plan)
            loader.assert_called_once_with(sample)
            printer.assert_called_once()
            saved = comparison.read_json(output_path)
            self.assertEqual(saved["split"], "dev")
            self.assertEqual(saved["summary"]["completed_runs"], 48)


if __name__ == "__main__":
    unittest.main()
