import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.evaluate_bm25 import summarize_retrieval
from experiments import validate_hybrid as validation


class TestHybridValidation(unittest.TestCase):
    def setUp(self):
        self.claims = {
            1: {"id": 1, "claim": "tuning claim", "evidence": {"10": []}},
            2: {"id": 2, "claim": "validation claim", "evidence": {"20": []}},
            3: {"id": 3, "claim": "unannotated claim", "evidence": {}},
        }
        self.split = {"schema_version": 1, "split": "train",
                      "tuning_claim_ids": [1], "validation_claim_ids": [2]}
        self.config = {"fusion": {"dense_weights": [0.25, 0.5, 0.75]}}

        def condition(document_id):
            rows = [{"claim_id": 1, "query": "tuning claim", "gold_document_ids": ["10"],
                     "results": [{"document_id": document_id}]}]
            return {"rows": rows, "summary": summarize_retrieval(rows)}

        self.tuning = {
            "status": "finished", "partition": "tuning", "dataset_split": "train",
            "configuration": copy.deepcopy(self.config), "bm25_baseline_reproduced": True,
            "claim_ids": [1], "bm25": condition("99"),
            "hybrids": [{"dense_weight": weight, **condition(doc)}
                        for weight, doc in ((0.25, "99"), (0.5, "10"), (0.75, "99"))],
            "selection": {"best_dense_weight": 0.5, "tuning_passed": True,
                          "validation_evaluated": False},
        }

    def test_selection_is_recovered_from_verified_tuning_rows(self):
        self.assertEqual(validation.selected_weight(
            self.tuning, self.config, self.split, self.claims,
        ), 0.5)

    def test_altered_winner_or_summary_is_rejected(self):
        for kind in ("winner", "summary"):
            tuning = copy.deepcopy(self.tuning)
            if kind == "winner":
                tuning["selection"]["best_dense_weight"] = 0.75
            else:
                tuning["hybrids"][1]["summary"]["at_3"]["recall"] = 0.0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validation.selected_weight(tuning, self.config, self.split, self.claims)

    def test_claim_and_paper_leakage_are_rejected(self):
        split = copy.deepcopy(self.split)
        split["validation_claim_ids"] = [1, 2]
        with self.assertRaises(ValueError):
            validation.validation_claims(split, self.claims)
        claims = copy.deepcopy(self.claims)
        claims[2]["evidence"] = {"10": []}
        with self.assertRaises(ValueError):
            validation.validation_claims(self.split, claims)

    def test_comparison_uses_only_validation_and_only_selected_weight(self):
        held_out = validation.validation_claims(self.split, self.claims)
        retriever = Mock()
        retriever.document_ids = ["20", "99"]
        retriever.embed_queries.return_value = (["embedding"], {"texts": 1})
        retriever.rankings.return_value = ([1], [0])
        retriever.results.return_value = [{"document_id": "20"}]
        search = Mock(return_value=[{"document_id": "99"}])
        with redirect_stdout(io.StringIO()):
            bm25, hybrid, _ = validation.compare(retriever, held_out, 0.5, search)
        retriever.embed_queries.assert_called_once_with(["validation claim"], progress=True)
        retriever.results.assert_called_once_with([1], [0], 0.5)
        search.assert_called_once_with("validation claim", top_k=5)
        self.assertEqual([row["claim_id"] for row in hybrid], [2])
        self.assertEqual(summarize_retrieval(bm25)["at_3"]["recall"], 0.0)
        self.assertEqual(summarize_retrieval(hybrid)["at_3"]["recall"], 1.0)

    def test_completed_validation_is_not_overwritten_or_reexecuted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / validation.OUTPUT_NAME
            output.parent.mkdir()
            output.write_text("original result")
            with patch.object(validation, "PROJECT_DIR", root):
                with patch.object(validation, "HybridRetriever") as retriever:
                    with self.assertRaises(FileExistsError):
                        validation.main(root / "unused.json")
                    retriever.assert_not_called()
            self.assertEqual(output.read_text(), "original result")


if __name__ == "__main__":
    unittest.main()
