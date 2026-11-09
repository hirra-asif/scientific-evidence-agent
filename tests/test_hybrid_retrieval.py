import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_retrieval import (
    check_embeddings, choose_weight, fuse_ranks, ranked_indices, read_cache,
)


class TestHybridRetrieval(unittest.TestCase):
    def test_ranking_filters_nonpositive_and_keeps_corpus_order_for_ties(self):
        scores = np.array([0.0, 4.0, -1.0, 4.0, 2.0])
        self.assertEqual(ranked_indices(scores, 100, True), [1, 3, 4])
        self.assertEqual(ranked_indices(scores, 2), [1, 3])

    def test_fusion_rewards_agreement_and_missing_ranks_contribute_zero(self):
        result = fuse_ranks([2, 0], [1, 0], 0.5)
        self.assertEqual([index for index, _ in result], [0, 1, 2])
        self.assertAlmostEqual(result[0][1], 1 / 62)
        self.assertAlmostEqual(result[1][1], 0.5 / 61)
        self.assertAlmostEqual(result[2][1], 0.5 / 61)

    def test_weight_changes_winner_and_zero_weight_adds_no_candidates(self):
        self.assertEqual(fuse_ranks([0], [1], 0.25)[0][0], 0)
        self.assertEqual(fuse_ranks([0], [1], 0.75)[0][0], 1)
        self.assertEqual(fuse_ranks([0], [1], 0), [(0, 1 / 61)])
        self.assertEqual(fuse_ranks([], [1], 0.5), [(1, 0.5 / 61)])

    def test_duplicate_candidates_and_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            fuse_ranks([0, 0], [1], 0.5)
        with self.assertRaises(ValueError):
            fuse_ranks([0], [1], 1.1)
        with self.assertRaises(ValueError):
            ranked_indices(np.array([1.0, np.nan]), 3)

    def test_selection_uses_primary_metric_then_both_tie_breakers(self):
        def metrics(at3, at5):
            return {"at_3": {"recall": at3}, "at_5": {"recall": at5}}
        self.assertEqual(choose_weight({0.25: metrics(.8, .95), .5: metrics(.81, .9)}), .5)
        self.assertEqual(choose_weight({0.25: metrics(.8, .9), .5: metrics(.8, .95)}), .5)
        self.assertEqual(choose_weight({.75: metrics(.8, .9), .25: metrics(.8, .9)}), .25)

    def test_embeddings_reject_bad_shape_values_dtype_and_norm(self):
        good = np.eye(2, dtype=np.float32)
        check_embeddings(good, 2, 2)
        bad_arrays = [good[:1], good.astype(np.float64), good * 2,
                      np.full((2, 2), np.nan, dtype=np.float32)]
        for values in bad_arrays:
            with self.subTest(values=values), self.assertRaises(ValueError):
                check_embeddings(values, 2, 2)

    def test_cache_preserves_values_and_rejects_different_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            metadata = {"identity": {"revision": "a"}, "document_ids": ["10", "20"],
                        "token_lengths": {"texts": 2}}
            np.savez(path, embeddings=np.eye(2, dtype=np.float32), metadata=json.dumps(metadata))
            values, stats = read_cache(path, {"revision": "a"}, ["10", "20"], 2)
            np.testing.assert_array_equal(values, np.eye(2, dtype=np.float32))
            self.assertEqual(stats, {"texts": 2})
            with self.assertRaises(ValueError):
                read_cache(path, {"revision": "b"}, ["10", "20"], 2)

    def test_cache_rejects_document_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            metadata = {"identity": {}, "document_ids": ["10", "20"], "token_lengths": {}}
            np.savez(path, embeddings=np.eye(2, dtype=np.float32), metadata=json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_cache(path, {}, ["20", "10"], 2)


if __name__ == "__main__":
    unittest.main()
