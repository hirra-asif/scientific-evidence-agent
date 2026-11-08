import unittest

from scoring import score_claim, summarize_scores


def make_report(verdict, sentences, document_id="paper"):
    return {
        "assessments": [
            {
                "document_id": document_id,
                "verdict": verdict,
                "evidence_sentence_ids": sentences,
            }
        ]
    }


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.claim = {
            "evidence": {
                "paper": [{"label": "SUPPORT", "sentences": [4]}]
            }
        }

    def test_exact_evidence_gets_full_credit(self):
        counts = score_claim(self.claim, make_report("SUPPORT", [4]))
        metrics = summarize_scores([counts])

        for name in (
            "document_labels",
            "abstract_with_evidence",
            "evidence_sentences",
        ):
            self.assertEqual(metrics[name]["f1"], 1.0)

    def test_extra_sentence_reduces_precision(self):
        counts = score_claim(self.claim, make_report("SUPPORT", [3, 4]))
        metrics = summarize_scores([counts])["evidence_sentences"]

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_partial_rationale_gets_no_evidence_credit(self):
        self.claim["evidence"]["paper"][0]["sentences"] = [3, 4]
        counts = score_claim(self.claim, make_report("SUPPORT", [4]))

        self.assertEqual(counts["correct_document_labels"], 1)
        self.assertEqual(counts["correct_abstracts_with_evidence"], 0)
        self.assertEqual(counts["correct_sentences"], 0)

    def test_wrong_label_gets_no_credit(self):
        counts = score_claim(self.claim, make_report("CONTRADICT", [4]))

        self.assertEqual(counts["correct_document_labels"], 0)
        self.assertEqual(counts["correct_sentences"], 0)

    def test_wrong_document_gets_no_credit(self):
        counts = score_claim(
            self.claim, make_report("SUPPORT", [4], "wrong-paper")
        )

        self.assertEqual(counts["predicted_documents"], 1)
        self.assertEqual(counts["correct_document_labels"], 0)
        self.assertEqual(counts["correct_sentences"], 0)

    def test_alternative_rationale_is_accepted(self):
        self.claim["evidence"]["paper"] = [
            {"label": "SUPPORT", "sentences": [3, 4]},
            {"label": "SUPPORT", "sentences": [7]},
        ]
        counts = score_claim(self.claim, make_report("SUPPORT", [7]))

        self.assertEqual(counts["correct_abstracts_with_evidence"], 1)
        self.assertEqual(counts["correct_sentences"], 1)
        self.assertEqual(counts["gold_sentences"], 3)

    def test_abstract_score_uses_first_three_sentences(self):
        counts = score_claim(
            self.claim, make_report("SUPPORT", [0, 1, 2, 4])
        )

        self.assertEqual(counts["correct_abstracts_with_evidence"], 0)
        self.assertEqual(counts["correct_sentences"], 1)

    def test_long_gold_rationale_increases_abstract_limit(self):
        self.claim["evidence"]["paper"][0]["sentences"] = [0, 1, 2, 3]
        counts = score_claim(
            self.claim, make_report("SUPPORT", [0, 1, 2, 3])
        )

        self.assertEqual(counts["correct_abstracts_with_evidence"], 1)
        self.assertEqual(counts["correct_sentences"], 4)

    def test_nei_requires_no_positive_predictions(self):
        gold_nei = {"evidence": {}}
        abstention = make_report("NOT_ENOUGH_INFO", [])
        positive = make_report("SUPPORT", [4])

        self.assertEqual(
            score_claim(gold_nei, abstention)["correct_nei"], 1
        )
        self.assertEqual(
            score_claim(gold_nei, positive)["correct_nei"], 0
        )
        self.assertEqual(
            score_claim(self.claim, abstention)["correct_document_labels"],
            0,
        )

    def test_duplicate_references_are_rejected(self):
        duplicate_documents = make_report("SUPPORT", [4])
        duplicate_documents["assessments"] *= 2
        duplicate_sentences = make_report("SUPPORT", [4, 4])

        for report in (duplicate_documents, duplicate_sentences):
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    score_claim(self.claim, report)

    def test_scores_pool_counts_across_claims(self):
        first = score_claim(self.claim, make_report("SUPPORT", [4]))
        second = score_claim(
            self.claim, make_report("SUPPORT", [0, 1, 4])
        )
        metrics = summarize_scores([first, second])["evidence_sentences"]

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_scifact_documented_example(self):
        claim = {
            "evidence": {
                "11": [
                    {"label": "SUPPORT", "sentences": [0, 1]},
                    {"label": "SUPPORT", "sentences": [11]},
                ],
                "15": [{"label": "SUPPORT", "sentences": [4]}],
            }
        }
        report = make_report("SUPPORT", [1, 11, 13], "11")
        report["assessments"].extend(
            make_report("CONTRADICT", [18, 20], "16")["assessments"]
        )
        metrics = summarize_scores([score_claim(claim, report)])

        self.assertEqual(metrics["abstract_with_evidence"]["f1"], 0.5)
        self.assertEqual(metrics["evidence_sentences"]["precision"], 0.2)
        self.assertEqual(metrics["evidence_sentences"]["recall"], 0.25)
        self.assertAlmostEqual(metrics["evidence_sentences"]["f1"], 2 / 9)


if __name__ == "__main__":
    unittest.main()