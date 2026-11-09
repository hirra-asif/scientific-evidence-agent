import unittest

from pydantic import ValidationError
from agent import AgentReport, validate_report


def make_report(verdict="SUPPORT", sentences=(0,), document_id="paper"):
    return AgentReport.model_validate({
        "assessments": [{
            "document_id": document_id,
            "verdict": verdict,
            "evidence_sentence_ids": list(sentences),
            "explanation": "Example explanation for a validation test.",
        }]
    })


class TestReportValidation(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "paper": {
                "sentences": [
                    {"sentence_id": 0, "text": "First sentence."},
                    {"sentence_id": 1, "text": "Second sentence."},
                ]
            }
        }

    def test_valid_verdict_and_evidence_combinations(self):
        cases = [
            ("SUPPORT", [0]),
            ("CONTRADICT", [1]),
            ("NOT_ENOUGH_INFO", []),
        ]
        for verdict, sentences in cases:
            with self.subTest(verdict=verdict):
                validate_report(make_report(verdict, sentences), self.documents)

    def test_no_abstracts_read_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not read any abstracts"):
            validate_report(make_report(), {})

    def test_unread_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_report(make_report(document_id="unread"), self.documents)

    def test_omitted_read_document_is_rejected(self):
        self.documents["second-paper"] = self.documents["paper"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_report(make_report(), self.documents)

    def test_empty_report_is_rejected(self):
        report = AgentReport(assessments=[])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_report(report, self.documents)

    def test_duplicate_documents_are_rejected(self):
        report = make_report()
        report.assessments.append(report.assessments[0].model_copy(deep=True))
        with self.assertRaisesRegex(ValueError, "Duplicate document"):
            validate_report(report, self.documents)

    def test_duplicate_sentences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate evidence"):
            validate_report(make_report(sentences=[0, 0]), self.documents)

    def test_positive_verdict_requires_evidence(self):
        for verdict in ("SUPPORT", "CONTRADICT"):
            with self.subTest(verdict=verdict):
                with self.assertRaisesRegex(ValueError, "requires evidence"):
                    validate_report(make_report(verdict, []), self.documents)

    def test_nei_must_have_no_evidence(self):
        with self.assertRaisesRegex(ValueError, "must have no evidence"):
            validate_report(make_report("NOT_ENOUGH_INFO", [0]), self.documents)

    def test_invalid_sentence_indices_are_rejected(self):
        for sentence_id in (-1, 2):
            with self.subTest(sentence_id=sentence_id):
                with self.assertRaisesRegex(ValueError, "Invalid sentence ID"):
                    validate_report(
                        make_report(sentences=[sentence_id]), self.documents
                    )

    def test_multiple_read_documents_are_accepted(self):
        self.documents["second-paper"] = self.documents["paper"]
        report = make_report()
        second = make_report("NOT_ENOUGH_INFO", [], "second-paper")
        report.assessments.extend(second.assessments)
        validate_report(report, self.documents)

    def test_unknown_verdict_is_rejected_by_schema(self):
        with self.assertRaises(ValidationError):
            make_report("MAYBE", [])


if __name__ == "__main__":
    unittest.main()