import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import corpus_tools as corpus


class TestCorpusLoading(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "corpus.jsonl"
        documents = [
            {"doc_id": 1, "title": "Raremarker", "abstract": ["First sentence.", "Second sentence."]},
            {"doc_id": 2, "title": "Raremarker", "abstract": ["First sentence.", "Second sentence."]},
            *[{"doc_id": i, "title": f"Other topic {i}", "abstract": ["Different research."]}
              for i in range(3, 8)],
        ]
        self.path.write_text("\n".join(json.dumps(doc) for doc in documents))
        self.patcher = patch.object(corpus, "DATA_PATH", self.path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        corpus._load_documents.cache_clear()
        corpus._load_index.cache_clear()
        self.addCleanup(corpus._load_documents.cache_clear)
        self.addCleanup(corpus._load_index.cache_clear)

    def test_agent_import_succeeds_without_a_data_directory(self):
        root = Path(corpus.__file__).resolve().parent
        for name in ("agent.py", "corpus_tools.py"):
            shutil.copy2(root / name, Path(self.directory.name) / name)
        result = subprocess.run([
            sys.executable, "-c",
            "import agent, corpus_tools; assert corpus_tools._load_documents.cache_info().currsize == 0",
        ], cwd=self.directory.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_read_does_not_build_the_search_index_and_returns_numbered_sentences(self):
        result = corpus.read_abstract("1")
        self.assertEqual(result["sentences"], [
            {"sentence_id": 0, "text": "First sentence."},
            {"sentence_id": 1, "text": "Second sentence."},
        ])
        self.assertEqual(corpus._load_index.cache_info().currsize, 0)

    def test_search_preserves_ties_and_filters_zero_score_documents(self):
        hits = corpus.search_abstracts("RAREMARKER!", top_k=5)
        self.assertEqual([hit["document_id"] for hit in hits], ["1", "2"])
        self.assertEqual(hits[0]["score"], hits[1]["score"])
        self.assertEqual(corpus.search_abstracts("absentterm"), [])

    def test_concurrent_first_reads_share_one_cached_load(self):
        with patch("builtins.open", wraps=open) as opening:
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(corpus.read_abstract, ["1"] * 12))
        self.assertEqual(opening.call_count, 1)
        self.assertTrue(all(row == rows[0] for row in rows))

    def test_search_index_is_created_once_and_legacy_attributes_still_work(self):
        with patch.object(corpus, "BM25Okapi", wraps=corpus.BM25Okapi) as build:
            corpus.search_abstracts("raremarker")
            corpus.search_abstracts("research")
            self.assertIs(corpus.index, corpus.index)
            self.assertEqual(len(corpus.documents), 7)
            self.assertEqual(corpus.documents_by_id["1"]["doc_id"], 1)
        self.assertEqual(build.call_count, 1)

    def test_missing_data_fails_when_used_with_setup_guidance(self):
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "README Setup"):
            corpus.read_abstract("1")

    def test_invalid_top_k_does_not_load_data(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "top_k"):
            corpus.search_abstracts("test", top_k=0)

    def test_empty_or_duplicate_corpus_is_rejected(self):
        for text in ("", '{"doc_id": 1}\n{"doc_id": 1}\n'):
            self.path.write_text(text)
            corpus._load_documents.cache_clear()
            with self.subTest(text=text), self.assertRaises(ValueError):
                corpus.read_abstract("1")
