"""Tests for local Sentence Transformer retrieval and standalone TF-IDF."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pdf_processor import AcademicPDFProcessor
from rag_engine import (
    AcademicRAGEngine,
    DEFAULT_EMBEDDING_MODEL,
    SemanticRetrievalError,
    _load_sentence_transformer_model,
)


class FakeSentenceTransformer:
    """Deterministic local stand-in that makes semantic ranking testable offline."""

    def __init__(self, vectors, fail_on_query=None):
        self.vectors = vectors
        self.fail_on_query = fail_on_query
        self.encoded = []

    def encode(self, texts, **_kwargs):
        values = [texts] if isinstance(texts, str) else list(texts)
        self.encoded.extend(values)
        if len(values) == 1 and values[0] == self.fail_on_query:
            raise RuntimeError("local model query failure")
        return np.asarray([self.vectors[text] for text in values], dtype=np.float32)


class SemanticRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "text": "Distributed practice and spaced recall help learners retain course material over time.",
                "page": 9,
                "source_doc": "learning.pdf",
                "doc_id": "doc-learning",
                "chunk_id": 12,
            },
            {
                "text": "Backup scheduling and recovery preserve files after system failure.",
                "page": 3,
                "source_doc": "storage.pdf",
                "doc_id": "doc-storage",
                "chunk_id": 4,
            },
            {
                "text": "Distributed practice improves retention for students.",
                "page": 2,
                "source_doc": "inactive.pdf",
                "doc_id": "doc-inactive",
                "chunk_id": 1,
            },
        ]
        self.query = "Which study approach helps remember material for longer?"
        self.vectors = {
            self.chunks[0]["text"]: [1.0, 0.0],
            self.chunks[1]["text"]: [0.0, 1.0],
            self.chunks[2]["text"]: [0.9, 0.1],
            self.query: [0.98, 0.2],
        }

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_is_loaded_once_and_reused(self, model_factory):
        _load_sentence_transformer_model.cache_clear()
        try:
            first = _load_sentence_transformer_model(DEFAULT_EMBEDDING_MODEL)
            second = _load_sentence_transformer_model(DEFAULT_EMBEDDING_MODEL)
            self.assertIs(first, second)
            model_factory.assert_called_once_with(DEFAULT_EMBEDDING_MODEL, device="cpu")
        finally:
            _load_sentence_transformer_model.cache_clear()

    @patch("rag_engine._load_sentence_transformer_model")
    def test_semantic_paraphrase_cosine_order_metadata_and_document_isolation(self, load_model):
        model = FakeSentenceTransformer(self.vectors)
        load_model.return_value = model
        engine = AcademicRAGEngine(retrieval_method="semantic", min_score_threshold=0.1)
        engine.build_index(self.chunks)

        results = engine.search(
            self.query,
            active_doc_ids=["doc-learning", "doc-storage"],
        )

        self.assertEqual(engine.model_name, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(engine.retrieval_mode, "Semantic search (local: all-MiniLM-L6-v2)")
        self.assertEqual(results[0]["doc_id"], "doc-learning")
        self.assertEqual(results[0]["source_doc"], "learning.pdf")
        self.assertEqual(results[0]["page"], 9)
        self.assertEqual(results[0]["score_kind"], "cosine_similarity")
        self.assertEqual([item["score"] for item in results], sorted(
            [item["score"] for item in results], reverse=True
        ))
        self.assertNotIn("doc-inactive", [item["doc_id"] for item in results])
        self.assertEqual(load_model.call_count, 1)

    @patch("rag_engine._load_sentence_transformer_model")
    def test_index_chunks_from_uploaded_text_with_page_and_filename_metadata(self, load_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "course_notes.txt"
            source.write_text(
                "Distributed practice and spaced recall help learners retain course material over time.",
                encoding="utf-8",
            )
            document = AcademicPDFProcessor().process_file(str(source), "course_notes.txt")
        chunk_texts = [chunk["text"] for chunk in document["chunks"]]
        model_vectors = {text: [1.0, 0.0] for text in chunk_texts}
        model_vectors[self.query] = [1.0, 0.0]
        model = FakeSentenceTransformer(model_vectors)
        load_model.return_value = model

        engine = AcademicRAGEngine(retrieval_method="semantic")
        engine.build_index(document["chunks"])
        results = engine.search(self.query, active_doc_ids=[document["doc_id"]])

        self.assertIsNotNone(engine.dense_embeddings)
        self.assertTrue(results)
        self.assertEqual(results[0]["source_doc"], "course_notes.txt")
        self.assertEqual(results[0]["page"], 1)

    def test_tfidf_runs_independently_when_explicitly_selected(self):
        engine = AcademicRAGEngine(retrieval_method="tfidf")
        engine.build_index(self.chunks[:2])
        results = engine.search(
            "backup scheduling preserves files after failures",
            active_doc_ids=["doc-storage"],
        )

        self.assertEqual(engine.retrieval_mode, "TF-IDF keyword search (local)")
        self.assertTrue(results)
        self.assertEqual(results[0]["score_kind"], "tfidf_cosine_similarity")
        self.assertEqual(results[0]["source_doc"], "storage.pdf")
        self.assertEqual(results[0]["page"], 3)

    @patch("rag_engine._load_sentence_transformer_model", side_effect=OSError("model download unavailable"))
    def test_model_load_failure_is_visible_and_does_not_fall_back(self, _load_model):
        engine = AcademicRAGEngine(retrieval_method="semantic")
        engine.build_index(self.chunks[:2])

        self.assertEqual(engine.retrieval_mode, "Semantic search unavailable")
        with self.assertRaisesRegex(SemanticRetrievalError, "retry semantic indexing"):
            engine.search(self.query)
        self.assertIsNone(engine.tfidf_vectorizer)

    @patch("rag_engine._load_sentence_transformer_model")
    def test_failed_index_can_be_retried_without_reupload(self, load_model):
        load_model.side_effect = [
            OSError("temporary model download failure"),
            FakeSentenceTransformer(self.vectors),
        ]
        engine = AcademicRAGEngine(retrieval_method="semantic")
        engine.build_index(self.chunks[:2])
        self.assertIsNotNone(engine.retrieval_error)

        engine.build_index(self.chunks[:2])

        self.assertIsNone(engine.retrieval_error)
        self.assertIsNotNone(engine.dense_embeddings)
        self.assertEqual(load_model.call_count, 2)

    @patch("rag_engine._load_sentence_transformer_model")
    def test_query_encoding_error_is_visible_and_never_uses_tfidf(self, load_model):
        model = FakeSentenceTransformer(self.vectors, fail_on_query=self.query)
        load_model.return_value = model
        engine = AcademicRAGEngine(retrieval_method="semantic")
        engine.build_index(self.chunks[:2])

        with self.assertRaisesRegex(SemanticRetrievalError, "Retry semantic search"):
            engine.search(self.query)
        self.assertEqual(engine.retrieval_mode, "Semantic search unavailable")

    @patch("rag_engine._load_sentence_transformer_model")
    def test_minimum_cosine_threshold_is_configurable(self, load_model):
        load_model.return_value = FakeSentenceTransformer(self.vectors)
        engine = AcademicRAGEngine(retrieval_method="semantic", min_score_threshold=0.95)
        engine.build_index(self.chunks[:2])

        results = engine.search(self.query, active_doc_ids=["doc-learning", "doc-storage"])

        self.assertEqual([item["doc_id"] for item in results], ["doc-learning"])
        self.assertGreaterEqual(results[0]["score"], 0.95)

    @patch("rag_engine._load_sentence_transformer_model")
    def test_blank_question_and_empty_index_are_handled(self, load_model):
        engine = AcademicRAGEngine(retrieval_method="semantic")
        engine.build_index([])
        self.assertEqual(engine.search("anything"), [])
        self.assertEqual(engine.search("  "), [])
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
