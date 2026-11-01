"""Document retrieval using local Sentence Transformers or independent TF-IDF."""

from functools import lru_cache
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.25
TFIDF_THRESHOLD = 0.05


class RetrievalError(RuntimeError):
    """Raised when the selected retrieval method cannot build or query."""


class SemanticRetrievalError(RetrievalError):
    """Raised when local semantic retrieval cannot build or query its index."""


@lru_cache(maxsize=1)
def _load_sentence_transformer_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load from cache first, download only when the model is not cached."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception as cache_error:
        try:
            return SentenceTransformer(model_name, device="cpu", local_files_only=False)
        except Exception as download_error:
            raise RuntimeError(
                f"Cached model load failed ({type(cache_error).__name__}: {cache_error}); "
                f"download failed ({type(download_error).__name__}: {download_error})."
            ) from download_error


class AcademicRAGEngine:
    """Retrieve page-cited chunks using either semantic or TF-IDF ranking."""

    def __init__(
        self,
        retrieval_method: str = "tfidf",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        min_score_threshold: Optional[float] = None,
    ):
        if retrieval_method not in {"semantic", "tfidf"}:
            raise ValueError("retrieval_method must be 'semantic' or 'tfidf'.")
        self.retrieval_method = retrieval_method
        self.model_name = model_name
        self.min_score_threshold = (
            float(min_score_threshold)
            if min_score_threshold is not None
            else (SEMANTIC_THRESHOLD if retrieval_method == "semantic" else TFIDF_THRESHOLD)
        )
        if not 0.0 <= self.min_score_threshold <= 1.0:
            raise ValueError("min_score_threshold must be between 0.0 and 1.0.")
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self._model = None
        self._index_signature: Optional[str] = None
        self.retrieval_error: Optional[str] = None
        self.retrieval_mode = self._mode_label()

    def _mode_label(self) -> str:
        if self.retrieval_method == "semantic":
            return f"Semantic search (local: {self.model_name.rsplit('/', 1)[-1]})"
        return "TF-IDF keyword search (local)"

    @staticmethod
    def _signature(chunks: List[Dict[str, Any]], model_name: str) -> str:
        digest = hashlib.sha256(model_name.encode("utf-8"))
        for chunk in chunks:
            digest.update(str(chunk.get("doc_id", "")).encode("utf-8"))
            digest.update(str(chunk.get("source_doc", "")).encode("utf-8"))
            digest.update(str(chunk.get("page", "")).encode("utf-8"))
            digest.update(str(chunk.get("text", "")).encode("utf-8"))
        return digest.hexdigest()

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Index extracted chunks; semantic failures are surfaced, never hidden."""
        usable_chunks = [
            chunk for chunk in chunks
            if isinstance(chunk.get("text"), str) and chunk["text"].strip()
        ]
        signature = self._signature(usable_chunks, self.model_name)
        index_ready = (
            self.dense_embeddings is not None
            if self.retrieval_method == "semantic"
            else self.tfidf_vectorizer is not None and self.tfidf_matrix is not None
        )
        if signature == self._index_signature and self.chunks and index_ready:
            self.retrieval_error = None
            self.retrieval_mode = self._mode_label()
            return

        self.chunks = usable_chunks
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dense_embeddings = None
        self.retrieval_error = None
        self._index_signature = signature

        if not self.chunks:
            self.retrieval_mode = "No searchable text found in the selected document."
            return

        texts = [chunk["text"] for chunk in self.chunks]
        if self.retrieval_method == "tfidf":
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2), stop_words="english", max_df=0.95, min_df=1
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.retrieval_mode = self._mode_label()
            except Exception as error:
                self.retrieval_error = "TF-IDF index could not be built. Check that the document contains readable text."
                self.retrieval_mode = "TF-IDF unavailable"
                return
            return

        try:
            self._model = _load_sentence_transformer_model(self.model_name)
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
                raise ValueError("The embedding model returned an invalid document index.")
            self.dense_embeddings = vectors
            self.retrieval_mode = self._mode_label()
        except Exception as error:
            self.dense_embeddings = None
            cause = str(error).splitlines()[0][:240] or type(error).__name__
            self.retrieval_error = (
                "Local semantic model could not be loaded or indexed. Check the installation/network, "
                f"then retry semantic indexing or explicitly select TF-IDF. Cause: {cause}"
            )
            self.retrieval_mode = "Semantic search unavailable"

    def clear(self):
        """Clear document-specific index state while retaining the cached model."""
        self.chunks = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dense_embeddings = None
        self._index_signature = None
        self.retrieval_error = None
        self.retrieval_mode = self._mode_label()

    def search(
        self,
        query: str,
        top_k: int = 5,
        active_doc_ids: Optional[List[str]] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return highest-scoring passages, scoped to the active document IDs."""
        if not isinstance(query, str) or not query.strip():
            return []
        if self.retrieval_error:
            error_type = SemanticRetrievalError if self.retrieval_method == "semantic" else RetrievalError
            raise error_type(self.retrieval_error)
        if not self.chunks:
            return []

        if active_doc_ids is None:
            valid_indices = list(range(len(self.chunks)))
        else:
            allowed_set = set(active_doc_ids)
            valid_indices = [
                i for i, chunk in enumerate(self.chunks)
                if chunk.get("doc_id") in allowed_set or chunk.get("source_doc") in allowed_set
            ]
        if not valid_indices:
            return []

        if self.retrieval_method == "semantic":
            if self.dense_embeddings is None or self._model is None:
                raise SemanticRetrievalError(
                    "Semantic index is not ready. Retry semantic indexing or explicitly choose TF-IDF."
                )
            try:
                query_vector = self._model.encode(
                    [query.strip()],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                query_vector = np.asarray(query_vector, dtype=np.float32)
                if query_vector.ndim != 2 or query_vector.shape[0] != 1 or not np.isfinite(query_vector).all():
                    raise ValueError("The embedding model returned an invalid query vector.")
                # SentenceTransformer returns normalized vectors, so their
                # cosine similarity is their dot product. Compute it as a
                # bounded elementwise reduction to avoid noisy BLAS warnings
                # from sklearn's dense matrix multiplication on macOS.
                document_vectors = self.dense_embeddings[valid_indices].astype(
                    np.float64, copy=False
                )
                normalized_query = query_vector[0].astype(np.float64, copy=False)
                document_norms = np.linalg.norm(document_vectors, axis=1)
                query_norm = np.linalg.norm(normalized_query)
                denominators = document_norms * query_norm
                dot_products = np.einsum(
                    "ij,j->i", document_vectors, normalized_query, optimize=False
                )
                score_values = np.divide(
                    dot_products,
                    denominators,
                    out=np.zeros_like(dot_products),
                    where=denominators > 0,
                )
                score_values = np.clip(score_values, -1.0, 1.0)
                score_kind = "cosine_similarity"
            except Exception as error:
                self.retrieval_error = (
                    "The local semantic model could not encode this question. Retry semantic search "
                    "or explicitly select TF-IDF."
                )
                self.retrieval_mode = "Semantic search unavailable"
                raise SemanticRetrievalError(self.retrieval_error) from error
        else:
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                return []
            query_vector = self.tfidf_vectorizer.transform([query.strip()])
            score_values = cosine_similarity(
                query_vector, self.tfidf_matrix[valid_indices]
            ).flatten()
            score_kind = "tfidf_cosine_similarity"

        threshold = (
            self.min_score_threshold
            if min_score_threshold is None
            else float(min_score_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score_threshold must be between 0.0 and 1.0.")
        ranked = sorted(
            (
                (valid_indices[position], float(score))
                for position, score in enumerate(score_values)
                if float(score) >= threshold
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, int(top_k))]

        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "score_kind": score_kind,
                "text": chunk["text"],
                "page": chunk.get("page", 1),
                "source_doc": chunk.get("source_doc", "Document"),
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", idx + 1),
                "section": chunk.get("section", f"Page {chunk.get('page', 1)}"),
            })
        return results

    def get_document_summary_chunks(
        self, limit: int = 4, active_doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return representative chunks from the selected documents only."""
        if active_doc_ids is None:
            return self.chunks[:limit]
        allowed_set = set(active_doc_ids)
        return [
            chunk for chunk in self.chunks
            if chunk.get("doc_id") in allowed_set or chunk.get("source_doc") in allowed_set
        ][:limit]
