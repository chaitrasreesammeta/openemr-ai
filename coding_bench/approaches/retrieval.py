"""Embedding retrieval baseline. CPU only, no API, no GPU.

This is the floor every LLM has to clear. It matters more than a floor usually
does here, because retrieval is cheap, private, and auditable, so an LLM that
only matches it is not worth deploying for this task.

The note is scored against code descriptions chunk by chunk rather than whole.
A discharge summary runs to thousands of characters and one embedding of the
whole thing washes out the single sentence that supports a tail code. Chunking
also gives the evidence span for free: the best matching chunk is the citation.
"""

from __future__ import annotations

import re
import time

from coding_bench.approaches.base import Candidate, Note, Prediction

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def split_chunks(text: str, target_chars: int = 400, overlap: int = 80) -> list[tuple[int, int]]:
    """Character spans covering the note, split on line breaks where possible.

    Returns offsets rather than strings so that a matched chunk is already an
    evidence span.
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_chars, length)
        if end < length:
            window = text[start:end]
            # Prefer a paragraph or line boundary in the last third of the window.
            for pattern in ("\n\n", "\n", ". "):
                cut = window.rfind(pattern, int(target_chars * 0.6))
                if cut != -1:
                    end = start + cut + len(pattern)
                    break
        spans.append((start, end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return spans


class RetrievalPredictor:
    """Cosine similarity between note chunks and code descriptions."""

    name = "retrieval"
    version = "v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 10,
        threshold: float = 0.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.threshold = threshold
        self.device = device
        self._model = None
        self._cache_key: tuple | None = None
        self._candidate_matrix = None
        self._candidate_codes: list[str] = []

    @property
    def model_id(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode_candidates(self, candidates: list[Candidate]):
        """Encode the candidate descriptions, reusing the work across notes."""
        key = tuple(sorted((c.code, c.description or "") for c in candidates))
        if key == self._cache_key:
            return self._candidate_codes, self._candidate_matrix

        model = self._load()
        codes = [c.code for c in candidates]
        # A code with no description is still searchable by its own string, which
        # is weak but honest. It is what happens for CPT in public runs.
        texts = [f"{c.code}: {c.description}" if c.description else c.code for c in candidates]
        matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        self._cache_key, self._candidate_codes, self._candidate_matrix = key, codes, matrix
        return codes, matrix

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction:
        import numpy as np

        start = time.perf_counter()
        if not candidates:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        model = self._load()
        codes, candidate_matrix = self._encode_candidates(candidates)

        spans = split_chunks(note.text)
        chunk_texts = [note.text[a:b] for a, b in spans]
        if not chunk_texts:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        chunk_matrix = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
        similarity = chunk_matrix @ candidate_matrix.T  # chunks by codes

        best_per_code = similarity.max(axis=0)
        best_chunk = similarity.argmax(axis=0)

        order = np.argsort(-best_per_code)[: self.top_k]
        predicted: dict[str, tuple[int, int] | None] = {}
        for index in order:
            if best_per_code[index] < self.threshold:
                continue
            predicted[codes[index]] = spans[int(best_chunk[index])]

        return Prediction(codes=predicted, latency_s=time.perf_counter() - start)
