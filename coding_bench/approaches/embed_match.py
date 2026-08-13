"""Embedding match: every code description against every chunk of the note.

A rewrite of `automated_coding/approaches/embed_match.py` against this package's
Predictor protocol, not a port. What carries over is the idea and the operating
point; what is new is evidence spans, no environment variable configuration,
and parameters that land in the run manifest and the cache key where a reader
can see them.

## What this is, next to `retrieval.py`

`retrieval.py` embeds each code description once, embeds the note in chunks, and
returns the top ten codes by cosine. This differs in three ways, each of which
was worth points in the original ablation:

  * **Several descriptions per code, scored by the best one.** A code carries a
    long official descriptor, and may also carry a short clinical form and
    synonyms. Averaging them dilutes: the one phrasing that actually appears in
    the note gets pulled toward the ones that do not. Taking the maximum lets a
    single good match fire the code.
  * **A similarity threshold, not just a rank cut.** `retrieval.py` returns ten
    codes for a note whose gold answer averages one. That is a recall probe, not
    a predictor. A threshold makes it possible to be wrong by saying nothing,
    which is what a deployable baseline has to be able to do.
  * **The retrieval prefix BGE was trained with.** Descriptions are the query
    side and note chunks are the passage side. The prefix is not decoration; it
    is the asymmetry the encoder was fine tuned for.

## What is missing, and what it is worth

The original's best configuration, 0.524 micro F1 on CPT, came from enriching
each description with UMLS synonyms plus LOINC and RadLex terms. Unenriched, the
same approach scored 0.484, and that is what this defaults to.

The three fetchers that build those tables were retired with the rest of
`automated_coding/` and are recoverable from git history:

    git show e460fab:automated_coding/scripts/fetch_umls_synonyms.py
    git show e460fab:automated_coding/scripts/fetch_loinc_radlex.py
    git show e460fab:automated_coding/scripts/fetch_umls_global.py

`variants` is the seam for that. Pass a mapping of code to extra phrasings and
this becomes the enriched version; pass nothing and it is honest about being the
floor. Porting the fetch scripts is the follow up, and it is the difference
between a baseline that is worth quoting and one that is only worth beating.

CPT has a second problem the number has to be read against: descriptors are AMA
copyright, so the committed CPT label space is codes only. Against codes alone
this approach is matching note text to the string "99213" and will score near
zero. It is only meaningful for CPT when run inside Modal, where the descriptions
load from the restricted volume. ICD-10 ships descriptions and is fine anywhere.
"""

from __future__ import annotations

import time

from coding_bench.approaches.base import Candidate, Note, Prediction
from coding_bench.approaches.retrieval import DEFAULT_MODEL, split_chunks

# BGE was trained asymmetrically: a retrieval query carries an instruction
# prefix and the passage does not. The descriptions are the queries here.
QUERY_PREFIX = "Represent this clinical code description for retrieval: "


class EmbedMatchPredictor:
    """Max cosine between note chunks and any phrasing of a code."""

    name = "embed_match"
    version = "v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.60,
        top_k: int = 2,
        variants: dict[str, list[str]] | None = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        # 0.60 and 2 are the original's tuned values, and they were tuned on the
        # same 312 notes they were then scored on. There is no held out split in
        # this dataset, so treat both as fitted to it rather than as defaults
        # that transfer.
        self.threshold = threshold
        self.top_k = top_k
        self.variants = variants or {}
        self.device = device
        self._model = None
        self._cache_key: tuple | None = None
        self._variant_matrix = None
        self._variant_owner: list[int] = []
        self._codes: list[str] = []

    @property
    def model_id(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def phrasings(self, candidate: Candidate) -> list[str]:
        """Every way this code might appear, best first.

        The official descriptor leads because it is the one phrasing that is
        always present. A code with no descriptor falls back to its own string,
        which is weak and is why the CPT caveat above matters.
        """
        seen: set[str] = set()
        out: list[str] = []
        for phrase in [candidate.description or candidate.code, *self.variants.get(candidate.code, [])]:
            phrase = (phrase or "").strip()
            key = phrase.lower()
            if phrase and key not in seen:
                seen.add(key)
                out.append(phrase)
        return out

    def _encode_candidates(self, candidates: list[Candidate]):
        """Encode every phrasing of every code, reused across notes.

        Flattened into one matrix with an owner index per row, so the maximum
        over phrasings is a scatter over that index rather than a Python loop
        per code. At the full ICD-10 catalogue that is 673 codes and the loop
        would run once per note.
        """
        key = tuple(sorted((c.code, c.description or "") for c in candidates))
        if key == self._cache_key:
            return self._codes, self._variant_matrix, self._variant_owner

        model = self._load()
        codes: list[str] = []
        texts: list[str] = []
        owner: list[int] = []
        for index, candidate in enumerate(candidates):
            codes.append(candidate.code)
            for phrase in self.phrasings(candidate):
                texts.append(QUERY_PREFIX + phrase)
                owner.append(index)

        matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._cache_key, self._codes = key, codes
        self._variant_matrix, self._variant_owner = matrix, owner
        return codes, matrix, owner

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction:
        import numpy as np

        start = time.perf_counter()
        if not candidates:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        codes, variant_matrix, owner = self._encode_candidates(candidates)
        spans = split_chunks(note.text)
        if not spans:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        model = self._load()
        chunks = model.encode(
            [note.text[a:b] for a, b in spans],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = chunks @ variant_matrix.T  # chunks by phrasings

        # Best (chunk, phrasing) pair per code. np.maximum.at scatters the
        # per-phrasing maxima onto their owning code in one pass.
        owner_index = np.asarray(owner)
        best_per_phrasing = similarity.max(axis=0)
        best_chunk_per_phrasing = similarity.argmax(axis=0)

        best = np.full(len(codes), -1.0)
        np.maximum.at(best, owner_index, best_per_phrasing)

        # Which phrasing won, so the evidence span is the chunk that matched it
        # rather than the chunk that matched some other phrasing of the code.
        winning_chunk = np.zeros(len(codes), dtype=int)
        for position, code_index in enumerate(owner_index):
            if best_per_phrasing[position] >= best[code_index]:
                winning_chunk[code_index] = best_chunk_per_phrasing[position]

        predicted: dict[str, tuple[int, int] | None] = {}
        for code_index in np.argsort(-best)[: self.top_k]:
            if best[code_index] < self.threshold:
                # Sorted descending, so nothing after this clears it either.
                break
            predicted[codes[code_index]] = spans[int(winning_chunk[code_index])]

        return Prediction(codes=predicted, latency_s=time.perf_counter() - start)
