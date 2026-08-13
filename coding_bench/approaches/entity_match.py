"""Entity match: run NER on both sides, then score a code by how much of it
the note actually mentions.

A rewrite of `automated_coding/approaches/entity_match.py` against this
package's Predictor protocol. The idea is unchanged and the scoring is
unchanged; what is new is evidence spans, explicit parameters instead of
environment variables, and a scoring loop that does not re-embed the catalogue
once per note.

## Why bidirectional matching, rather than mention against description

The obvious version of this, and the one the original package tried first, is to
find entities in the note and match each against whole code descriptions. It
scored 0.158. The failure is a shape mismatch: a mention is two or three words
and a descriptor is a clause, so in SapBERT's space the descriptors cluster
together and stop discriminating between each other.

Running NER on the descriptor too fixes the shape. Both sides become sets of
short entity strings, which is what the encoder is good at comparing. A code
then scores by **coverage**: what fraction of the entities in its own descriptor
have a semantically close entity somewhere in the note. That asks the right
question. "Does this note mention the things this code is about" is nearer to
what a coder decides than "is this note broadly similar to this descriptor".

The rewrite scored 0.298 against 0.158 for the mention-to-description version,
which is the reason this approach is worth porting and the other one is not.

## What it costs to run

Heavier than `embed_match` or `retrieval`. It needs scispaCy with
`en_core_sci_lg` (around 800MB) and SapBERT, and it runs NER over every note
plus every descriptor. The descriptor side is cached across notes, so the per
note cost is one NER pass and one encode. Install with:

    pip install -e 'coding_bench[entity]'
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

The model wheel is not on PyPI, which is why it is a separate line rather than a
dependency. A run that cannot import it fails loudly at construction rather than
silently scoring zero.

## Read the number against the label space

Like every approach here, this is only meaningful when the candidates carry
descriptions. The committed CPT label space is codes only, because the
descriptors are AMA copyright, so a public CPT run has nothing to run NER over
on the code side and will score near zero. Inside Modal the descriptions load
from the restricted volume and the number means something. ICD-10 ships
descriptions and is fine anywhere.
"""

from __future__ import annotations

import re
import time

from coding_bench.approaches.base import Candidate, Note, Prediction

SCISPACY_MODEL = "en_core_sci_lg"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def fallback_entities(text: str) -> list[str]:
    """Split on punctuation when NER finds nothing in a descriptor.

    Short official descriptors are written in a register the biomedical models
    were not trained on, and a clause like "office or other outpatient visit"
    can come back with no entities at all. Falling back to punctuation segments
    keeps the code in the running instead of scoring it zero for a tokenizer's
    opinion. Note text never needs this; descriptors often do.
    """
    segments = []
    for piece in re.split(r"[;,./()]", text):
        words = [w for w in _WORD.findall(piece) if len(w) >= 3]
        if words:
            segments.append(" ".join(words))
    return segments


class EntityMatchPredictor:
    """Score a code by the share of its descriptor the note mentions."""

    name = "entity_match"
    version = "v1"

    def __init__(
        self,
        entity_similarity: float = 0.80,
        min_coverage: float = 0.50,
        top_k: int = 2,
        scispacy_model: str = SCISPACY_MODEL,
        encoder_name: str = SAPBERT_MODEL,
        device: str = "cpu",
    ):
        # All three thresholds are the original's tuned values, fitted on the
        # same 312 notes they were scored on. There is no held out split in this
        # dataset. Treat them as fitted, not as defaults that transfer.
        self.entity_similarity = entity_similarity
        self.min_coverage = min_coverage
        self.top_k = top_k
        self.scispacy_model = scispacy_model
        self.encoder_name = encoder_name
        self.device = device
        self._nlp = None
        self._model = None
        self._cache_key: tuple | None = None
        self._codes: list[str] = []
        self._code_entity_matrix = None
        self._code_entity_owner: list[int] = []

    @property
    def model_id(self) -> str:
        # Both halves, because the pairing is the approach. Reporting only the
        # encoder would make two very different pipelines look like one.
        return f"{self.scispacy_model}+{self.encoder_name}"

    def _load_ner(self):
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load(self.scispacy_model)
            except OSError as exc:
                raise RuntimeError(
                    f"{self.scispacy_model} is not installed. It is not on PyPI; see the "
                    f"install line at the top of this file. Failing here rather than "
                    f"falling back, because a silent fallback would report a score for a "
                    f"pipeline that never ran."
                ) from exc
        return self._nlp

    def _load_encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.encoder_name, device=self.device)
        return self._model

    def entities(self, text: str, allow_fallback: bool = False) -> list[tuple[str, int, int]]:
        """Entity strings with their character offsets in `text`.

        Offsets are carried because they are the evidence span. An approach that
        returns codes without them cannot be scored on the evidence metric, and
        a coder reviewing the suggestion has nothing to look at.
        """
        doc = self._load_ner()(text)
        found = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.text.strip()]
        if found or not allow_fallback:
            return found
        # Fallback segments have no reliable offsets, so they are located by
        # search rather than guessed at.
        out = []
        for segment in fallback_entities(text):
            at = text.find(segment)
            out.append((segment, at, at + len(segment)) if at != -1 else (segment, -1, -1))
        return out

    def _encode_candidates(self, candidates: list[Candidate]):
        """NER and encode every descriptor once, then reuse across notes."""
        key = tuple(sorted((c.code, c.description or "") for c in candidates))
        if key == self._cache_key:
            return self._codes, self._code_entity_matrix, self._code_entity_owner

        model = self._load_encoder()
        codes: list[str] = []
        texts: list[str] = []
        owner: list[int] = []
        for index, candidate in enumerate(candidates):
            codes.append(candidate.code)
            description = candidate.description or ""
            for text, _start, _end in self.entities(description, allow_fallback=True):
                texts.append(text)
                owner.append(index)

        matrix = (
            model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            if texts
            else None
        )
        self._cache_key, self._codes = key, codes
        self._code_entity_matrix, self._code_entity_owner = matrix, owner
        return codes, matrix, owner

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction:
        import numpy as np

        start = time.perf_counter()
        if not candidates:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        codes, code_matrix, owner = self._encode_candidates(candidates)
        note_entities = self.entities(note.text)
        if code_matrix is None or not note_entities:
            return Prediction(codes={}, latency_s=time.perf_counter() - start)

        model = self._load_encoder()
        note_matrix = model.encode(
            [text for text, _s, _e in note_entities],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = code_matrix @ note_matrix.T  # code entities by note entities

        best = similarity.max(axis=1)
        best_note_entity = similarity.argmax(axis=1)
        matched = best >= self.entity_similarity

        # Coverage per code, plus the single best matched entity as the citation.
        totals = np.zeros(len(codes))
        hits = np.zeros(len(codes))
        best_score = np.full(len(codes), -1.0)
        best_span = [None] * len(codes)

        for position, code_index in enumerate(owner):
            totals[code_index] += 1
            if not matched[position]:
                continue
            hits[code_index] += 1
            if best[position] > best_score[code_index]:
                best_score[code_index] = best[position]
                _text, span_start, span_end = note_entities[int(best_note_entity[position])]
                best_span[code_index] = (span_start, span_end) if span_start >= 0 else None

        coverage = np.divide(hits, totals, out=np.zeros(len(codes)), where=totals > 0)

        predicted: dict[str, tuple[int, int] | None] = {}
        for code_index in np.argsort(-coverage)[: self.top_k]:
            if coverage[code_index] < self.min_coverage or totals[code_index] == 0:
                break
            predicted[codes[code_index]] = best_span[code_index]

        return Prediction(codes=predicted, latency_s=time.perf_counter() - start)
