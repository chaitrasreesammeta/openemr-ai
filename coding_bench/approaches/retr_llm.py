"""Retrieval shortlists, then the LLM selects.

This is the shape a real deployment takes, because the full catalogue is far
larger than the 673 codes in the ICD-10 gold set, and no prompt holds it. The
benchmark's job is to show what that shortlisting costs: recall the retriever
throws away is recall the LLM can never recover, so the ceiling is reported
alongside the score.
"""

from __future__ import annotations

import time

from coding_bench.approaches.base import Candidate, Note, Prediction
from coding_bench.approaches.llm import LLMPredictor
from coding_bench.approaches.retrieval import RetrievalPredictor


class RetrievalThenLLM:
    """Shortlist with embeddings, decide with a model."""

    name = "retr_llm"

    def __init__(self, retriever: RetrievalPredictor, llm: LLMPredictor, shortlist: int = 50):
        self.retriever = retriever
        self.llm = llm
        self.shortlist = shortlist
        self.version = f"retr:{retriever.version}+llm:{llm.version}"

    @property
    def model_id(self) -> str:
        return self.llm.model_id

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction:
        start = time.perf_counter()
        if not candidates:
            return self.llm.predict(note, candidates)

        by_code = {candidate.code: candidate for candidate in candidates}

        # Widen the retriever to shortlist depth for this call only, so the same
        # retriever instance keeps its encoded candidate cache across notes.
        original_k = self.retriever.top_k
        self.retriever.top_k = self.shortlist
        try:
            retrieved = self.retriever.predict(note, candidates)
        finally:
            self.retriever.top_k = original_k

        narrowed = [by_code[code] for code in retrieved.codes if code in by_code]
        prediction = self.llm.predict(note, narrowed)

        # Recall the shortlist made unreachable, recorded per note so the cost of
        # the retrieval stage is visible rather than folded into the model's score.
        gold = set(note.gold_codes)
        reachable = gold & set(retrieved.codes)
        prediction.usage = dict(prediction.usage)
        prediction.usage["shortlist_size"] = len(narrowed)
        prediction.usage["shortlist_recall_pct"] = round(
            100 * len(reachable) / len(gold) if gold else 100.0
        )
        prediction.latency_s = time.perf_counter() - start
        return prediction
