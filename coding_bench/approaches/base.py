"""The contract every approach and every adapter implements.

This protocol is also the contract the inference service exposes, so that the
service is a thin deployment of a benchmarked approach rather than a second,
divergent implementation. Two things in it are load bearing:

  * a prediction is code to evidence span, not a bare code list, because a coder
    reviewing a suggestion needs the text that justifies it
  * truncation is a typed error, never an empty result, because "the model ran
    out of tokens" and "no codes apply" are opposite findings and must never
    arrive looking the same
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class Truncated(RuntimeError):
    """The provider stopped generation before the model finished.

    Adapters raise this instead of returning a partial parse. The runner records
    it against the note and scores that note as an empty prediction, but the
    truncation rate is reported separately so the two are never conflated.
    """

    def __init__(self, model_id: str, stop_reason: str, produced_tokens: int | None = None):
        self.model_id = model_id
        self.stop_reason = stop_reason
        self.produced_tokens = produced_tokens
        super().__init__(
            f"{model_id} stopped with reason {stop_reason!r}"
            + (f" after {produced_tokens} tokens" if produced_tokens else "")
        )


@dataclass(frozen=True)
class Note:
    note_id: str
    text: str
    gold_codes: tuple[str, ...] = ()
    # code -> gold evidence spans from MDACE
    gold_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    category: str = ""
    description: str = ""


@dataclass(frozen=True)
class Candidate:
    """One code offered to an approach, with whatever description we may use."""

    code: str
    description: str | None = None


@dataclass
class Prediction:
    """What an approach returns for one note."""

    # code -> the character span in the note that justifies it, or None
    codes: dict[str, tuple[int, int] | None] = field(default_factory=dict)
    truncated: bool = False
    latency_s: float = 0.0
    # Provider token counts, for cost per note. No text ever goes here.
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Completion:
    """One raw response from a model adapter."""

    text: str
    stop_reason: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """A model, wherever it is hosted.

    Implementations must map the provider's own stop reason onto a plain string
    and raise Truncated when generation was cut short.
    """

    model_id: str

    def complete(self, system: str, user: str, max_tokens: int) -> Completion: ...


@runtime_checkable
class Predictor(Protocol):
    """An approach: a way of turning a note into codes with evidence."""

    name: str
    version: str

    def predict(self, note: Note, candidates: list[Candidate] | None = None) -> Prediction: ...


def locate_span(text: str, quote: str, hint: int = 0) -> tuple[int, int] | None:
    """Turn a quoted snippet back into character offsets in the note.

    Models cite evidence by quoting. Offsets are what the evidence metric scores
    against, so the quote has to be found in the note. An unfindable quote is
    reported as no evidence rather than as a guess, because a fabricated offset
    would score as a near miss when it is really a hallucination.
    """
    quote = (quote or "").strip()
    if not quote:
        return None

    index = text.find(quote, hint)
    if index == -1:
        index = text.find(quote)
    if index != -1:
        return index, index + len(quote)

    # Whitespace in clinical notes is not reproduced faithfully by models, so
    # fall back to a whitespace insensitive search before giving up.
    collapsed = " ".join(quote.split())
    if collapsed and collapsed != quote:
        return locate_span(text, collapsed, hint)

    return None
