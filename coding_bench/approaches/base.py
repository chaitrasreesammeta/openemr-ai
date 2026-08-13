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
    # The codes were recovered from a response whose JSON did not parse, so they
    # carry no evidence spans. Recorded rather than hidden: it is the difference
    # between a model that cited its work and one whose citation was unreadable.
    salvaged: bool = False


@dataclass
class Completion:
    """One raw response from a model adapter."""

    text: str
    stop_reason: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)


# Where a provider may put generated text when it is not in `content`. Ordered,
# so the first one present wins.
REASONING_CHANNELS = ("reasoning_content", "reasoning")


def _channel(message, name: str) -> str:
    """One channel, from a dict or from a provider SDK's response object."""
    if isinstance(message, dict):
        return message.get(name) or ""
    return getattr(message, name, None) or ""


def answer_text(message) -> str:
    """The generated text, wherever the provider decided to put it.

    OpenAI compatible servers split a reasoning model's output into channels:
    `content` carries the final answer and something like `reasoning_content`
    carries the thinking. A model that never leaves the thinking channel returns
    an empty `content` while having generated a page, and an adapter that reads
    only `content` records that as the model having nothing to say.

    That is not hypothetical, and it is not cheap. It cost 124 of 312 notes on
    Gemma 4 and 120 of 150 on Muse, every one of them scored as an empty
    prediction. See the note at the top of adapters/modal_gemma4_gguf.py.

    `content` wins whenever it holds anything, and is returned unchanged, so a
    response that answered normally is byte for byte what it was before. The
    fallback only fires where the alternative is discarding the response
    entirely, which is why it can be reasoned about one note at a time.

    Reading the thinking is not as good as reading an answer. The parser already
    expects to find the answer inside a reasoning trace, since extract_json
    takes the last object carrying codes rather than the first, but a model cut
    off mid thought has no answer anywhere and stays a truncation.

    Groq does not split channels for either model in this benchmark: their
    responses arrive whole in `content`, which was verified per note with
    scripts/groq_silence_probe.py. This runs there anyway, so that every adapter
    reads a response the same way and a provider changing its `reasoning_format`
    default cannot quietly cost a run.
    """
    content = _channel(message, "content")
    if content.strip():
        return content
    for channel in REASONING_CHANNELS:
        value = _channel(message, channel)
        if value.strip():
            return value
    return ""


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
