"""Claude Sonnet 5 over the Anthropic Messages API.

Served over HTTP like the Groq models, so there is no Modal wrapper here and no
GPU. What is different from every other adapter in this package is the
governance question, which has to be settled before this file is ever run.

## Before running this: the note text goes to a third provider

The MIMIC-III data use agreement governs who may see the notes. Groq is the
only external provider this benchmark has been cleared for, and every Groq run
records `external_provider: "groq"` so that clearance stays answerable after
the fact. Running this adapter sends the same restricted note text to
Anthropic, and that is a decision about the DUA, not a decision about code.

Two things must be true first, and neither can be asserted by a run:

  * Anthropic is covered for MIMIC derived data under whatever agreement the
    workspace operates under, on the same footing Groq already is.
  * Zero data retention is enabled for the Anthropic organisation, matching the
    ZDR requirement this package already imposes on Groq. Claude Sonnet 5 is
    available under ZDR. Claude Fable 5 is not, which is one reason this
    adapter does not offer it.

Runs made through this file record `external_provider: "anthropic"` so a reader
in six months can tell which provider saw which notes.

## The API surface, which differs from the Groq models in four ways

1. **No temperature.** Claude Sonnet 5 rejects a non-default `temperature`,
   `top_p`, or `top_k` with a 400. Every other adapter here pins temperature to
   0 so a rerun reproduces the run; this one cannot. Results from this model are
   not greedy and two runs of the same note may differ. That is a real gap in
   comparability and it is recorded here rather than hidden.

2. **Truncation is `max_tokens`, not `length`.** The OpenAI compatible
   endpoints report a cut off generation as `finish_reason: "length"`. The
   Anthropic API reports `stop_reason: "max_tokens"`. An adapter that copied the
   Groq check verbatim would never raise Truncated and would score every cut off
   note as a model that found nothing, which is the exact confusion this package
   exists to prevent.

3. **Refusal is its own stop reason.** `stop_reason: "refusal"` arrives as a
   successful HTTP 200 with empty or partial content. It is raised here rather
   than returned, so it lands in the error rate instead of masquerading as a
   model that declined to code. A refusal is not transient, so a rerun will pay
   for it again; that is the right trade, because a model that will not do the
   task should show up as a failed run rather than a low score.

4. **Adaptive thinking is on by default.** Omitting the `thinking` parameter
   runs it, unlike the models this package started with. Thinking shares the
   `max_tokens` budget with the answer, so a budget sized for the answer alone
   will truncate. The package default of 16,384 has room; a smaller one does
   not. Depth is steered by the run's existing `reasoning_strength`, mapped onto
   Anthropic's `effort`, so the parameter that is already in the run manifest
   and the cache key keeps meaning what it says.

The answer arrives in `text` content blocks with thinking in separate blocks, so
the response is read the same way `approaches/base.py::answer_text` reads the
llama.cpp channels: prefer the answer, fall back to the reasoning only when
there is no answer at all. Same rule, different wire format.

## Cost

Claude Sonnet 5 is $3 per million input tokens and $15 per million output, with
an introductory $2/$10 through 2026-08-31. At gold candidates the ICD-10 prompts
are around 1k tokens, so 578 notes is a couple of dollars. At the full catalogue
they are around 11k, so the same run is closer to twenty. Neither is a GPU hour,
but neither is free, and `--limit` exists for iteration.

Environment:
    ANTHROPIC_API_KEY   required, via the `anthropic-api` Modal secret
"""

from __future__ import annotations

import os
import time

from coding_bench.approaches.base import Completion, Truncated

# Claude Sonnet 5. The id carries no date suffix, which is correct: the aliases
# are complete as written and appending a snapshot date is a 404.
SONNET_5 = "claude-sonnet-5"

MODELS = {
    "sonnet-5": SONNET_5,
}

# The run parameter this package already records, mapped onto Anthropic's
# effort levels. The names line up for the three the benchmark uses, so the
# manifest keeps meaning what it says without a second vocabulary.
EFFORT = {"low": "low", "medium": "medium", "high": "high"}


class AnthropicClient:
    """One Anthropic model behind the LLMClient protocol."""

    def __init__(
        self,
        model_id: str = SONNET_5,
        api_key: str | None = None,
        max_retries: int = 6,
        reasoning_strength: str | None = "medium",
        timeout_s: float = 600.0,
    ):
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        self.model_id = model_id
        self.max_retries = max_retries
        self.reasoning_strength = reasoning_strength
        # Ten minutes, because a full catalogue prompt with thinking on is a
        # long generation and a stalled socket must fail one note rather than
        # hang a worker and quietly drain the pool.
        self._client = Anthropic(api_key=key, timeout=timeout_s, max_retries=0)

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        effort = EFFORT.get(self.reasoning_strength or "")
        if effort:
            kwargs["output_config"] = {"effort": effort}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                # Streamed rather than a plain create. The SDK refuses a non
                # streaming request whose max_tokens it estimates will outrun
                # the HTTP timeout, and 16,384 with thinking on is over that
                # line. get_final_message gives back the same object a non
                # streaming call would have returned.
                with self._client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
            except (RateLimitError, APIStatusError, APIConnectionError) as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                wait = 2**attempt
                header = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    wait = max(wait, float(header.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
                time.sleep(min(wait, 60))
                continue

            elapsed = time.perf_counter() - start
            stop_reason = message.stop_reason or "unknown"

            if stop_reason == "max_tokens":
                raise Truncated(
                    self.model_id, stop_reason, produced_tokens=message.usage.output_tokens
                )

            if stop_reason == "refusal":
                # Deliberately an error rather than an empty answer. `category`
                # is informational and may be absent, so it is read defensively.
                details = getattr(message, "stop_details", None)
                category = getattr(details, "category", None) or "unspecified"
                raise RuntimeError(
                    f"{self.model_id} refused this note (category {category}). A refusal "
                    f"is a fact about the request, not a prediction, so it is recorded as "
                    f"an error rather than scored as finding no codes."
                )

            return Completion(
                text=answer_from_blocks(message.content),
                stop_reason=stop_reason,
                latency_s=elapsed,
                usage={
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                },
            )

        raise RuntimeError(
            f"{self.model_id} failed after {self.max_retries} attempts: {last_error}"
        )


def answer_from_blocks(blocks) -> str:
    """Flatten a content block list into the text the parser should read.

    The Anthropic API returns a list of typed blocks rather than one string, so
    `approaches/base.py::answer_text` cannot be applied to it directly. The rule
    it encodes still is: prefer the answer, and fall back to the reasoning only
    when there is no answer at all. Reading only `text` blocks would repeat the
    llama.cpp mistake in a different wire format.

    Thinking blocks are usually empty here, because `display` defaults to
    `omitted`, so the fallback is a guard rather than a working path. It costs
    nothing and it is the guard whose absence cost 124 notes on Gemma.
    """
    text = "".join(
        block.text for block in blocks if getattr(block, "type", None) == "text"
    )
    if text.strip():
        return text

    thinking = "".join(
        getattr(block, "thinking", "") or ""
        for block in blocks
        if getattr(block, "type", None) == "thinking"
    )
    return thinking if thinking.strip() else ""


def sonnet_client(**kwargs) -> AnthropicClient:
    return AnthropicClient(model_id=SONNET_5, **kwargs)
