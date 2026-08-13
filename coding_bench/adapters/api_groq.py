"""Groq hosted models: Qwen 3.6 27B and GPT OSS 120B.

Groq serves these over HTTP, so there is no Modal wrapper here. Wrapping an API
call in a GPU container would cost money and add a failure mode without adding
anything.

Both models are reasoning models. They spend tokens thinking before they answer,
which is why max_tokens is generous and why truncation has to be checked rather
than assumed away: a length stop lands mid thought, and the JSON never arrives.

Unlike the two llama.cpp adapters, Groq puts the whole response in `content`
with no separate reasoning channel, so the channel bug those had never applied
here. That was checked per note rather than assumed, see
scripts/groq_silence_probe.py. What the probe did find is a different problem
that this file cannot fix: qwen quotes clinical text verbatim into its JSON
without escaping the quote marks in it, which leaves the answer unparseable.
The evidence is in that script's docstring.

Environment:
    GROQ_API_KEY   required
"""

from __future__ import annotations

import os
import time

from coding_bench.approaches.base import Completion, Truncated, answer_text

# Verified against the Groq catalogue on 2026-08-11.
QWEN_3_6_27B = "qwen/qwen3.6-27b"
GPT_OSS_120B = "openai/gpt-oss-120b"

MODELS = {
    "qwen3.6-27b": QWEN_3_6_27B,
    "gpt-oss-120b": GPT_OSS_120B,
}


class GroqClient:
    """One Groq model behind the LLMClient protocol."""

    def __init__(
        self,
        model_id: str = QWEN_3_6_27B,
        temperature: float = 0.0,
        api_key: str | None = None,
        # Groq enforces a tokens per minute ceiling, and a clinical note prompt
        # is around 7k tokens, so a pool of workers saturates it routinely.
        # Retrying generously is cheaper than losing notes: a note that gives up
        # is scored as an empty prediction and quietly drags the score down.
        max_retries: int = 6,
        reasoning_effort: str | None = None,
        timeout_s: float = 180.0,
    ):
        from groq import Groq

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")

        self.model_id = model_id
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        # Without a timeout a single stalled connection pins a worker forever,
        # and with a thread pool that quietly drains the whole run. Better to
        # fail one note and record it than to hang the evaluation.
        self._client = Groq(api_key=key, timeout=timeout_s, max_retries=0)

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        from groq import APIStatusError, APITimeoutError, RateLimitError

        kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APIStatusError, APITimeoutError) as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                # Rate limits are the common case on preview models, and Groq
                # reports how long to wait. Respect that rather than guessing.
                wait = 2**attempt
                retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    wait = max(wait, float(retry_after.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
                time.sleep(min(wait, 60))
                continue

            elapsed = time.perf_counter() - start
            choice = response.choices[0]
            stop_reason = choice.finish_reason or "unknown"

            if stop_reason == "length":
                raise Truncated(
                    self.model_id,
                    stop_reason,
                    produced_tokens=getattr(response.usage, "completion_tokens", None),
                )

            usage = response.usage
            return Completion(
                # Measured, not assumed: both of these models return everything
                # in `content` and no reasoning channel at all, so this is a
                # pass through today. It is here so that every adapter reads a
                # response the same way, and so that Groq changing a default
                # cannot cost a run the way it cost the llama.cpp models.
                text=answer_text(choice.message),
                stop_reason=stop_reason,
                latency_s=elapsed,
                usage={
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                },
            )

        raise RuntimeError(f"{self.model_id} failed after {self.max_retries} attempts: {last_error}")


def qwen_client(**kwargs) -> GroqClient:
    return GroqClient(model_id=QWEN_3_6_27B, **kwargs)


def gpt_oss_client(**kwargs) -> GroqClient:
    return GroqClient(model_id=GPT_OSS_120B, **kwargs)
