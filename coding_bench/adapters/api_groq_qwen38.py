"""Qwen3.8 27B on Groq, `qwen/qwen3.8-27b`.

Groq serves this model over HTTP, so it is a plain API client and there is no
Modal wrapper, no GPU to size, and no vLLM version to pin. The id was read off
the live catalogue on 2026-09-07 rather than assumed from the 3.6 name, using a
throwaway Modal function holding the `groq-api` secret, because `GROQ_API_KEY`
deliberately exists nowhere else.

## Why this is its own file rather than a third entry in api_groq.py

The prediction cache key includes the hash of the adapter file, and CI re-runs
any model whose adapter changed. `api_groq.py` backs `qwen3.6-27b` and
`gpt-oss-120b`, which have eight banked cells between them, two of them 578 note
ICD-10 runs at the full catalogue. Adding a line to that file would change its
hash, miss every one of those cached notes, and re-infer the pair at full price
on the next push, for a change that has nothing to do with either of them. A
separate file keeps that bill at this model's own notes, which is what the
workflow's own comment says an adapter change should cost.

**What that choice costs, stated rather than left to be discovered.** This
file's hash does not cover `GroqClient`, which lives in `api_groq.py` and does
the actual work. A change to the shared client that changes answers, the retry
path or the truncation check, will not invalidate this model's cached notes.
Nothing detects that; it has to be remembered, and the honest fix is the
explicit per model `ADAPTER_VERSION` that would let all three share one file
again without sharing one cache key.

## Thinking is off by default here, and that is the trap

The two Qwen rows have opposite defaults on Groq, which is measured rather than
assumed: with no `reasoning_effort` the 3.6 endpoint thinks and the 3.8 endpoint
does not. 3.6 accepts only `none` and `default` and rejects `low`/`medium`/`high`
with a 400; 3.8 accepts all of them and needs one of the three to think at all.

Leaving the field unset therefore does not put this row where the 3.6 row is, it
puts it in instruct mode against a reasoning model, which reports a decoder
difference as a model difference and flatters the newer model on latency while
doing it. One set of runs was banked that way on 2026-09-07 and discarded.

So `reasoning_strength`, the package's recorded run parameter, is mapped onto
`reasoning_effort` here. It is already in the cache key, so changing it
correctly re-runs, and it is already in the manifest, so what a row was measured
at stays answerable.

**`high` is measured broken on the full ICD-10 catalogue and must not be used
there.** Over ten notes it averaged 16,084 completion tokens against a ceiling of
16,384, truncated nine of ten, and returned empty content on all ten: it spends
the entire budget thinking and never emits the answer. `medium` on the same ten
averaged 6,057 with a longest of 14,704, no truncation and no empty content,
which also puts it alongside the 3.6 row it is compared against, 4,944 tokens and
12.4 seconds per note against 6,057 and 13.4. That measurement is one cell and
ten notes, so it is a reason to default to `medium`, not a reason to believe
`high` is broken everywhere.

## The token ceiling, which is the risk on this row

Groq reports `max_completion_tokens` of 16,384 for this model. That is exactly
the budget the board's cells already use, so unlike every self hosted row here
there is no headroom to buy: the budget cannot be raised, only spent better.

This matters because the budget covers reasoning and answer together, and the
margin on the row next door is already thin. On the banked 578 note ICD-10 full
catalogue run, `qwen3.6-27b` averaged 4,972 completion tokens and its longest
note spent 16,351 of the 16,384 available, which is 33 tokens of headroom, at a
0.7% truncation rate. GPT-OSS on the same cell averaged 1,968 with a longest of
8,087. Same family, same size, same ceiling, and no way to raise it.

A self hosted attempt at this same budget did hit the cap often enough to be
thrown away, but that run was incomplete and is not evidence of a rate. Whether
Groq's serving of this model runs longer traces than 3.6 does is a question for
a measured run. If it truncates, the lever is `reasoning_effort` rather than a
bigger budget.

## Precision

Groq does not publish what precision it serves at, so this id carries no build
tag, exactly like `qwen3.6-27b` beside it. It is not the `Qwen/Qwen3.8-27B-FP8`
build the repository briefly self hosted and must not be tabulated as though it
were: that row was a named, verified fp8 weight load, and this one is a hosted
endpoint whose stack nobody outside Groq can check.

Environment:
    GROQ_API_KEY   required, resolved server side from the Modal secret
"""

from __future__ import annotations

from coding_bench.adapters.api_groq import GroqClient

# Verified against the live Groq catalogue on 2026-09-07: active, context window
# 131,042, max_completion_tokens 16,384, owned by Alibaba Cloud.
QWEN_3_8_27B = "qwen/qwen3.8-27b"

# The provider's ceiling, not a preference. Passing more is not an error, it is
# silently clamped, which would make a truncated run look like a chosen budget.
MAX_COMPLETION_TOKENS = 16384

MODELS = {
    "qwen3.8-27b": QWEN_3_8_27B,
}


# The package's reasoning_strength, as Groq names it. A pass through rather than
# a translation, kept explicit so an unknown strength fails loudly here instead
# of silently reaching the API and coming back as a 400 mid run.
REASONING_EFFORT = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def reasoning_effort_for(reasoning_strength: str) -> str:
    """Groq's effort level for a package reasoning strength."""
    try:
        return REASONING_EFFORT[reasoning_strength]
    except KeyError:
        raise ValueError(
            f"Unknown reasoning_strength {reasoning_strength!r} for {QWEN_3_8_27B}. "
            f"Known: {sorted(REASONING_EFFORT)}"
        ) from None


def qwen38_client(reasoning_strength: str = "medium", **kwargs) -> GroqClient:
    """Qwen3.8 27B behind the same client every other Groq row uses.

    The default is the package default, and it is the one that thinks. Passing
    nothing here is what produced the discarded instruct mode runs.
    """
    return GroqClient(
        model_id=QWEN_3_8_27B,
        reasoning_effort=reasoning_effort_for(reasoning_strength),
        **kwargs,
    )
