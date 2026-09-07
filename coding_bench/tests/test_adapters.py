"""The adapter registry, and the one contract every adapter has to keep.

No network, no GPU, no restricted data, like the rest of the suite. The remote
server is replaced by a stub, so what is under test is the client side logic
that turns a provider response into either a Completion or a typed Truncated.

These tests exist because of a specific failure. Two Gemma 4 runs were banked
from an adapter that was never pushed, and the only things tying those records
to any code are the model id string and the adapter file the manifest hashed.
Both are easy to break by hand and neither was checked by anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_bench.approaches.base import Completion, Truncated, answer_text

modal = pytest.importorskip("modal", reason="adapters are an optional install")

from coding_bench.adapters import modal_gemma4_gguf as gemma4  # noqa: E402
from coding_bench.eval_remote import ADAPTER_FILES, EXTERNAL_PROVIDERS, LOCAL_MODELS  # noqa: E402
from coding_bench.bench.runner import RUNS_DIR  # noqa: E402

ADAPTER_DIR = gemma4.__file__.rsplit("/", 1)[0]


# --------------------------------------------------------------------------
# The registry


def test_every_registered_model_points_at_an_adapter_that_exists():
    """A wrong filename here is silent and expensive.

    The manifest hashes whatever file this names, and that hash is both the
    audit trail and part of the prediction cache key. Point it at the wrong
    adapter and the run is misattributed, while an edited adapter goes on
    serving stale cached answers.
    """
    from pathlib import Path

    for model, filename in ADAPTER_FILES.items():
        assert (Path(ADAPTER_DIR) / filename).is_file(), f"{model} names a missing {filename}"


def test_local_models_are_exactly_those_with_no_external_provider():
    assert LOCAL_MODELS == set(ADAPTER_FILES) - set(EXTERNAL_PROVIDERS)
    assert "gemma4-26b-a4b-gguf" in LOCAL_MODELS
    # Nothing that runs on our own GPU may claim an outside provider saw the
    # note text, and nothing reached over an API may claim it did not.
    for model in LOCAL_MODELS:
        assert EXTERNAL_PROVIDERS.get(model) is None


def test_gemma4_is_named_as_a_quantised_build():
    """The label carries the build, so a QAT 4 bit is never read as the release."""
    assert gemma4.MODEL_ID == "google/gemma-4-26B-A4B-it:qat-UD-Q4_K_XL"
    assert gemma4.Gemma4GGUFClient.model_id == gemma4.MODEL_ID
    assert ":" in gemma4.MODEL_ID, "the quantisation tag is what stops the confusion"


def test_gemma4_model_id_still_matches_the_banked_runs():
    """Renaming the label would orphan the runs already on disk.

    The leaderboard groups on `model_id`, so a change here does not fail loudly.
    It quietly produces two models where there is one, and the head to head then
    pairs a model against itself.
    """
    banked = [
        json.loads(path.read_text(encoding="utf8"))["manifest"]["model_id"]
        for path in sorted(RUNS_DIR.glob("*gemma*.json"))
    ]
    assert banked, "the Gemma 4 run records are missing from results/runs"
    for model_id in banked:
        assert model_id == gemma4.MODEL_ID


# --------------------------------------------------------------------------
# The adapter contract


class StubRemote:
    """Stands in for the deployed llama.cpp server."""

    def __init__(self, **response):
        self.complete = self
        self.response = response
        self.calls: list[dict] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def build_client(**response) -> tuple[gemma4.Gemma4GGUFClient, StubRemote]:
    """A client wired to a stub, without constructing a Modal handle."""
    client = object.__new__(gemma4.Gemma4GGUFClient)
    client.temperature = 0.0
    client.reasoning_strength = "medium"
    stub = StubRemote(**response)
    client._remote = stub
    return client, stub


def test_a_finished_generation_comes_back_as_a_completion():
    client, _ = build_client(
        message={"content": '{"codes": [{"code": "99213", "quote": "office visit"}]}'},
        stop_reason="stop",
        latency_s=12.5,
        usage={"prompt_tokens": 4000, "completion_tokens": 40},
    )
    result = client.complete("system", "user", max_tokens=16384)
    assert isinstance(result, Completion)
    assert result.stop_reason == "stop"
    assert result.usage["completion_tokens"] == 40


def test_a_cut_off_generation_raises_rather_than_returning_a_partial_answer():
    """This is the path both banked runs took on about one note in ten.

    Truncation and "no codes apply" are opposite findings. If a cut off
    generation returned an empty Completion instead, it would score as the model
    correctly finding nothing, and the 10% failure rate that quarantined those
    runs would never have been visible at all.
    """
    client, _ = build_client(
        message={"content": '{"codes": [{"code": "99213", "quo'},
        stop_reason="length",
        latency_s=300.0,
        usage={"prompt_tokens": 4000, "completion_tokens": 32768},
    )
    with pytest.raises(Truncated) as raised:
        client.complete("system", "user", max_tokens=32768)
    assert raised.value.produced_tokens == 32768
    assert raised.value.model_id == gemma4.MODEL_ID


def test_an_empty_answer_is_passed_through_as_an_answer():
    """An empty response is a finding about the model, not an error.

    124 of 312 notes came back like this in both banked runs, with no error and
    no truncation, so the runner has to receive them as ordinary completions.
    Turning them into errors here would hide the thing worth investigating.
    """
    client, _ = build_client(
        message={"content": "", "reasoning_content": ""}, stop_reason="stop", latency_s=30.0,
        usage={"prompt_tokens": 4000, "completion_tokens": 1810},
    )
    result = client.complete("system", "user", max_tokens=16384)
    assert result.text == ""
    assert result.usage["completion_tokens"] == 1810


def test_the_run_parameters_reach_the_server():
    client, stub = build_client(
        message={"content": "{}"}, stop_reason="stop", latency_s=1.0, usage={},
    )
    client.complete("a system prompt", "a user prompt", max_tokens=32768)
    call = stub.calls[0]
    assert call["max_tokens"] == 32768
    assert call["temperature"] == 0.0, "greedy, so a rerun reproduces the run"
    assert call["reasoning_strength"] == "medium"


# --------------------------------------------------------------------------
# Which channel the answer is read from


def test_content_is_returned_untouched_when_it_has_anything():
    """The fix must be a no op for every response that already worked.

    This is the property the cache carry forward rests on. If content wins
    byte for byte, a note that produced codes cannot have changed, and paying
    to generate it again buys nothing.
    """
    message = {"content": '  {"codes": []}  ', "reasoning_content": "thinking out loud"}
    assert answer_text(message) == '  {"codes": []}  '


def test_the_reasoning_channel_is_read_when_content_is_empty():
    """Otherwise a model that never stopped thinking scores as finding nothing."""
    message = {"content": "", "reasoning_content": 'I will answer {"codes": []}'}
    assert answer_text(message) == 'I will answer {"codes": []}'


def test_whitespace_only_content_does_not_count_as_an_answer():
    message = {"content": "\n \n", "reasoning_content": "the actual output"}
    assert answer_text(message) == "the actual output"


def test_channels_are_read_off_a_provider_object_as_well_as_a_dict():
    """The Groq SDK hands back a model object, not a mapping.

    Reading it with `.get` would raise, and reading it with a bare getattr would
    silently return an empty answer for every note if the field were ever
    renamed. Both adapters go through the same function for that reason.
    """

    class SDKMessage:
        role = "assistant"
        content = '{"codes": []}'

    assert answer_text(SDKMessage()) == '{"codes": []}'

    class ThinkingSDKMessage:
        role = "assistant"
        content = None
        reasoning = "the answer was in here"

    assert answer_text(ThinkingSDKMessage()) == "the answer was in here"


def test_a_genuinely_empty_response_stays_empty():
    """No channel had anything, so there is nothing to recover and no pretending."""
    assert answer_text({"content": "", "reasoning_content": ""}) == ""
    assert answer_text({"role": "assistant"}) == ""


def test_the_client_reads_the_answer_out_of_the_reasoning_channel():
    """The end to end shape of the bug, through the adapter that had it."""
    client, _ = build_client(
        message={
            "role": "assistant",
            "content": "",
            "reasoning_content": '{"codes": [{"code": "99213", "quote": "office visit"}]}',
        },
        stop_reason="stop",
        latency_s=30.0,
        usage={"prompt_tokens": 4000, "completion_tokens": 1810},
    )
    result = client.complete("system", "user", max_tokens=16384)
    assert "99213" in result.text


# --------------------------------------------------------------------------
# Anthropic: a different wire format for the same two questions


def test_anthropic_truncation_is_max_tokens_not_length():
    """The Groq check copied verbatim would never fire here.

    OpenAI compatible endpoints report a cut off generation as
    `finish_reason: "length"`; the Anthropic API reports
    `stop_reason: "max_tokens"`. An adapter that checked for "length" would
    return a truncated answer as an ordinary completion, which scores as the
    model finding nothing. This pins the constant so the two cannot drift.
    """
    from coding_bench.adapters import api_anthropic

    source = Path(api_anthropic.__file__).read_text(encoding="utf8")
    assert 'stop_reason == "max_tokens"' in source
    assert 'stop_reason == "refusal"' in source
    assert '"length"' not in source.split('"""', 2)[2], "no OpenAI stop reason in the code"


def test_anthropic_reads_text_blocks_and_falls_back_to_thinking():
    """The same rule as answer_text, expressed for a list of typed blocks."""
    from coding_bench.adapters.api_anthropic import answer_from_blocks

    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    answered = [
        Block(type="thinking", thinking="deliberating"),
        Block(type="text", text='{"codes": []}'),
    ]
    assert answer_from_blocks(answered) == '{"codes": []}'

    # Text blocks are concatenated, not just the first one taken.
    assert answer_from_blocks([Block(type="text", text="a"), Block(type="text", text="b")]) == "ab"

    # Nothing in the answer channel: read the reasoning rather than discard it.
    thinking_only = [Block(type="thinking", thinking='{"codes": [{"code": "I10"}]}')]
    assert "I10" in answer_from_blocks(thinking_only)

    assert answer_from_blocks([Block(type="text", text="  ")]) == ""
    assert answer_from_blocks([]) == ""


def test_anthropic_is_recorded_as_a_second_external_provider():
    """A run has to say which provider saw the restricted note text."""
    assert EXTERNAL_PROVIDERS["sonnet-5"] == "anthropic"
    assert ADAPTER_FILES["sonnet-5"] == "api_anthropic.py"
    assert "sonnet-5" not in LOCAL_MODELS


# --------------------------------------------------------------------------
# Qwen3.8 27B: a row is only worth having if it is comparable
#
# Everything below guards the same thing, which is not correctness. The adapter
# can be entirely correct and still produce a number that cannot be read next to
# the rest of the board, and nothing fails when that happens: the run banks, the
# leaderboard tabulates, and a difference in serving lands in the table looking
# like a difference in models. These are the only place that error is
# detectable.
#
# This model was self hosted on vLLM at fp8 and now runs on Groq. The tests that
# guarded the serving build, the KV cache dtype, the slot count and the thinking
# switch went with it: none of those are ours to set any more, which is itself
# the thing the first test below records.


def test_qwen38_is_registered_as_a_hosted_model():
    assert ADAPTER_FILES["qwen3.8-27b"] == "api_groq_qwen38.py"
    # The inverse of what this asserted while the model was self hosted. Note
    # text now leaves Modal for Groq, and a row that does not say so is a
    # provenance claim the run manifest would repeat.
    assert EXTERNAL_PROVIDERS["qwen3.8-27b"] == "groq"
    assert "qwen3.8-27b" not in LOCAL_MODELS


def test_the_fp8_build_is_no_longer_reachable():
    """The self hosted arm is gone rather than dormant.

    Left registered and unqueued it would be a second Qwen3.8 row that anyone
    could launch, measured on a different stack at a precision this one cannot
    name, and the two would tabulate as though they were one model.
    """
    assert "qwen3.8-27b-fp8" not in ADAPTER_FILES
    assert "qwen3.8-27b-fp8" not in EXTERNAL_PROVIDERS
    assert not (Path(ADAPTER_DIR) / "modal_qwen38_vllm.py").exists()


def test_the_id_is_the_catalogue_id_and_carries_no_build_tag():
    """Groq does not publish its serving precision, so the id claims none.

    The fp8 row could name its build because the weights were loaded here and
    the load was verified. This one cannot, and inventing a tag would be a claim
    about somebody else's stack.
    """
    from coding_bench.adapters import api_groq_qwen38 as qwen38

    assert qwen38.QWEN_3_8_27B == "qwen/qwen3.8-27b"
    assert ":" not in qwen38.QWEN_3_8_27B.split("/", 1)[1]


def test_qwen38_stays_out_of_the_shared_groq_adapter():
    """The separate file is a cost decision, and this is what protects it.

    `api_groq.py` is in the cache key of `qwen3.6-27b` and `gpt-oss-120b`, and
    CI re-runs any model whose adapter file changed. Moving this model into it
    would silently re-infer eight banked cells on the next push, two of them 578
    note ICD-10 runs. Nothing else would notice: the suite would stay green and
    the bill would arrive later.
    """
    from coding_bench.adapters import api_groq

    assert "qwen3.8-27b" not in api_groq.MODELS
    assert ADAPTER_FILES["qwen3.6-27b"] == "api_groq.py"
    assert ADAPTER_FILES["gpt-oss-120b"] == "api_groq.py"
    assert ADAPTER_FILES["qwen3.8-27b"] != ADAPTER_FILES["qwen3.6-27b"]


def test_the_token_ceiling_is_the_providers_and_not_a_preference():
    """16,384 is what Groq allows, which is exactly what the board asks for.

    Every self hosted row here could buy headroom by raising the budget, and the
    Gemma 4 row did. This one cannot, so the number is recorded next to the
    model rather than left implicit in a chain file, and a future run that wants
    more has to confront that it is asking for something the provider will clamp.
    """
    from coding_bench.adapters import api_groq_qwen38 as qwen38

    assert qwen38.MAX_COMPLETION_TOKENS == 16384


def test_qwen38_reuses_the_client_the_other_groq_rows_use(monkeypatch):
    """The truncation contract is tested once, on GroqClient, and inherited.

    A second client class here would need its own copy of those tests and would
    drift from them, which is how a length stop stops raising and starts
    scoring as an answer with no codes.
    """
    import sys
    import types

    from coding_bench.adapters.api_groq import GroqClient
    from coding_bench.adapters.api_groq_qwen38 import qwen38_client

    # `groq` is an optional extra and public CI installs [dev,modal,anthropic]
    # without it, while the client imports it inside __init__. A stub module
    # keeps this test about the wiring rather than about the dependency.
    stub = types.ModuleType("groq")
    stub.Groq = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "groq", stub)

    client = qwen38_client(api_key="not-a-real-key")
    assert isinstance(client, GroqClient)
    assert client.model_id == "qwen/qwen3.8-27b"
    assert client.temperature == 0.0, "greedy, so a rerun reproduces the run"


def test_qwen38_thinks_by_default(monkeypatch):
    """The regression that cost a full set of runs.

    Groq's 3.8 endpoint does not think unless asked, while the 3.6 endpoint it
    is compared against does. A client built with no reasoning_effort measures
    an instruct model against a reasoning one, banks four plausible looking
    records, and reports the decoder difference as a model difference. Nothing
    else detects that: the runs succeed, the error rate is clean, and the board
    tabulates them.
    """
    import sys
    import types

    from coding_bench.adapters.api_groq_qwen38 import qwen38_client

    stub = types.ModuleType("groq")
    stub.Groq = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "groq", stub)

    assert qwen38_client(api_key="k").reasoning_effort == "medium"
    assert qwen38_client(api_key="k", reasoning_strength="low").reasoning_effort == "low"


def test_an_unknown_reasoning_strength_fails_here_rather_than_mid_run():
    """Groq answers a bad effort with a 400, one note at a time, having charged
    for every note before it."""
    from coding_bench.adapters.api_groq_qwen38 import reasoning_effort_for

    assert reasoning_effort_for("medium") == "medium"
    with pytest.raises(ValueError, match="Unknown reasoning_strength"):
        reasoning_effort_for("enthusiastic")


def test_the_run_parameter_reaches_the_qwen38_client(monkeypatch):
    """build_predictor must pass it through, which is where it was dropped."""
    import sys
    import types

    from coding_bench.eval_remote import build_predictor

    stub = types.ModuleType("groq")
    stub.Groq = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "groq", stub)
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")

    predictor = build_predictor("llm", "qwen3.8-27b", "ICD-10-CM", 16384, "medium")
    assert predictor.client.reasoning_effort == "medium"
    low = build_predictor("llm", "qwen3.8-27b", "ICD-10-CM", 16384, "low")
    assert low.client.reasoning_effort == "low"


def test_candidate_caching_approaches_are_pinned_to_one_thread():
    """These cache encoded candidates on the instance; sharing corrupts it.

    The failure mode is a wrong answer rather than an exception, so the guard is
    asserted rather than left to whoever adds the next approach.
    """
    from coding_bench.eval_remote import SINGLE_THREADED_APPROACHES

    assert {"retrieval", "retr_llm", "embed_match", "entity_match"} <= SINGLE_THREADED_APPROACHES
