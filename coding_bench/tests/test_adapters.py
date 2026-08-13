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
