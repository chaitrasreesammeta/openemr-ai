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
# the rest of the board, and nothing fails when that happens: the chain runs, the
# records bank, the leaderboard tabulates, and a difference in serving lands in
# the table looking like a difference in models. These are the only place that
# error is detectable.


def test_both_qwen38_arms_are_registered_and_self_hosted():
    assert ADAPTER_FILES["qwen3.8-27b-fp8"] == "modal_qwen38_vllm.py"
    assert ADAPTER_FILES["qwen3.8-27b-bf16"] == "modal_qwen38_vllm.py"
    # Neither arm may claim an outside provider saw the note text. Both run on
    # our own GPU, which is the point of having them.
    assert {"qwen3.8-27b-fp8", "qwen3.8-27b-bf16"} <= LOCAL_MODELS
    assert EXTERNAL_PROVIDERS.get("qwen3.8-27b-fp8") is None
    assert EXTERNAL_PROVIDERS.get("qwen3.8-27b-bf16") is None


def test_the_arms_name_the_exact_released_checkpoints():
    """The manifest id has to be something a reader can fetch and rerun."""
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    assert qwen38.FP8_CHECKPOINT == "Qwen/Qwen3.8-27B-FP8"
    assert qwen38.BF16_CHECKPOINT == "Qwen/Qwen3.8-27B"
    # The FP8 repo carries its precision in its own name, so unlike the GGUF
    # builds neither arm needs a `:quantisation` tag bolted on to stay honest.
    assert qwen38.ARMS["fp8"][0] == qwen38.FP8_CHECKPOINT
    assert qwen38.ARMS["bf16"][0] == qwen38.BF16_CHECKPOINT


def test_the_two_arms_differ_only_in_the_checkpoint():
    """The bf16 arm is unqueued, not unmaintained.

    It is kept so that what FP8 costs is a chain step away rather than a
    rewrite, and that is only true while the two arms are otherwise identical.
    A serving flag that reaches one and not the other produces a number that
    looks exactly like a precision effect. The two likeliest are
    `--kv-cache-dtype`, which vLLM's own recipe for this model suggests setting
    to fp8 and which would then quantise the cache on one side only, and
    `--max-num-seqs`, which somebody will eventually raise on the FP8 arm
    because it has the spare VRAM, changing its batch composition and with it
    its greedy output.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    bf16 = qwen38.server_command(qwen38.BF16_CHECKPOINT)
    fp8 = qwen38.server_command(qwen38.FP8_CHECKPOINT)

    assert len(bf16) == len(fp8), "the arms build different length argument lists"
    differing = [(a, b) for a, b in zip(bf16, fp8) if a != b]
    assert differing == [
        (qwen38.BF16_CHECKPOINT, qwen38.FP8_CHECKPOINT),
        (qwen38.BF16_CHECKPOINT, qwen38.FP8_CHECKPOINT),
    ], f"the arms differ in more than the checkpoint: {differing}"


def test_the_kv_cache_is_not_quantised():
    """Named separately because it is the confound most likely to be introduced.

    The failure is not that an fp8 KV cache is wrong. It is a perfectly good
    third arm. It is that switching it on for one arm attributes the sum of two
    effects to one of them.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    args = qwen38.SERVER_ARGS
    assert args[args.index("--kv-cache-dtype") + 1] == "auto"
    assert "--model" not in args, "the checkpoint must come from the arm, not the shared args"


def test_note_text_cannot_reach_the_server_log():
    """Tier 2 text stays inside Modal, and a log is outside enough to matter."""
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    assert "--disable-log-requests" in qwen38.SERVER_ARGS


def test_thinking_rides_on_the_recorded_run_parameter():
    """So the mode a number was produced under is in the manifest, not in a default.

    reasoning_strength is already carried in every run manifest and in the
    prediction cache key. Introducing a separate thinking flag would put the
    mode outside both, and two runs that decoded differently would then share a
    cache entry.
    """
    from coding_bench.adapters.modal_qwen38_vllm import thinking_enabled

    assert thinking_enabled("none") is False
    assert thinking_enabled("off") is False
    assert thinking_enabled("") is False
    assert thinking_enabled(None) is False
    assert thinking_enabled("high") is True
    # The package default, and therefore the mode the board's row is measured
    # in. See the next test for why that is the one that matters.
    assert thinking_enabled("medium") is True


def test_the_chain_thinks_because_the_model_it_is_compared_against_thinks():
    """The equivalence that is easiest to lose and hardest to notice afterwards.

    Qwen3.6 27B is the nearest row on the board, same family and same size, and
    Groq serves it as a reasoning model. An instruct mode Qwen3.8 measured
    against it would report a decoder difference as a model difference, and
    would flatter the newer model on latency at the same time. Nothing about a
    run record would look wrong.
    """
    from coding_bench.adapters.modal_qwen38_vllm import thinking_enabled
    from coding_bench import run_chain_qwen38 as chain

    assert thinking_enabled(chain.DEFAULTS["reasoning_strength"])
    for step in chain.CHAIN:
        assert thinking_enabled(step.get("reasoning_strength", chain.DEFAULTS["reasoning_strength"]))


def test_the_chain_runs_the_fp8_build_on_every_condition_the_board_compares_on():
    """Four conditions, because a partial row cannot be ranked against a full one.

    And FP8 only. The bf16 arm is registered for the ablation and is not the
    build that gets a row: every self hosted row on the board is a deployment
    build, `full` is the deployment condition, and no hosted provider here
    publishes the precision a bf16 row would supposedly be matching.
    """
    from coding_bench import run_chain_qwen38 as chain

    assert {step["model"] for step in chain.CHAIN} == {"qwen3.8-27b-fp8"}
    assert {(step["task"], step["candidate_space"]) for step in chain.CHAIN} == {
        ("cpt", "gold"), ("cpt", "full"), ("icd10", "gold"), ("icd10", "full"),
    }
    # In the cache key, so a different value is not just incomparable with the
    # board's other runs, it also misses every cached note and pays again.
    assert chain.DEFAULTS["max_tokens"] == 16384


def test_the_watch_catches_the_builtin_timeout_and_not_modals():
    """Modal exports a TimeoutError that is not a subclass of the builtin one.

    `FunctionCall.get(timeout=...)` raises the builtin when the wait expires.
    Catching `modal.exception.TimeoutError` instead would read as the more
    careful choice, would never fire, and would turn every run that outlasts the
    watch into a failed CI job. The chain is a spawn and survives the watch, so
    that failure would be entirely cosmetic and entirely misleading.
    """
    from modal.exception import TimeoutError as ModalTimeoutError
    from coding_bench import run_chain_qwen38 as chain

    assert not issubclass(ModalTimeoutError, TimeoutError), (
        "modal now aliases the builtin, so the catch below can be simplified"
    )
    source = Path(chain.__file__).read_text(encoding="utf8")
    assert "except (TimeoutError, OutputExpiredError)" in source
    assert "from modal.exception import TimeoutError" not in source


def build_qwen38_client(arm: str = "fp8", reasoning_strength: str = "medium", **response):
    """A client wired to a stub, without constructing a Modal handle."""
    from coding_bench.adapters.modal_qwen38_vllm import ARMS, Qwen38Client, thinking_enabled

    client = object.__new__(Qwen38Client)
    client.arm = arm
    client.model_id = ARMS[arm][0]
    client.temperature = 0.0
    client.reasoning_strength = reasoning_strength
    client.enable_thinking = thinking_enabled(reasoning_strength)
    stub = StubRemote(**response)
    client._remote = stub
    return client, stub


def test_a_cut_off_generation_raises_rather_than_scoring_as_no_codes():
    """The failure this configuration actually has.

    Greedy decoding in thinking mode is what Qwen warns about, and a looping
    trace hits the cap with no JSON ever produced. Returning that as an empty
    completion would score it as the model correctly finding nothing, and the
    truncation rate that should quarantine the run would never be visible.
    """
    for arm in ("fp8", "bf16"):
        client, _ = build_qwen38_client(
            arm,
            message={"content": '{"codes": [{"code": "I10", "quo'},
            stop_reason="length",
            latency_s=120.0,
            usage={"prompt_tokens": 11000, "completion_tokens": 16384},
        )
        with pytest.raises(Truncated) as raised:
            client.complete("system", "user", max_tokens=16384)
        assert raised.value.produced_tokens == 16384
        assert raised.value.model_id == client.model_id


def test_the_answer_is_read_out_of_the_reasoning_channel():
    """`--reasoning-parser qwen3` splits the channels, so the same rule applies.

    This is not hypothetical on a thinking model. It cost 124 of 312 notes on
    Gemma 4 and 120 of 150 on Muse, each one scored as an empty prediction.
    """
    client, _ = build_qwen38_client(
        message={
            "role": "assistant",
            "content": "",
            "reasoning_content": '{"codes": [{"code": "I10", "quote": "hypertension"}]}',
        },
        stop_reason="stop",
        latency_s=40.0,
        usage={"prompt_tokens": 11000, "completion_tokens": 900},
    )
    assert "I10" in client.complete("system", "user", max_tokens=16384).text


def test_the_run_parameters_reach_the_server():
    client, stub = build_qwen38_client(
        message={"content": "{}"}, stop_reason="stop", latency_s=1.0, usage={},
    )
    client.complete("a system prompt", "a user prompt", max_tokens=16384)
    call = stub.calls[0]
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0.0, "greedy, so a rerun reproduces the run"
    assert call["enable_thinking"] is True, "thinking, matching the Qwen3.6 row"


def test_the_arms_ask_for_identical_generations():
    """Kept honest for the ablation that is not queued yet."""
    calls = {}
    for arm in ("fp8", "bf16"):
        client, stub = build_qwen38_client(
            arm, message={"content": "{}"}, stop_reason="stop", latency_s=1.0, usage={},
        )
        client.complete("a system prompt", "a user prompt", max_tokens=16384)
        calls[arm] = stub.calls[0]
    assert calls["fp8"] == calls["bf16"], "the arms asked for different generations"


def test_candidate_caching_approaches_are_pinned_to_one_thread():
    """These cache encoded candidates on the instance; sharing corrupts it.

    The failure mode is a wrong answer rather than an exception, so the guard is
    asserted rather than left to whoever adds the next approach.
    """
    from coding_bench.eval_remote import SINGLE_THREADED_APPROACHES

    assert {"retrieval", "retr_llm", "embed_match", "entity_match"} <= SINGLE_THREADED_APPROACHES
