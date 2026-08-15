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


def test_qwen38_is_registered_and_self_hosted():
    assert ADAPTER_FILES["qwen3.8-27b-fp8"] == "modal_qwen38_vllm.py"
    # It may not claim an outside provider saw the note text. It runs on our own
    # GPU, which is the point of having it.
    assert "qwen3.8-27b-fp8" in LOCAL_MODELS
    assert EXTERNAL_PROVIDERS.get("qwen3.8-27b-fp8") is None


def test_only_the_fp8_build_is_reachable():
    """Qwen publishes bf16 weights too, and this package does not serve them.

    Removed rather than registered and left unqueued, so there is no path by
    which a bf16 number reaches the board without somebody rewriting the adapter
    and re-reading the argument for why FP8 is the build that gets a row.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    assert qwen38.CHECKPOINT == "Qwen/Qwen3.8-27B-FP8"
    assert not hasattr(qwen38, "BF16_CHECKPOINT")
    assert not hasattr(qwen38, "Qwen38BF16")
    assert qwen38.Qwen38Client.model_id == qwen38.CHECKPOINT
    # No bf16 model id survives in the registry either.
    assert not [model for model in ADAPTER_FILES if model.endswith("bf16")]

    # The served checkpoint appears in the argv exactly where the id says it is.
    command = qwen38.server_command()
    assert command[:3] == ["vllm", "serve", qwen38.CHECKPOINT]
    assert command[command.index("--served-model-name") + 1] == qwen38.CHECKPOINT


def test_the_kv_cache_is_not_quantised():
    """The id on this row says fp8 weights, and only weights.

    vLLM's own recipe for this model passes `--kv-cache-dtype fp8`, so this is
    the flag most likely to be switched on by somebody copying it. Doing that
    would make the row a build Qwen never published, quantised in a second place
    that nothing here has measured, still labelled with the released id.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    args = qwen38.SERVER_ARGS
    assert args[args.index("--kv-cache-dtype") + 1] == "auto"
    assert "--model" not in args, "the checkpoint belongs to server_command, not the shared args"


def test_the_vllm_and_transformers_pins_are_mutually_satisfiable():
    """They were not, and it cost a launch.

    The model's vLLM recipe states a floor of 0.17.0 and, separately, that it
    needs transformers >= 5.8.0 because that is what wrote its config.json.
    Pinning the floor as a version asks for both at once, and vLLM 0.17.0
    requires `transformers<5`, so pip returned ResolutionImpossible and the
    image never built.

    0.24.0 is the first vLLM to require `transformers>=5.5.3` outright. Below it
    the constraint is either that hard `<5` cap or, from 0.20 to 0.23, a list of
    exclusions across the 5.x line. Nothing here may reach PyPI, so what is
    asserted is the floor, and the reason it exists is written down.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    def parts(version: str) -> tuple[int, ...]:
        return tuple(int(piece) for piece in version.split("."))

    assert parts(qwen38.VLLM_VERSION) >= (0, 24, 0), (
        "vLLM below 0.24.0 does not permit the transformers 5.x this model's processor needs"
    )
    assert parts(qwen38.TRANSFORMERS_VERSION) >= (5, 8, 0)

    # And both actually reach the image, or pinning them is decoration.
    source = Path(qwen38.__file__).read_text(encoding="utf8")
    assert 'f"vllm=={VLLM_VERSION}"' in source
    assert 'f"transformers>={TRANSFORMERS_VERSION}"' in source


def test_note_text_cannot_reach_the_server_log():
    """Tier 2 text stays inside Modal, and a log is outside enough to matter.

    The flag is asserted by its current name. `--disable-log-requests` is what
    older vLLM called this, it is gone in 0.27.1, and passing it made `vllm
    serve` exit at argument parsing and crash-loop a GPU container for 22
    minutes. Its replacement is `--enable-log-requests`, defaulting to false,
    and the negation is passed rather than the default relied on, because
    whether restricted notes are logged should be stated by the command.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    assert "--no-enable-log-requests" in qwen38.SERVER_ARGS
    assert "--disable-log-requests" not in qwen38.SERVER_ARGS, "removed in vLLM 0.27.1"
    # And never the bare form, which would turn logging on.
    assert "--enable-log-requests" not in qwen38.SERVER_ARGS


def test_the_weights_are_fetched_before_the_server_launches():
    """So an interrupted start keeps what it already pulled.

    There was briefly a `_check_flags` in front of this that asked `vllm serve
    --help` whether every flag was accepted. It could not parse the help output,
    concluded that vLLM rejects `--max-model-len`, `--host` and `--port`, and
    crash-looped a correct configuration for an hour. Detection was never the
    gap: a bad flag makes the server exit at argument parsing and `_await_ready`
    raises within seconds with the reason. The bound on Modal's restarts belongs
    in the caller, and it is the smoke step's `timeout-minutes`.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    source = Path(qwen38.__file__).read_text(encoding="utf8")
    body = source.split("def start_server(self):")[1].split("    def ")[0]
    assert body.index("_fetch_weights") < body.index("Popen")
    assert "_check_flags" not in body, "the flag preflight blocked a working config"


def test_the_weights_are_committed_to_the_volume():
    """A Modal volume keeps nothing until something commits it.

    Letting vLLM download implicitly looks equivalent and is not: a start that
    is interrupted takes the whole 27 GB with it and the next one pays again.
    Both llama.cpp adapters in this package already download and commit, and
    this one was the outlier.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    source = Path(qwen38.__file__).read_text(encoding="utf8")
    assert "snapshot_download(" in source
    assert "model_cache.commit()" in source


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


def test_the_chain_asks_for_exactly_as_many_notes_as_the_server_has_slots():
    """A mismatch here is silent, which is what makes it worth a test.

    Ask for more than the server has slots and the surplus queues at the door
    instead of widening the batch, so latency per note climbs for no throughput.
    Ask for fewer and slots sit idle. Either way the run completes and every
    accuracy column is correct, and only the latency column is quietly wrong.
    """
    from coding_bench.adapters.modal_qwen38_vllm import MAX_NUM_SEQS
    from coding_bench import run_chain_qwen38 as chain

    assert chain.CONCURRENCY == MAX_NUM_SEQS
    assert chain.DEFAULTS["concurrency"] == MAX_NUM_SEQS


def test_the_slot_count_leaves_room_for_the_weights_and_then_some():
    """The batch width is the one setting not held to what the other rows used.

    KV here is 64 KB a token: only 16 of the 64 layers hold a cache, at 4 KV
    heads by 256 dims, because the other 48 are Gated DeltaNet with constant
    state. Slots times context times that is the cache, and it has to sit beside
    27 GB of fp8 weights inside the 0.90 of a 96 GB card vLLM will use.

    The headroom left over is not slack. Prefill activations at a 19,130 token
    prompt, the vision tower and cuda graph capture all come out of it, and none
    of them appear in this arithmetic. The assertion is that the two terms it
    can compute leave a real margin, so raising the slot count has to be a
    decision rather than an edit to a comment.
    """
    from coding_bench.adapters import modal_qwen38_vllm as qwen38

    kv_bytes_per_token = 2 * 4 * 256 * 2 * 16
    kv_gb = qwen38.MAX_NUM_SEQS * qwen38.MAX_MODEL_LEN * kv_bytes_per_token / 1e9
    weights_gb = 27
    usable_gb = 96 * qwen38.GPU_MEMORY_UTILISATION

    assert kv_gb + weights_gb < usable_gb * 0.85, (
        f"{qwen38.MAX_NUM_SEQS} slots is {kv_gb:.0f}GB of KV, which with the weights "
        f"is {kv_gb + weights_gb:.0f}GB of {usable_gb:.0f}GB usable and leaves too "
        f"little for prefill activations and graph capture"
    )


def test_the_chain_runs_the_fp8_build_on_every_condition_the_board_compares_on():
    """Four conditions, because a partial row cannot be ranked against a full one.

    And the FP8 build, which is the only one this package serves: every self
    hosted row on the board is a deployment build, `full` is the deployment
    condition, and no hosted provider here publishes the precision a bf16 row
    would supposedly be matching.
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


def build_qwen38_client(reasoning_strength: str = "medium", **response):
    """A client wired to a stub, without constructing a Modal handle."""
    from coding_bench.adapters.modal_qwen38_vllm import Qwen38Client, thinking_enabled

    client = object.__new__(Qwen38Client)
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
    client, _ = build_qwen38_client(
        message={"content": '{"codes": [{"code": "I10", "quo'},
        stop_reason="length",
        latency_s=120.0,
        usage={"prompt_tokens": 11000, "completion_tokens": 16384},
    )
    with pytest.raises(Truncated) as raised:
        client.complete("system", "user", max_tokens=16384)
    assert raised.value.produced_tokens == 16384
    assert raised.value.model_id == "Qwen/Qwen3.8-27B-FP8"


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


def test_the_qwen38_run_parameters_reach_the_server():
    # Named for its model rather than reusing the gemma4 test's name. A second
    # `def` of the same name silently replaces the first, so the shadowed test
    # stops running and the suite still goes green with one fewer check.
    client, stub = build_qwen38_client(
        message={"content": "{}"}, stop_reason="stop", latency_s=1.0, usage={},
    )
    client.complete("a system prompt", "a user prompt", max_tokens=16384)
    call = stub.calls[0]
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0.0, "greedy, so a rerun reproduces the run"
    assert call["enable_thinking"] is True, "thinking, matching the Qwen3.6 row"


def test_candidate_caching_approaches_are_pinned_to_one_thread():
    """These cache encoded candidates on the instance; sharing corrupts it.

    The failure mode is a wrong answer rather than an exception, so the guard is
    asserted rather than left to whoever adds the next approach.
    """
    from coding_bench.eval_remote import SINGLE_THREADED_APPROACHES

    assert {"retrieval", "retr_llm", "embed_match", "entity_match"} <= SINGLE_THREADED_APPROACHES
