"""Qwen3.8 27B, self hosted on a Modal GPU with vLLM.

The build on the board is **FP8**, `Qwen/Qwen3.8-27B-FP8`. That is a choice and
it is worth writing down, because the bf16 weights are also released and the
instinct is to benchmark those and call it the honest number.

## Why FP8 is the build that gets the row

The board is a deployment board. Its `full` candidate condition exists because
gold candidates pin precision at 1.000 by construction and predict nothing, and
every self hosted row already on it is a deployment build rather than a
reference build: Muse Glimmer at roughly 4 bit, Gemma 4 at QAT 4 bit. A bf16 row
would be the only self hosted number on the board measured at a precision nobody
would actually serve for this task at this size.

It is also the only choice that stays honest about the hosted rows. Groq and
Anthropic do not publish what precision they serve at, and for throughput at
their prices it is very unlikely to be bf16. Running bf16 here to "match" them
would be a claim about their stack that nobody outside it can check. FP8 is at
least a precision this repository can name.

And FP8 is a Qwen release rather than somebody's quantisation of one. It has its
own repository, its own card, and Qwen puts it at "nearly identical" to the
original using fine grained fp8 at block size 128. The published evidence for
the format agrees: W8A8-FP8 measures as lossless across model scales in the
largest study of the question. That evidence is about general benchmarks, not
about rare label recall, so it is a reason to expect FP8 to be fine here and not
a reason to skip measuring it.

The id carries the build, the way the GGUF ids do, so no row on the board can be
read as the bf16 release.

## The bf16 arm is registered and is not queued

`Qwen38BF16` exists, is registered in `eval_remote.ADAPTER_FILES`, and nothing
runs it. It is here for the same reason `modal_muse.py` is: it is the comparison
that would settle what FP8 costs, and it is not the comparison the board needs
in order to be useful. Keeping it costs nothing and means the experiment is a
chain step away rather than a rewrite.

If it is ever run, it has to be comparable, so the two arms are built to differ
in exactly one thing. `SERVER_ARGS` is shared, `server_command` varies only the
checkpoint, and `tests/test_adapters.py::test_the_two_arms_differ_only_in_the_
checkpoint` asserts it. Two flags in particular are pinned rather than left to
judgement:

  * **`--kv-cache-dtype`.** vLLM's own recipe for this model passes `fp8`, which
    is the obvious thing to switch on for the FP8 arm and leave off the other.
    That would quantise the weights and the cache on one side and report the sum
    as the weight effect. Both arms pin `auto`. An fp8 KV cache is a worthwhile
    third arm and it is a separate arm.
  * **`--max-num-seqs`.** The FP8 arm has the spare VRAM to run wider. Batch
    composition changes reduction order and therefore changes greedy output, so
    running it wider would make it faster and also make it decode differently.

## What fits, and why the numbers are these numbers

Dense 27B, so bf16 weights are about 54 GB and fp8 weights about 27 GB. The card
is the RTX PRO 6000 at 96 GB, which is the card Muse Glimmer and Gemma 4 ran on.
That is deliberate: it is the cheapest per generated token of the three the
repository measured, by about 11 percent over the H100, and running on the same
card as the other self hosted rows is what makes the latency column comparable
across them rather than a fact about procurement.

KV per token is small here, and that is the architecture doing the work. 48 of
the 64 layers are Gated DeltaNet, whose state is constant per sequence rather
than growing with it, so only the 16 Gated Attention layers hold a cache. At 4
KV heads by 256 dims a token costs
2 (K and V) x 4 x 256 x 2 bytes x 16 layers = 64 KB. Four sequences of 40,960
tokens is 163,840 tokens, or about 10.5 GB.

At 0.90 utilisation there are about 86 GB to spend. The FP8 arm uses 27 + 10.5
and has room to spare. The bf16 arm uses 54 + 10.5 and still fits, which is the
point of sizing for it: whichever arm runs, the scheduler behaves the same way.

40,960 is not arbitrary either. It is the number the llama.cpp adapter sizes its
slots to, for the same reason: the longest full catalogue ICD-10 prompt in this
dataset is 19,130 tokens and the generation budget is 16,384, so the worst case
conversation is 35,514 tokens and has to fit whole.

## Thinking is on, because Qwen3.6 on the board is thinking

This is the equivalence that matters more than the precision one. The nearest
comparable row is Qwen3.6 27B, same family and same size, served by Groq as a
reasoning model. Running Qwen3.8 in instruct mode against it would compare a
model that thinks with one that does not and report the gap as a generation.

So `reasoning_strength` drives the model's own thinking switch: "none" or "off"
sends `enable_thinking: false` and anything else sends true, and the package
default of "medium" therefore thinks. The llama.cpp adapters prepend
"Reasoning strength: x" to the system prompt because a text prefix is the only
lever they have. This model has a real switch, so the same recorded parameter
drives that instead and the system prompt is left alone, which keeps the prompt
bytes identical to every other model's.

The cost of that choice is real and is the thing to watch. Qwen warns that
greedy decoding in thinking mode can fall into repetition, which arrives here as
a length stop and is scored as a truncation. Muse ran the same way and came back
at 4.0 percent truncation on ICD-10 full, which is just inside the 5 percent
quarantine ceiling. If a run here quarantines on truncation, that is the first
thing to suspect and `max_tokens` is the first thing to raise.

Greedy at all is the package's standing deviation from every model card in it. A
benchmark that cannot be rerun to the same number is not a benchmark. Worth
knowing that greedy under vLLM is not bitwise reproducible either: batched
matmuls reduce in an order that depends on how many sequences are in flight, so
a rerun can disagree on a handful of notes with nothing having changed.

## Note text must not reach a log

vLLM will happily log request bodies, and a request body here is a restricted
MIMIC-III note. `--disable-log-requests` is not tidiness, it is the same rule as
`scripts/check_restricted.py`. `diagnostics()` returns only the snapshot taken
at startup, before any note existed, so there is no path from a served note to a
caller's terminal.

    modal deploy coding_bench/adapters/modal_qwen38_vllm.py
    modal run coding_bench/adapters/modal_qwen38_vllm.py::smoke
"""

from __future__ import annotations

import subprocess
import time

import modal

# coding_bench is not imported at module level. Modal imports this file inside
# the GPU container, whose image carries vLLM and nothing of ours. Only the
# client class, which runs on the caller, needs the shared types.

FP8_CHECKPOINT = "Qwen/Qwen3.8-27B-FP8"
BF16_CHECKPOINT = "Qwen/Qwen3.8-27B"

# Pinned. `main` moving under a rerun is a silent change to what was measured,
# and it is the kind that leaves the manifest looking correct.
MODEL_REVISION = "main"

# Bump when the image or the generation contract changes, so a run record points
# at the exact server that produced it.
APP_NAME = "coding-bench-qwen38-v1"

CACHE_DIR = "/cache"
PORT = 8000

# The same card Muse Glimmer and Gemma 4 ran on. Cheapest per generated token of
# the three measured in scripts/gpu_bench.py, and using it keeps the latency
# column comparable across every self hosted row rather than making it a fact
# about which card was free that week. Blackwell, so fp8 is a native kernel
# rather than a dequantise-then-bf16 fallback; `smoke` prints what vLLM actually
# selected, because that decides whether the latency column means anything. It
# does not affect the codes either way.
GPU = "RTX-PRO-6000"

# Sized for the bf16 arm so that whichever arm runs, the scheduler is the same.
# See the header for the arithmetic.
MAX_MODEL_LEN = 40960
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILISATION = 0.90

# vLLM 0.17.0 is the first release that serves this architecture, and the
# multimodal path needs transformers 5.8. Pinned exactly rather than floored: a
# minor bump that changes a kernel should invalidate the cache deliberately,
# through the adapter hash, rather than arrive on a rebuild.
VLLM_VERSION = "0.17.0"
TRANSFORMERS_VERSION = "5.8.0"

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("qwen38-model-cache", create_if_missing=True)

# One image for both arms. Two images would be two sets of kernels, and the
# difference between them would arrive labelled as a difference between
# precisions.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers>={TRANSFORMERS_VERSION}",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "requests",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
        }
    )
)


# Everything the server is told, other than which checkpoint to load. Neither
# arm may vary any of it; that is the whole design and it is asserted in tests.
SERVER_ARGS: tuple[str, ...] = (
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILISATION),
    # See the header. vLLM's recipe for this model suggests fp8 here, which
    # would quantise the cache on one arm and not the other.
    "--kv-cache-dtype", "auto",
    # Splits <think>...</think> into `reasoning_content`, which `answer_text`
    # already knows how to fall back to. Without it a thinking response arrives
    # as one blob and the JSON parser has to find the answer inside the trace.
    "--reasoning-parser", "qwen3",
    # Restricted note text must never reach a log. Same rule as
    # scripts/check_restricted.py, enforced at the server instead of after it.
    "--disable-log-requests",
    "--host", "127.0.0.1",
    "--port", str(PORT),
)


def server_command(checkpoint: str) -> list[str]:
    """The full argv for one arm. The checkpoint is the only free variable."""
    return [
        "vllm", "serve", checkpoint,
        "--served-model-name", checkpoint,
        "--revision", MODEL_REVISION,
        *SERVER_ARGS,
    ]


# Kept out of the startup snapshot, chosen to answer the questions this adapter
# actually raises: which quantisation method loaded, how much of the card the
# weights took, and how much KV cache was left over.
STARTUP_KEYS = (
    "quantization", "quantisation", "fp8", "bfloat16", "dtype",
    "kv cache", "gpu blocks", "memory", "graph capturing", "model loading",
    "engine", "cuda", "error", "warning",
)


class _Server:
    """A vLLM OpenAI server holding one arm's weights.

    A plain object rather than a Modal base class. Modal's lifecycle decorators
    do not reliably survive inheritance, and the failure mode of finding that
    out on a deployed GPU is an hour of card time, so the two Modal classes
    below are thin and duplicated on purpose and all the logic lives here.
    """

    def __init__(self, arm: str, checkpoint: str):
        self.arm = arm
        self.checkpoint = checkpoint
        self.process: subprocess.Popen | None = None
        self.startup_log = ""

    def start(self) -> None:
        command = server_command(self.checkpoint)
        print(" ".join(command), flush=True)
        # Keep the server's own output. Without it a startup failure is an exit
        # code, and the reason it refused to load the weights is exactly what
        # you need to read.
        self.log = open(f"/tmp/vllm-{self.arm}.log", "w+")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self._await_ready()
        except Exception:
            # Tear down before propagating. A GPU container whose startup raised
            # is still a GPU container, and one bad flag once left a card
            # running for over an hour after the run that spawned it had died.
            self.stop()
            raise
        # Snapshot now, while the log provably contains no note text, and serve
        # every later diagnostics call from this rather than from the live file.
        self.startup_log = self._read_log()

    def _read_log(self, lines: int = 4000) -> str:
        try:
            self.log.flush()
            with open(f"/tmp/vllm-{self.arm}.log") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError as exc:
            return f"(could not read the server log: {exc})"

    def _await_ready(self, timeout_s: int = 2700) -> None:
        """Wait for /health.

        Generous, because the first start on a cold volume pulls 27 GB. Every
        start after that reads from the volume and takes a couple of minutes.
        """
        import requests

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self.process.returncode}\n{self._read_log(60)}"
                )
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    print(f"vllm ready, arm={self.arm}", flush=True)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)
        raise RuntimeError(f"vllm serve was not ready within {timeout_s}s")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def warm(self) -> dict:
        return {"status": "warm", "arm": self.arm, "model": self.checkpoint}

    def diagnostics(self, lines: int = 60) -> str:
        """The startup report, and only ever that.

        Deliberately not the live log. Reading the live file would put whatever
        the server has printed since into a caller's terminal, and this server
        is handling restricted note text.
        """
        kept = [
            line for line in self.startup_log.splitlines()
            if any(key in line.lower() for key in STARTUP_KEYS)
        ]
        return "\n".join(kept[-lines:])

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> dict:
        import requests

        started = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={
                "model": self.checkpoint,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Greedy when temperature is 0, so a rerun reproduces the run as
                # far as batched reduction order allows. See the header.
                "top_p": 1.0 if temperature == 0.0 else 0.95,
                "top_k": -1 if temperature == 0.0 else 20,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            timeout=1800,
        )
        response.raise_for_status()
        payload = response.json()
        elapsed = time.perf_counter() - started

        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        return {
            # The whole message, both channels. `--reasoning-parser qwen3` routes
            # thinking to `reasoning_content`, and a model that never leaves that
            # channel looks exactly like a model with nothing to say. That cost
            # 124 of 312 notes on Gemma 4. The choice of channel is made client
            # side by `answer_text`, because this container cannot import
            # coding_bench.
            "message": choice["message"],
            "stop_reason": choice.get("finish_reason") or "unknown",
            "latency_s": elapsed,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }


# The two arms. Duplicated rather than generated, because Modal resolves the
# decorated class at deploy time and a loop over a dict produces one class that
# cannot be addressed by name. The bodies delegate immediately, so what is
# duplicated is six lines of plumbing and no behaviour.

_CLS_KWARGS = dict(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: model_cache},
    timeout=3600,
    # Idle GPU time is pure cost. Reloading from the volume is cheap next to
    # holding a card open between runs.
    scaledown_window=90,
    # MUST STAY 1. Concurrent callers would otherwise each get a container, and
    # each container is a fresh GPU loading its own copy of the weights.
    max_containers=1,
)


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class Qwen38FP8:
    """`Qwen/Qwen3.8-27B-FP8`, Qwen's own fine grained fp8 at block size 128.

    The build on the board.
    """

    @modal.enter()
    def start(self):
        self.server = _Server("fp8", FP8_CHECKPOINT)
        self.server.start()

    @modal.exit()
    def stop(self):
        self.server.stop()

    @modal.method()
    def warm(self) -> dict:
        return self.server.warm()

    @modal.method()
    def diagnostics(self, lines: int = 60) -> str:
        return self.server.diagnostics(lines)

    @modal.method()
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        enable_thinking: bool = True,
    ) -> dict:
        return self.server.complete(system, user, max_tokens, temperature, enable_thinking)


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class Qwen38BF16:
    """`Qwen/Qwen3.8-27B` as released, bf16 weights.

    Registered and never queued. This is the arm that would settle what FP8
    costs, and it is not the arm the board needs in order to be useful. Run it
    against the FP8 rows only, on the same task and candidate space, and read
    the paired interval on the leaderboard rather than the raw gap.
    """

    @modal.enter()
    def start(self):
        self.server = _Server("bf16", BF16_CHECKPOINT)
        self.server.start()

    @modal.exit()
    def stop(self):
        self.server.stop()

    @modal.method()
    def warm(self) -> dict:
        return self.server.warm()

    @modal.method()
    def diagnostics(self, lines: int = 60) -> str:
        return self.server.diagnostics(lines)

    @modal.method()
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        enable_thinking: bool = True,
    ) -> dict:
        return self.server.complete(system, user, max_tokens, temperature, enable_thinking)


ARMS: dict[str, tuple[str, str]] = {
    # arm -> (model id recorded in the run manifest, Modal class name)
    "fp8": (FP8_CHECKPOINT, "Qwen38FP8"),
    "bf16": (BF16_CHECKPOINT, "Qwen38BF16"),
}

# reasoning_strength values that mean "answer directly". Anything else leaves
# thinking on, which is what the package default of "medium" does and what the
# Qwen3.6 row on the board was measured with. Kept as a constant so the chain,
# the tests and the adapter cannot disagree about what "none" means.
NO_THINKING = frozenset({"none", "off", "", "0"})


def thinking_enabled(reasoning_strength: str | None) -> bool:
    return str(reasoning_strength or "").strip().lower() not in NO_THINKING


class Qwen38Client:
    """Client side handle satisfying the LLMClient protocol, for one arm.

    Works from a laptop and from inside another Modal function, which is what
    eval_remote.py needs so that note text and weights stay in the same place.
    """

    def __init__(
        self,
        arm: str = "fp8",
        temperature: float = 0.0,
        reasoning_strength: str = "medium",
        app_name: str = APP_NAME,
    ):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}. Known: {sorted(ARMS)}")
        checkpoint, class_name = ARMS[arm]
        self.arm = arm
        # The exact HF repo id, so the manifest names something that reproduces
        # the run. The FP8 repo carries its precision in its own name, which is
        # why neither arm needs a `:tag` the way the GGUF builds do.
        self.model_id = checkpoint
        self.temperature = temperature
        self.reasoning_strength = reasoning_strength
        self.enable_thinking = thinking_enabled(reasoning_strength)
        self._remote = modal.Cls.from_name(app_name, class_name)()

    def complete(self, system: str, user: str, max_tokens: int):
        from coding_bench.approaches.base import Completion, Truncated, answer_text

        result = self._remote.complete.remote(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        # An OpenAI compatible server reports a cut off generation as
        # `finish_reason: "length"`. Truncation and "no codes apply" are opposite
        # findings and must never arrive looking the same. On a thinking model
        # under greedy decoding this is the failure that actually happens: the
        # trace loops, hits the cap, and no JSON is ever produced.
        if result["stop_reason"] == "length":
            raise Truncated(
                self.model_id, "length", produced_tokens=result["usage"]["completion_tokens"]
            )
        return Completion(
            text=answer_text(result["message"]),
            stop_reason=result["stop_reason"],
            latency_s=result["latency_s"],
            usage=result["usage"],
        )


@app.local_entrypoint()
def smoke(arm: str = "fp8", prompt: str = "Count from 1 to 40, then reply with the JSON {\"ok\": true}."):
    """Check the server answers, and measure throughput honestly.

    Two requests, because the first carries graph capture and would understate
    the rate. The second is the number that predicts what a 578 note run costs.

    The startup report at the end is where you confirm vLLM loaded a native fp8
    kernel rather than dequantising to bf16. That decides whether the latency
    column means anything and does not affect the codes either way.
    """
    if arm not in ARMS:
        raise SystemExit(f"Unknown arm {arm!r}. Known: {sorted(ARMS)}")
    _checkpoint, class_name = ARMS[arm]
    model = modal.Cls.from_name(APP_NAME, class_name)()
    print(model.warm.remote())

    for label in ("cold", "warm"):
        result = model.complete.remote(
            system="You answer concisely.", user=prompt, max_tokens=512
        )
        produced = result["usage"]["completion_tokens"]
        print(
            f"{label}: stop_reason={result['stop_reason']} "
            f"latency={result['latency_s']:.1f}s tokens={produced} "
            f"rate={produced / max(result['latency_s'], 0.001):.1f} tok/s"
        )

    print(f"\n--- vllm startup report, arm={arm} ---")
    print(model.diagnostics.remote())
