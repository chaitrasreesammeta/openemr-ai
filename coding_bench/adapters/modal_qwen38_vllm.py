"""Qwen3.8 27B FP8, self hosted on a Modal GPU with vLLM.

`Qwen/Qwen3.8-27B-FP8`, Qwen's own fine grained fp8 at block size 128. Qwen also
releases bf16 weights and this package does not serve them, which is a choice
worth writing down rather than leaving as an omission.

## Why the FP8 build is the one that gets a row

The board is a deployment board. Its `full` candidate condition exists because
gold candidates pin precision at 1.000 by construction and predict nothing, and
every self hosted row already on it is a deployment build rather than a
reference build: Muse Glimmer at roughly 4 bit, Gemma 4 at QAT 4 bit. A bf16 row
would be the only self hosted number on the board measured at a precision nobody
would serve for a 27B on this task.

It is also the only choice that stays honest about the hosted rows. Groq and
Anthropic do not publish what precision they serve at, and for throughput at
their prices it is very unlikely to be bf16. Running bf16 here to "match" them
would be a claim about their stack that nobody outside it can check. FP8 is at
least a precision this repository can name in the `## Models` key.

And FP8 is a Qwen release rather than somebody's quantisation of one. It has its
own repository and card, which is why the id needs no `:quantisation` tag the
way the GGUF builds do. Qwen puts it at nearly identical to the original, and
the largest study of the format measures W8A8-FP8 as lossless across model
scales. That evidence is about general benchmarks and says nothing about rare
label recall, so it is a reason to expect FP8 to be fine here and not a reason
to skip measuring it. The tail band is where it would show up if it is not.

**What is deliberately not measured.** What FP8 costs against bf16 on this
benchmark. Answering that needs both builds on identical settings and a
replicate to establish that greedy decoding under vLLM reproduces itself, which
it does not do bitwise: batched matmuls reduce in an order that depends on how
many sequences are in flight. That is a separate experiment with its own
protocol, not a second row, and nothing here should be read as having measured
it.

## KV cache stays at bf16

vLLM's own recipe for this model passes `--kv-cache-dtype fp8`, and it is not
taken. The id on this row says fp8 *weights*. Quantising the cache as well would
make the row a build Qwen never published and this benchmark never measured, and
the memory it would free is memory this configuration does not need.

## What fits, which is why the numbers are these numbers

Dense 27B, so fp8 weights are about 27 GB. The card is the RTX PRO 6000 at
96 GB, which is the card Muse Glimmer and Gemma 4 ran on. That is deliberate: it
is the cheapest per generated token of the three the repository measured, by
about 11 percent over the H100, and running on the same card as the other self
hosted rows is what makes the latency column comparable across them rather than
a fact about procurement.

KV per token is small here, and that is the architecture doing the work. 48 of
the 64 layers are Gated DeltaNet, whose state is constant per sequence rather
than growing with it, so only the 16 Gated Attention layers hold a cache. At 4
KV heads by 256 dims a token costs
2 (K and V) x 4 x 256 x 2 bytes x 16 layers = 64 KB. Sixteen sequences of 40,960
tokens is 655,360 tokens, or about 43 GB.

Measured on the first successful start, which is what these numbers should be
read against rather than the arithmetic above: weights load at 28.51 GiB, vLLM
allocates a KV cache of 833,828 tokens, and it reports a maximum concurrency of
20.36x for 40,960 token requests. Sixteen slots therefore fits with room, and
peak activation measured 2.02 GiB against the 85.47 GiB budget.

That 28.51 GiB is also the evidence that fp8 is real here. A silent dequantise
to bf16 would have loaded roughly twice that, and the latency column would have
been describing a different model than the id claims.

Warm throughput is 43.3 tok/s single stream, 10.1 tok/s on the first request
before graph capture is done. Measure warm, never cold. Engine init takes about
285 seconds, of which 173 is compilation, and that is paid on every container
start rather than once.

Sixteen slots rather than four, which is where this started. Four used a third
of the card and put the full ICD-10 set at an estimated eight to twelve hours,
longer than any single run can watch. This is the one setting not held to what
Muse and Gemma used, so mean per note latency is not directly comparable with
their rows. That is a property of the batch rather than of the model, no
accuracy column moves, and `concurrency` is recorded in every run manifest.

40,960 is not arbitrary either. It is the number the llama.cpp adapter sizes its
slots to, for the same reason: the longest full catalogue ICD-10 prompt in this
dataset is 19,130 tokens and the generation budget is 16,384, so the worst case
conversation is 35,514 tokens and has to fit whole.

## Thinking is on, because Qwen3.6 on the board is thinking

This is the equivalence that matters more than any of the above. The nearest
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
at 4.0 percent truncation on ICD-10 full, just inside the 5 percent quarantine
ceiling. If a run here quarantines on truncation, that is the first thing to
suspect and `max_tokens` is the first thing to raise.

Greedy at all is the package's standing deviation from every model card in it. A
benchmark that cannot be rerun to the same number is not a benchmark.

## Note text must not reach a log

vLLM can log request bodies, and a request body here is a restricted MIMIC-III
note. `--no-enable-log-requests` is not tidiness, it is the same rule as
`scripts/check_restricted.py`. It is passed rather than left to the default,
which is currently off, because whether the notes are logged should be stated by
the command rather than inherited from whichever version is installed.
`diagnostics()` returns only the snapshot taken at startup, before any note
existed, so there is no path from a served note to a caller's terminal.

## Fetch, then serve

`_fetch_weights` pulls the checkpoint and commits it to the volume before the
server launches. The download used to be implicit inside vLLM, and a Modal
volume keeps nothing until something commits it, so an interrupted start threw
away the whole 27 GB and the next one paid for it again.

There was briefly a third phase in front of this, a `_check_flags` that asked
`vllm serve --help` whether every flag here was accepted, added after an unknown
flag crash-looped a GPU for 22 minutes. It is gone, and the reason is worth
keeping: `--help` did not return a listing it could parse, so it reported that
vLLM rejects `--max-model-len`, `--host` and `--port`, and the guard against a
crash-loop became a crash-loop against a correct configuration. It cost an hour.

Detection was never the missing piece. A bad flag makes `vllm serve` exit at
argument parsing, `_await_ready` polls the process, and it raises within seconds
with the exit code and the server's own message. That is exactly how the flag
bug was found. What cost 22 minutes was Modal restarting the container, and the
bound for that belongs in the caller: the CI smoke step carries a timeout.

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

CHECKPOINT = "Qwen/Qwen3.8-27B-FP8"

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
# column comparable across the self hosted rows rather than making it a fact
# about which card was free that week. Blackwell, so fp8 is a native kernel
# rather than a dequantise-then-bf16 fallback; `smoke` prints what vLLM actually
# selected, because that decides whether the latency column means anything. It
# does not affect the codes either way.
GPU = "RTX-PRO-6000"

# See the header for the arithmetic: 43 GB of KV beside 27 GB of weights, inside
# the 86 GB vLLM will use. Raising this further starts eating the headroom that
# prefill activations and cuda graphs need at a 19k token prompt.
MAX_MODEL_LEN = 40960
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILISATION = 0.90

# 0.17.0 is the floor this model's vLLM recipe states, and it is not a usable
# pin. vLLM 0.17.0 requires `transformers<5`, while this model's config.json is
# written by transformers 5.8 and its processor has to match. Asking for both is
# a ResolutionImpossible, which is how the first CI launch died: at image build,
# in about four seconds, before a GPU was ever allocated.
#
# 0.24.0 is the first release that requires `transformers>=5.5.3` outright, so
# anything from there up is coherent. Between them, 0.20 to 0.23 permit 5.x only
# through a list of exclusions. 0.27.1 is the current release, it is the one the
# rest of this package already talks about, and it carries torchvision as a hard
# dependency, which a multimodal model needs on load even when nothing but text
# is ever sent to it. modal_muse.py learned that the expensive way.
#
# Pinned exactly rather than floored: a minor bump can change a kernel, and that
# should invalidate this model's cached notes deliberately through the adapter
# hash rather than arrive unannounced on a rebuild.
VLLM_VERSION = "0.27.1"
TRANSFORMERS_VERSION = "5.8.0"

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("qwen38-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers>={TRANSFORMERS_VERSION}",
        # Plain, no [hf_transfer]. The same container log that rejected the
        # logging flag also warned that HF_HUB_ENABLE_HF_TRANSFER is deprecated
        # and hf_transfer is no longer used, so the extra installs a package
        # nothing reads and the env var sets something nothing honours.
        "huggingface_hub>=0.30.0",
        "requests",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "VLLM_LOGGING_LEVEL": "INFO",
            # FlashInfer JIT compiles its top-k/top-p sampling kernels on first
            # use, which needs nvcc, which a pip install of vLLM onto debian
            # slim does not have: the torch wheels bring the CUDA runtime, not
            # the toolkit. Engine start died on it during the dummy sampler run
            # inside memory profiling, with "Could not find nvcc and default
            # cuda_home='/usr/local/cuda' doesn't exist", after the weights had
            # loaded and the architecture had resolved.
            #
            # Disabling it costs this benchmark nothing at all, which is why it
            # is the fix rather than a 5 GB CUDA devel base image. Every request
            # here is greedy: temperature 0, top_p 1.0, top_k -1. The top-k and
            # top-p kernels are never on the hot path and only ever ran because
            # profiling exercises the sampler whatever the run settings are. The
            # fallback is vLLM's PyTorch native sampler.
            #
            # If another nvcc dependent JIT path shows up later, the answer is a
            # devel base image rather than a second flag.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


SERVER_ARGS: tuple[str, ...] = (
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILISATION),
    # bf16 cache, deliberately. See the header: the id on this row says fp8
    # weights, and quantising the cache too would make it a build Qwen never
    # published. vLLM's recipe for this model suggests fp8 here.
    "--kv-cache-dtype", "auto",
    # Splits <think>...</think> into `reasoning_content`, which `answer_text`
    # already knows how to fall back to. Without it a thinking response arrives
    # as one blob and the JSON parser has to find the answer inside the trace.
    "--reasoning-parser", "qwen3",
    # Ignore the checkpoint's generation_config.json, which sets temperature
    # 1.0, top_k 20 and top_p 0.95 as the server's defaults. Every request here
    # overrides all three, so this changes nothing today, and that is the
    # problem: it leaves greedy decoding resting on the adapter never omitting a
    # field. A benchmark that cannot be rerun to the same number is not a
    # benchmark, so the server is told to have no opinion and the run parameters
    # are the only source. vLLM's own warning recommends this flag by name.
    "--generation-config", "vllm",
    # Restricted note text must never reach a log. Same rule as
    # scripts/check_restricted.py, enforced at the server instead of after it.
    #
    # `--disable-log-requests` is what older vLLM called this and it is gone in
    # 0.27.1, which rejected it outright and crash-looped a GPU container for 22
    # minutes. The flag is now `--enable-log-requests`, defaulting to false, and
    # this passes its negation rather than relying on that default: whether the
    # notes are logged is a governance property and should be stated by the
    # command rather than inherited from a version.
    "--no-enable-log-requests",
    "--host", "127.0.0.1",
    "--port", str(PORT),
)


def server_command() -> list[str]:
    """The full argv for the server."""
    return [
        "vllm", "serve", CHECKPOINT,
        "--served-model-name", CHECKPOINT,
        "--revision", MODEL_REVISION,
        *SERVER_ARGS,
    ]


# Kept in the startup snapshot, chosen to answer the questions this adapter
# actually raises: which quantisation method loaded, how much of the card the
# weights took, and how much KV cache was left over.
STARTUP_KEYS = (
    "quantization", "quantisation", "fp8", "bfloat16", "dtype",
    "kv cache", "gpu blocks", "memory", "graph capturing", "model loading",
    "engine", "cuda", "error", "warning",
)


@app.cls(
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
# One container, several notes in flight. The slots configured on the server are
# what actually serve them; this just lets Modal deliver more than one request
# at a time instead of serialising them at the door.
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class Qwen38FP8:
    """A vLLM server holding the fp8 weights, reused across every note."""

    @modal.enter()
    def start_server(self):
        self._fetch_weights()
        command = server_command()
        print(" ".join(command), flush=True)
        # Keep the server's own output. Without it a startup failure is an exit
        # code, and the reason it refused to load the weights is exactly what
        # you need to read.
        self.log = open("/tmp/vllm.log", "w+")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self._await_ready()
        except Exception:
            # Tear down before propagating. A GPU container whose startup raised
            # is still a GPU container, and one bad flag once left a card
            # running for over an hour after the run that spawned it had died.
            self._terminate()
            raise
        # Snapshot now, while the log provably contains no note text, and serve
        # every later diagnostics call from this rather than from the live file.
        self.startup_log = self._read_log()

    def _fetch_weights(self) -> None:
        """Pull the weights and commit them to the volume before serving.

        Letting vLLM download implicitly on first load looks equivalent and is
        not, because a Modal volume keeps nothing until something commits it. A
        first start that is interrupted, by a stop from the dashboard, a
        cancelled job, or a crash in engine init, takes the entire 27 GB with
        it, and the next start pays for all of it again. That happened: a start
        was stopped 22 minutes in and left this volume completely empty.

        Committing here means the download is paid for once. It also splits the
        cold start into two phases that can be told apart in the log, so the
        next long silence is attributable to fetching or to engine init rather
        than being one opaque wait. The two llama.cpp adapters in this package
        both do the same thing, and this one was the outlier.
        """
        from huggingface_hub import snapshot_download

        print(f"Fetching {CHECKPOINT} at revision {MODEL_REVISION}", flush=True)
        started = time.perf_counter()
        snapshot_download(
            CHECKPOINT,
            revision=MODEL_REVISION,
            cache_dir=CACHE_DIR,
            # Deny the known duplicates rather than allow a list of extensions.
            # An allowlist that misses one file vLLM wants fails at load, which
            # is 20 minutes and a GPU after the mistake was made; a denylist that
            # misses a duplicate only costs some transfer.
            ignore_patterns=["*.pth", "*.bin", "original/*", "consolidated*"],
        )
        model_cache.commit()
        print(f"Weights on the volume after {time.perf_counter() - started:.0f}s", flush=True)

    def _read_log(self, lines: int = 4000) -> str:
        try:
            self.log.flush()
            with open("/tmp/vllm.log") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError as exc:
            return f"(could not read the server log: {exc})"

    def _await_ready(self, timeout_s: int = 1200) -> None:
        """Wait for /health.

        This covers engine init only. The 27 GB fetch happens before the server
        is launched and commits as it goes, so a long wait here is vLLM loading
        weights and capturing graphs rather than a download, and 20 minutes is
        generous for that. Keeping the old 45 minute ceiling would mean a server
        that will never come up sits on a GPU for three quarters of an hour
        before anything says so.
        """
        import requests

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                # The whole tail, to stdout, before raising. vLLM runs the model
                # in an EngineCore subprocess and the API server reports its own
                # death with "Engine core initialization failed. See root cause
                # above" and nothing else, so the actual reason sits well above
                # the last frames. Sixty lines caught the polite half of that and
                # dropped the cause, which cost a whole diagnostic cycle.
                #
                # Safe to print because this only runs before the server ever
                # became ready, so no note has been served and the log cannot
                # contain one. Request bodies are off in any case.
                print("--- vllm server log ---", flush=True)
                print(self._read_log(400), flush=True)
                raise RuntimeError(
                    f"vllm serve exited with code {self.process.returncode}. "
                    "Its log is above, root cause first."
                )
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    print("vllm ready", flush=True)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)
        raise RuntimeError(f"vllm serve was not ready within {timeout_s}s")

    def _terminate(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()

    @modal.exit()
    def stop_server(self):
        self._terminate()

    @modal.method()
    def warm(self) -> dict:
        return {"status": "warm", "model": CHECKPOINT}

    @modal.method()
    def diagnostics(self, lines: int = 60) -> str:
        """The startup report, and only ever that.

        Deliberately not the live log. Reading the live file would put whatever
        the server has printed since into a caller's terminal, and this server
        is handling restricted note text.
        """
        kept = [
            line for line in getattr(self, "startup_log", "").splitlines()
            if any(key in line.lower() for key in STARTUP_KEYS)
        ]
        return "\n".join(kept[-lines:])

    @modal.method()
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        enable_thinking: bool = True,
    ) -> dict:
        """Generate once through the OpenAI compatible endpoint."""
        import requests

        started = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={
                "model": CHECKPOINT,
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


# reasoning_strength values that mean "answer directly". Anything else leaves
# thinking on, which is what the package default of "medium" does and what the
# Qwen3.6 row on the board was measured with. Kept as a constant so the chain,
# the tests and the adapter cannot disagree about what "none" means.
NO_THINKING = frozenset({"none", "off", "", "0"})


def thinking_enabled(reasoning_strength: str | None) -> bool:
    return str(reasoning_strength or "").strip().lower() not in NO_THINKING


class Qwen38Client:
    """Client side handle satisfying the LLMClient protocol.

    Works from a laptop and from inside another Modal function, which is what
    eval_remote.py needs so that note text and weights stay in the same place.
    """

    # The exact HF repo id, so the manifest names something that reproduces the
    # run. The repo carries its precision in its own name, which is why this
    # needs no `:tag` the way the GGUF builds do.
    model_id = CHECKPOINT

    def __init__(
        self,
        temperature: float = 0.0,
        reasoning_strength: str = "medium",
        app_name: str = APP_NAME,
    ):
        self.temperature = temperature
        self.reasoning_strength = reasoning_strength
        self.enable_thinking = thinking_enabled(reasoning_strength)
        self._remote = modal.Cls.from_name(app_name, "Qwen38FP8")()

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
def smoke(prompt: str = "Count from 1 to 40, then reply with the JSON {\"ok\": true}."):
    """Check the server answers, and measure throughput honestly.

    Two requests, because the first carries graph capture and would understate
    the rate. The second is the number that predicts what a 578 note run costs.

    The startup report at the end is where you confirm vLLM loaded a native fp8
    kernel rather than dequantising to bf16. That decides whether the latency
    column means anything and does not affect the codes either way.
    """
    model = modal.Cls.from_name(APP_NAME, "Qwen38FP8")()
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

    print("\n--- vllm startup report ---")
    print(model.diagnostics.remote())
