"""Gemma 4 26B A4B, the QAT UD-Q4_K_XL GGUF, served by llama.cpp on a Modal GPU.

**This file is a reconstruction.** Two runs of this model are banked in
`results/runs/`, both CPT at gold candidates over all 312 notes, and the adapter
that produced them was written on another machine and never pushed. Their
manifests name commits `a5d3c7a9` and `4d973679`, neither of which exists in
this history. So the code here reproduces the configuration the run manifests
record, and it is not the original bytes.

Two consequences, both worth knowing before trusting a comparison:

  * `adapter_sha256` in those two records will not match this file. The repo
    treats that hash as the audit trail for which code produced a number, and
    for these two records that trail leads somewhere this checkout cannot see.
  * the prediction cache keys on the same hash, so re-running against this
    adapter will miss every cached note and pay for the whole run again.

Recover the original if the other machine still has it, and prefer it to this.

The model is a mixture of experts, 26B total parameters with about 4B active per
token, so it generates far faster than its size suggests and needs roughly 15GB
of weights at this quantisation. QAT means the quantisation was trained rather
than applied afterwards, and UD-Q4_K_XL is Unsloth's dynamic build, which keeps
the layers that suffer most at higher precision.

    modal deploy coding_bench/adapters/modal_gemma4_gguf.py
    modal run coding_bench/adapters/modal_gemma4_gguf.py::smoke

## What the two banked runs say, and why both are quarantined

Both failed 10.9% and 9.9% of notes against a 5% ceiling, and every failure is
truncation, on a task whose gold answer averages 1.01 codes. The second run
doubled `max_tokens` from 16,384 to 32,768 and dropped concurrency from 4 to 3,
which moved truncation by one point and nearly doubled mean latency.

Separately, 124 of 312 notes came back with no codes, no error and no
truncation. Those are not the model declining to code. They average 1,810
completion tokens against 882 for the notes it answered, so the model wrote more
than twice as much on the notes that produced nothing. An honest empty answer is
a dozen tokens.

## The cause, which the smoke test reproduces in one request

Both symptoms are the same bug, and it is ours rather than the model's.

Ask this server to count to forty and it returns `finish_reason: length`, 512
completion tokens, and an **empty** `content`. The generated text is all in
`reasoning_content`. llama.cpp under `--jinja` splits a reasoning model's output
by channel, and Gemma 4's template puts thinking in a channel this adapter never
reads, so a model that is still thinking when the budget runs out looks exactly
like a model that found nothing to say.

That accounts for both numbers at once. The notes that truncated are the ones
that ran the budget out mid thought, and the 124 silent notes are the ones where
generation ended without the model ever leaving the reasoning channel.

It was not specific to Gemma. `modal_muse_gguf.py` read `content` the same way,
its CPT run is silent on 120 of 150 notes, and 83 notes are silent for both
models. The Groq models answered those same notes at 0.92 micro F1 with no
failures, because their adapter is a different code path. What Muse and Gemma
shared is this one.

Both adapters now return every channel the server produced and let
`approaches/base.py::answer_text` choose, which prefers `content` and falls back
to the reasoning channel only when `content` is empty. The two banked runs above
predate that fix and are kept as the record of what it cost.

What the fix does not do is stop a model thinking past its budget. The notes
that truncated have no answer in any channel, so they truncate again, and the
truncation rate is a separate problem from the silence.
"""

from __future__ import annotations

import subprocess
import time

import modal

# coding_bench is deliberately not imported at module level. Modal imports this
# file inside the GPU container, whose image carries llama.cpp and none of ours.

HF_REPO = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
MODEL_FILE = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"

# The label that goes in every run manifest, and the string the leaderboard
# groups on. It names Google's model and the exact build, because a QAT dynamic
# 4 bit is not the released bf16 model and must never be tabulated as though it
# were. Changing this string orphans the two banked runs from anything measured
# afterwards, so it matches them character for character on purpose.
MODEL_ID = "google/gemma-4-26B-A4B-it:qat-UD-Q4_K_XL"
APP_NAME = "coding-bench-gemma4-gguf-v1"

CACHE_DIR = "/cache"
PORT = 8080

# Inherited from the Muse adapter, where it was chosen by measurement, see
# scripts/gpu_bench.py. It has not been measured for this model, and the
# arithmetic that picked it does not obviously carry across: that comparison was
# between dense 30B models bound by weight reads, while 4B active parameters per
# token shifts the balance. Treat this as an untested default rather than as a
# result, and measure before quoting a cost per token.
GPU = "RTX-PRO-6000"

# Four notes in flight, which is what both banked runs used at the server even
# though the second lowered the client side concurrency to 3.
PARALLEL_SLOTS = 4

# Every slot holds one whole conversation, so it has to fit the longest prompt
# plus the entire generation budget. The full catalogue prompts measured 19,130
# tokens at the longest on this dataset, with Muse's tokenizer rather than this
# one, so treat that as approximate. Against the 32,768 generation budget the
# second run used, the worst case is close to 52k, and this leaves headroom.
#
# Both banked runs are CPT at gold candidates, where the prompt is a note plus
# roughly one candidate code, so neither came anywhere near this ceiling. The
# sizing is here so that a full catalogue ICD-10 run does not silently overflow
# a slot and come back truncated, which scores as an empty answer while still
# paying full price for the GPU time.
CONTEXT_PER_SLOT = 57344
TOTAL_CONTEXT = PARALLEL_SLOTS * CONTEXT_PER_SLOT

# The upstream server image ships the binary in /app next to its ggml shared
# objects, and puts neither on PATH nor on the loader path.
LLAMA_SERVER = "/app/llama-server"

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("gemma4-gguf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
    .pip_install("huggingface_hub>=0.30.0", "requests")
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "LD_LIBRARY_PATH": "/app",
            "PATH": "/app:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin",
        }
    )
)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: model_cache},
    timeout=3600,
    scaledown_window=90,
    # THIS MUST STAY 1, for the same reason it is 1 on the Muse server.
    # Concurrent callers would otherwise each get their own container, and each
    # container is a fresh GPU loading its own copy of the weights.
    max_containers=1,
)
# One container, several inputs in flight. The llama-server slots are what
# actually serve them. This only stops Modal serialising requests at the door.
@modal.concurrent(max_inputs=PARALLEL_SLOTS)
class Gemma4GGUF:
    """A llama.cpp server holding the quantised weights, reused across notes."""

    @modal.enter()
    def start_server(self):
        from huggingface_hub import hf_hub_download

        print(f"Fetching {MODEL_FILE}", flush=True)
        model_path = hf_hub_download(HF_REPO, MODEL_FILE, cache_dir=CACHE_DIR)
        model_cache.commit()

        command = [
            LLAMA_SERVER,
            "--model", model_path,
            "--gpu-layers", "999",
            "--ctx-size", str(TOTAL_CONTEXT),
            "--parallel", str(PARALLEL_SLOTS),
            # Batch waiting requests into each forward pass, which is where the
            # throughput of several concurrent notes actually comes from.
            "--cont-batching",
            "--port", str(PORT),
            "--host", "127.0.0.1",
            # Use the model's own chat template. Gemma's differs from Muse's in
            # ways that matter to a system prompt, see the note in complete().
            "--jinja",
        ]
        # The repo also ships MTP drafts for speculative decoding. They are not
        # wired up here. Nothing in either banked run's numbers suggests
        # speculation was contributing, and claiming a speedup that is not
        # actually running is exactly the trap the Muse adapter documents.
        print(" ".join(command), flush=True)
        # Keep the server's own output. Without it a startup failure is only an
        # exit code, and the reason it refused the model is what you need.
        self.server_log = open("/tmp/llama-server.log", "w+")
        self.server = subprocess.Popen(command, stdout=self.server_log, stderr=subprocess.STDOUT)
        try:
            self._await_ready()
        except Exception:
            # Tear the server down before propagating. A GPU container whose
            # startup raised is still a GPU container, and it will sit there
            # billing until something stops it.
            self._terminate()
            raise

    def _terminate(self):
        server = getattr(self, "server", None)
        if server is None or server.poll() is not None:
            return
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    def _server_output(self, lines: int = 40, from_start: bool = False) -> str:
        """The server log, from either end.

        The end is what a crash needs. The beginning is what a throughput
        question needs, and those are not the same lines: once the server has
        answered a few requests its per request logging has pushed the startup
        report out of any tail worth printing.
        """
        try:
            self.server_log.flush()
            with open("/tmp/llama-server.log") as handle:
                rows = handle.readlines()
            return "".join(rows[:lines] if from_start else rows[-lines:])
        except OSError as exc:
            return f"(could not read the server log: {exc})"

    def _await_ready(self, timeout_s: int = 900):
        import requests

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.server.returncode}\n"
                    f"{self._server_output()}"
                )
            try:
                response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
                if response.status_code == 200:
                    print("llama-server ready", flush=True)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        raise RuntimeError(f"llama-server was not ready within {timeout_s}s")

    @modal.exit()
    def stop_server(self):
        self._terminate()

    @modal.method()
    def warm(self) -> dict:
        return {"status": "warm", "model": MODEL_ID}

    @modal.method()
    def diagnostics(self, lines: int = 200, from_start: bool = True) -> str:
        """The server's own startup report, which says where the layers went."""
        return self._server_output(lines, from_start=from_start)

    @modal.method()
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        reasoning_strength: str = "medium",
    ) -> dict:
        """Generate once through the OpenAI compatible endpoint."""
        import requests

        # Gemma's chat template has no system role. llama.cpp with --jinja
        # handles that by folding a system message into the first user turn,
        # which is fine, but it means the reasoning strength line lands inside
        # the same turn rather than above it. Prepending it to the system text
        # keeps the ordering the other adapters produce.
        if reasoning_strength:
            system = f"Reasoning strength: {reasoning_strength}\n\n{system}"

        started = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Greedy at temperature 0, so a rerun reproduces the run.
                "top_p": 1.0 if temperature == 0.0 else 0.95,
                "top_k": -1 if temperature == 0.0 else 64,
            },
            timeout=1800,
        )
        response.raise_for_status()
        payload = response.json()
        elapsed = time.perf_counter() - started

        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        return {
            # Every channel the server produced, chosen between on the client
            # side. Picking one here would put that decision in a container
            # that cannot import coding_bench, so the two llama.cpp adapters
            # would each need their own copy of the rule. See answer_text in
            # approaches/base.py, which is the one copy.
            "message": choice["message"],
            "stop_reason": choice.get("finish_reason") or "unknown",
            "latency_s": elapsed,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }


class Gemma4GGUFClient:
    """Client side handle satisfying the LLMClient protocol."""

    model_id = MODEL_ID

    def __init__(
        self,
        temperature: float = 0.0,
        reasoning_strength: str = "medium",
        app_name: str = APP_NAME,
    ):
        self.temperature = temperature
        self.reasoning_strength = reasoning_strength
        self._remote = modal.Cls.from_name(app_name, "Gemma4GGUF")()

    def complete(self, system: str, user: str, max_tokens: int):
        from coding_bench.approaches.base import Completion, Truncated, answer_text

        result = self._remote.complete.remote(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=self.temperature,
            reasoning_strength=self.reasoning_strength,
        )
        # A cut off generation is a typed error, never a partial parse. This is
        # the path that produced the 10% failure rate in both banked runs, and
        # reading the reasoning channel does not change it: a model interrupted
        # mid thought has no answer in any channel.
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

    Two requests, because the first carries warmup and would understate the
    rate. The second is the number that predicts what a full run costs.

    The response text is printed, not just the token count. A smoke test that
    reports `stop_reason=length` and nothing else leaves you guessing whether
    the model is slow or whether it never stops talking, and on this model that
    is the entire question. The prompt is synthetic, so nothing restricted can
    reach a log this way.
    """
    from coding_bench.approaches.base import answer_text

    model = Gemma4GGUF()
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
        message = result["message"]
        sizes = {k: len(v) for k, v in message.items() if isinstance(v, str) and k != "role"}
        text = answer_text(message)
        print(f"  channels {sizes}, {len(text)} chars taken")
        print(f"  opens: {text[:160]!r}\n")

    print("--- llama-server startup report ---")
    report = model.diagnostics.remote()
    wanted = ("offload", "layer", "cuda", "device", "expert", "buffer")
    matched = [line for line in report.splitlines() if any(k in line.lower() for k in wanted)]
    # Say when there was nothing to show. An empty section under a heading reads
    # as "the model loaded and had nothing to report", which is not the same
    # thing as the server having logged nothing at all.
    print("\n".join(matched) if matched else f"(no matching lines in {len(report)} chars of log)")
