"""Muse Glimmer 30B, K-quant GGUF, served by llama.cpp on a Modal GPU.

This is a **different model from the bf16 weights** and is named as such
everywhere it appears. Its weights are quantised to roughly 4 bits, and Meta
measures 0.2 percent average degradation for the dynamic K-quant across fifteen
general benchmarks. That average should not be assumed to hold here: this
benchmark's whole point is the long tail, and quantisation costs rare label
recall first. Comparing this against a bf16 number without saying so would be a
straightforward error, which is why the model id carries the quantisation.

Why GGUF at all: the bf16 path through transformers generates unbatched at a
speed that put a 578 note run at roughly sixty hours. This path measures 72
tok/s warm on an H100, which brings the same run to about two hours.

That 72 tok/s is Meta's quoted baseline, not their 233 tok/s speculative figure.
The DFlash drafter does not load under llama.cpp today, see the note on the
--spec-draft-model flag below, so speculation is not contributing. Measure warm,
never cold: the first request carries graph capture and KV allocation and
reports roughly a tenth of the real rate.

    modal deploy coding_bench/adapters/modal_muse_gguf.py
    modal run coding_bench/adapters/modal_muse_gguf.py::smoke
"""

from __future__ import annotations

import os
import subprocess
import time

import modal

# coding_bench is not imported at module level: Modal imports this file inside
# the GPU container, whose image carries llama.cpp and nothing of ours.

HF_REPO = "meta-models/Muse-Glimmer-30B-GGUF"

# The dynamic K-quant, not the 17GB one. Meta puts its degradation at 0.2
# percent against 1.0 percent, and an H100 has room to spare, so there is no
# reason to take the lossier build.
MODEL_FILE = "muse-glimmer-30B-kquant-dynamic.gguf"
DRAFT_FILE = "dflash-kquant.gguf"

MODEL_ID = f"{HF_REPO}/{MODEL_FILE}"
APP_NAME = "coding-bench-muse-gguf-v1"

CACHE_DIR = "/cache"
PORT = 8080

# Chosen by measurement, see scripts/gpu_bench.py for the numbers. On cost per
# generated token this card beats the H100 by about 11 percent (0.0135 against
# 0.0152) despite being slower in absolute terms, and it loads the weights in 9
# seconds rather than 70. A100 and L40S are cheaper per hour but proportionally
# slower, so they cost slightly more per token. Bandwidth predicted none of
# this: the H200 has 1.4x the H100's and delivered 5 percent.
GPU = "RTX-PRO-6000"

# The real speedup. At batch size 1 the GPU spends most of its time waiting on
# weight reads, so serving several notes at once costs almost nothing per extra
# note. Slots share --ctx-size between them, hence the large total.
#
# Four slots rather than eight, because a slot has to hold the whole
# conversation. Eight was sized when this model had only ever seen gold
# candidates, where the longest prompt in the 578 note set is 8,228 tokens and
# 20,480 is generous. The full catalogue offers all 673 codes and measures
# 10,920 tokens on average with a longest of 19,130, which together with the
# 16,384 token generation budget needs 35,514 in the worst case. At 20,480 a
# long note would have overflowed its slot and come back truncated, which
# scores as an empty answer and still costs full price for the GPU time.
#
# Halving the slots to pay for it keeps TOTAL_CONTEXT where it was, so the KV
# cache is exactly the size already known to fit on this card. Throughput drops
# by less than the slot count suggests: at 11k tokens of prompt per note the
# server spends much of its time in prefill, which does not batch the way
# generation does.
PARALLEL_SLOTS = 4
CONTEXT_PER_SLOT = 40960
TOTAL_CONTEXT = PARALLEL_SLOTS * CONTEXT_PER_SLOT

# The upstream server image ships the binary in /app alongside its ggml shared
# objects, and puts neither on PATH nor on the loader path.
LLAMA_SERVER = "/app/llama-server"

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("muse-gguf-cache", create_if_missing=True)

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
    # Idle GPU time is pure cost. The container reloads in about 70 seconds,
    # which is cheap next to holding one open between runs.
    scaledown_window=90,
    # THIS MUST STAY 1. Concurrent callers would otherwise each get their own
    # container, and each container is a fresh GPU loading its own 20GB copy of
    # the weights. Eight parallel notes would mean eight GPUs rather than eight
    # slots on one, which is the opposite of the intended saving.
    max_containers=1,
)
# One container, several inputs in flight. The slots configured on llama-server
# are what actually serve them; this just lets Modal deliver more than one
# request at a time instead of serialising them at the door.
@modal.concurrent(max_inputs=PARALLEL_SLOTS)
class MuseGlimmerGGUF:
    """A llama.cpp server holding the quantised weights, reused across notes."""

    @modal.enter()
    def start_server(self):
        from huggingface_hub import hf_hub_download

        print(f"Fetching {MODEL_FILE}", flush=True)
        model_path = hf_hub_download(HF_REPO, MODEL_FILE, cache_dir=CACHE_DIR)
        draft_path = hf_hub_download(HF_REPO, DRAFT_FILE, cache_dir=CACHE_DIR)
        model_cache.commit()

        command = [
            LLAMA_SERVER,
            "--model", model_path,
            # Speculative decoding with the shipped DFlash drafter, which is
            # where the 3x comes from. Verified output is identical to running
            # the target model alone, so it costs nothing in quality.
            # NOTE: this does not currently work. llama.cpp logs "[spec] failed
            # to measure draft model memory: failed to create llama_context from
            # model" and falls back to ordinary decoding, which is why measured
            # throughput lands on Meta's ~75 tok/s baseline rather than the
            # ~233 tok/s they report with speculation. DFlash is a block
            # diffusion drafter that emits 16 tokens per forward pass, not the
            # autoregressive draft model this flag expects. Left in place, and
            # documented, so nobody assumes the 3x is being realised.
            "--spec-draft-model", draft_path,
            "--gpu-layers", "999",
            "--spec-draft-ngl", "999",
            # DFlash proposes blocks of 16 tokens, so draft that many per step.
            "--spec-draft-n-max", "16",
            "--ctx-size", str(TOTAL_CONTEXT),
            # Concurrent slots. Each note needs its prompt plus up to
            # 16k of generation, so the total is sized per slot.
            "--parallel", str(PARALLEL_SLOTS),
            # Batch the waiting requests together on each forward pass,
            # which is where the throughput actually comes from.
            "--cont-batching",
            "--port", str(PORT),
            "--host", "127.0.0.1",
            "--jinja",
        ]
        print(" ".join(command), flush=True)
        # Keep the server's own output. Without it, a startup failure is just an
        # exit code, and the reason it refused to load the model is exactly what
        # you need to see.
        self.server_log = open("/tmp/llama-server.log", "w+")
        self.server = subprocess.Popen(command, stdout=self.server_log, stderr=subprocess.STDOUT)
        try:
            self._await_ready()
        except Exception:
            # Tear the server down before propagating. A GPU container whose
            # startup raised is still a GPU container: one bad flag once left an
            # H100 running for over an hour after the run that spawned it had
            # already failed and exited.
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

    def _server_output(self, lines: int = 40) -> str:
        try:
            self.server_log.flush()
            with open("/tmp/llama-server.log") as handle:
                return "".join(handle.readlines()[-lines:])
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
    def diagnostics(self, lines: int = 60) -> str:
        """The server's own startup report, which says where the layers went.

        Throughput questions are almost always answered here: whether the model
        actually landed on the GPU, and whether the draft model loaded.
        """
        return self._server_output(lines)

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
                # Greedy when temperature is 0, so a rerun reproduces the run.
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
            "text": choice["message"].get("content") or "",
            "stop_reason": choice.get("finish_reason") or "unknown",
            "latency_s": elapsed,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }


class MuseGGUFClient:
    """Client side handle satisfying the LLMClient protocol."""

    # Named for what it is. Never report this as plain Muse Glimmer 30B.
    model_id = "meta-models/Muse-Glimmer-30B-GGUF:kquant-dynamic"

    def __init__(
        self,
        temperature: float = 0.0,
        reasoning_strength: str = "medium",
        app_name: str = APP_NAME,
    ):
        self.temperature = temperature
        self.reasoning_strength = reasoning_strength
        self._remote = modal.Cls.from_name(app_name, "MuseGlimmerGGUF")()

    def complete(self, system: str, user: str, max_tokens: int):
        from coding_bench.approaches.base import Completion, Truncated

        result = self._remote.complete.remote(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=self.temperature,
            reasoning_strength=self.reasoning_strength,
        )
        if result["stop_reason"] == "length":
            raise Truncated(
                self.model_id, "length", produced_tokens=result["usage"]["completion_tokens"]
            )
        return Completion(
            text=result["text"],
            stop_reason=result["stop_reason"],
            latency_s=result["latency_s"],
            usage=result["usage"],
        )


@app.local_entrypoint()
def smoke(prompt: str = "Count from 1 to 40, then reply with the JSON {\"ok\": true}."):
    """Check the server answers, and measure throughput honestly.

    Two requests, because the first one carries warmup and would understate the
    rate. The second is the number that predicts what a 578 note run costs.
    """
    model = MuseGlimmerGGUF()
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

    print("\n--- llama-server startup report ---")
    for line in model.diagnostics.remote().splitlines():
        if any(k in line.lower() for k in ("offload", "layer", "cuda", "device", "draft", "buffer")):
            print(line)
