"""Measure a model's real throughput on each GPU before choosing one.

Written because specification reasoning kept being wrong. The H200 has 1.4x the
memory bandwidth of an H100 and delivered 5 percent. Meta measured 74.9 tok/s on
a consumer RTX 5090, and an H100 at nearly twice the bandwidth gave 72.3. Three
cards spanning 2.7x in bandwidth all land within a few percent of each other,
which means something other than bandwidth sets the ceiling and the only honest
way to pick hardware is to run it.

Measured results, 2026-08-12, Muse Glimmer 30B kquant-dynamic:

    GPU            decode tok/s   $/hr    $/1k tokens   load s
    RTX PRO 6000           62.3   3.03         0.0135        9
    H100                   72.3   3.95         0.0152       70
    A100-80GB              44.0   2.50         0.0158        -
    L40S                   33.5   1.95         0.0162        -

RTX PRO 6000 wins on cost per token, by about 11 percent over the H100, and
loads the weights roughly eight times faster. It runs fine on the prebuilt
image, so the sm_120 concern was unfounded. The cheaper Ampere and Ada cards do
not win: they are slower by almost exactly the amount they are cheaper, so cost
per token is flat across H100, A100 and L40S within seven percent.

Bandwidth does not explain the ordering. An RTX 5090 at 1.8 TB/s is reported by
Meta at 74.9 tok/s while the A100 at 2.0 TB/s manages 44, and the H200 at 4.8
gives only 5 percent over the H100 at 3.35. Architecture and kernel quality for
4 bit weights dominate: Blackwell and Hopper do well, Ampere and Ada do not.

Prefill and decode are reported separately because they behave differently.
Decode is what the bandwidth argument is about; prefill is compute bound and is
where a cheaper card can genuinely lose, which matters for full catalogue
prompts of eleven thousand tokens but not for gold candidate prompts of one.

    modal run coding_bench/scripts/gpu_bench.py --gpus "H100,A100-80GB,L40S"

Cost is a few cents per card: one container start, one model load, three
generations. Run it once when a new card appears rather than arguing from
datasheets.
"""

from __future__ import annotations

import json
import subprocess
import time

import modal

HF_REPO = "meta-models/Muse-Glimmer-30B-GGUF"
MODEL_FILE = "muse-glimmer-30B-kquant-dynamic.gguf"
CACHE_DIR = "/cache"
PORT = 8080
LLAMA_SERVER = "/app/llama-server"

app = modal.App("coding-bench-gpu-bench")
model_cache = modal.Volume.from_name("muse-gguf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
    .pip_install("huggingface_hub>=0.30.0", "requests")
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "LD_LIBRARY_PATH": "/app",
            "PATH": "/app:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin",
        }
    )
)


def _measure(gpu_name: str) -> dict:
    """Start the server, run a short and a long prompt, report both phases."""
    import requests
    from huggingface_hub import hf_hub_download

    started_load = time.perf_counter()
    model_path = hf_hub_download(HF_REPO, MODEL_FILE, cache_dir=CACHE_DIR)

    log = open("/tmp/bench-server.log", "w+")
    server = subprocess.Popen(
        [
            LLAMA_SERVER, "--model", model_path,
            "--gpu-layers", "999", "--ctx-size", "32768",
            "--port", str(PORT), "--host", "127.0.0.1", "--jinja",
        ],
        stdout=log, stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if server.poll() is not None:
                log.flush()
                with open("/tmp/bench-server.log") as handle:
                    tail = "".join(handle.readlines()[-25:])
                return {"gpu": gpu_name, "error": f"server exited: {tail}"}
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        else:
            return {"gpu": gpu_name, "error": "server never became ready"}

        load_s = time.perf_counter() - started_load

        def generate(prompt: str, max_tokens: int) -> dict:
            response = requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=600,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage", {})
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

        # Discard the first generation: it carries graph capture and KV setup and
        # reports roughly a tenth of the real rate.
        generate("Say hi.", 32)

        short = "Count from 1 to 60, one number per line."
        t0 = time.perf_counter()
        short_usage = generate(short, 512)
        short_s = time.perf_counter() - t0

        # A long prompt to expose prefill, which is where a cheaper card with less
        # compute can lose even when its decode rate matches.
        long_prompt = ("Summarise this list in one word.\n" + "item alpha beta gamma delta\n" * 700)
        t0 = time.perf_counter()
        long_usage = generate(long_prompt, 128)
        long_s = time.perf_counter() - t0

        return {
            "gpu": gpu_name,
            "load_s": round(load_s),
            "decode_tok_s": round(short_usage["completion_tokens"] / max(short_s, 1e-3), 1),
            "long_prompt_tokens": long_usage["prompt_tokens"],
            "long_total_s": round(long_s, 1),
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


# One function per card. Modal fixes the GPU at decoration time, so a card is a
# separate function rather than a parameter.
@app.function(image=image, gpu="H100", volumes={CACHE_DIR: model_cache}, timeout=1800)
def bench_h100() -> dict:
    return _measure("H100")


@app.function(image=image, gpu="A100-80GB", volumes={CACHE_DIR: model_cache}, timeout=1800)
def bench_a100() -> dict:
    return _measure("A100-80GB")


@app.function(image=image, gpu="L40S", volumes={CACHE_DIR: model_cache}, timeout=1800)
def bench_l40s() -> dict:
    return _measure("L40S")


@app.function(image=image, gpu="RTX-PRO-6000", volumes={CACHE_DIR: model_cache}, timeout=1800)
def bench_rtx_pro_6000() -> dict:
    return _measure("RTX-PRO-6000")


# Dollars per hour, from Modal's published per second pricing. Used to turn a
# throughput number into the only comparison that matters, cost per note.
PRICE_PER_HOUR = {
    "H100": 3.95,
    "A100-80GB": 2.50,
    "L40S": 1.95,
    "RTX-PRO-6000": 3.03,
}

BENCHMARKS = {
    "H100": bench_h100,
    "A100-80GB": bench_a100,
    "L40S": bench_l40s,
    "RTX-PRO-6000": bench_rtx_pro_6000,
}


@app.local_entrypoint()
def main(gpus: str = "H100,A100-80GB"):
    """Run one card at a time, so only one GPU is ever billed."""
    results = []
    for name in [g.strip() for g in gpus.split(",") if g.strip()]:
        if name not in BENCHMARKS:
            print(f"Unknown GPU {name!r}, known: {sorted(BENCHMARKS)}")
            continue
        print(f"\nMeasuring {name}...")
        result = BENCHMARKS[name].remote()
        price = PRICE_PER_HOUR.get(name)
        if price and not result.get("error") and result.get("decode_tok_s"):
            # Cost of a thousand generated tokens, which is roughly one note.
            result["usd_per_1k_tokens"] = round(
                price / 3600 * (1000 / result["decode_tok_s"]), 4
            )
        results.append(result)
        print(json.dumps(result, indent=2))

    print("\n=== summary ===")
    print(f"{'GPU':<16} {'decode tok/s':>13} {'$/hr':>7} {'$/1k tok':>10}")
    for result in results:
        if result.get("error"):
            print(f"{result['gpu']:<16} {'FAILED':>13}")
            continue
        print(
            f"{result['gpu']:<16} {result['decode_tok_s']:>13} "
            f"{PRICE_PER_HOUR.get(result['gpu'], 0):>7.2f} "
            f"{result.get('usd_per_1k_tokens', 0):>10.4f}"
        )
