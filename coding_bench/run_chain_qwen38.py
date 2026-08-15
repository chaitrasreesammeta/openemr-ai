"""Qwen3.8 27B FP8, self hosted, on all four conditions the board compares on.

One model, four runs, matching what every hosted row on the board already has:
cpt and icd10, gold and full candidates. Nothing here is an ablation. The point
is a row that can be read next to the others without a footnote.

    modal deploy coding_bench/adapters/modal_qwen38_vllm.py
    modal run   coding_bench/adapters/modal_qwen38_vllm.py::smoke
    modal deploy coding_bench/run_chain_qwen38.py
    modal run   coding_bench/run_chain_qwen38.py::launch

Smoke first. A chain that discovers on step one that vLLM will not serve this
architecture has already paid for a GPU to find out, and the smoke entrypoint
also prints whether an fp8 kernel actually loaded.

## What "comparable" is made of, since that is the whole design

Four things are held to what the rest of the board used, and each one is a place
a number could quietly stop being comparable.

**The build is FP8.** `Qwen/Qwen3.8-27B-FP8`, not the bf16 release. Every self
hosted row on the board is a deployment build, `full` candidates is the
deployment condition, and the hosted providers do not publish their serving
precision, so a bf16 row would be the only number on the board measured at a
precision nobody serves. The reasoning is at the top of
`adapters/modal_qwen38_vllm.py`; the bf16 arm is registered there and is not
queued here.

**Thinking is on.** `reasoning_strength: "medium"`, the package default, which
on this adapter sets `enable_thinking: true`. This is the equivalence that
matters most and the easiest one to get wrong. The nearest comparable row is
Qwen3.6 27B, same family and same size, and Groq serves it as a reasoning model.
An instruct mode Qwen3.8 measured against it would report a decoder difference
as a model difference, and it would flatter the newer model on latency while
doing it.

**max_tokens is 16,384.** What the board's ICD-10 and CPT runs used. It is in
the prediction cache key, so a different value would not just be incomparable,
it would miss every cached note and pay again.

**The card is the RTX PRO 6000**, which is what Muse Glimmer and Gemma 4 ran on,
at four concurrent notes, which is what they used. Latency between self hosted
rows is only a comparison if the hardware is the same one.

## Order, and what an interruption costs

Full catalogue before gold candidates. `full` is the deployment condition and
gold is a recall ceiling, so an interruption should cost the ceiling and not the
answer. ICD-10 before CPT because it is the larger and more discriminating set.

Every note is cached as it lands, so an interruption costs the notes in flight
rather than the run, and relaunching resumes.

## Expect a long afternoon

578 and 312 note sets, at roughly 11k prompt tokens each for the full catalogue,
four sequences in flight on one card, against a model that is thinking before it
answers. Three to six hours is the realistic range.

## The failure to watch for is truncation, not a low score

Greedy decoding in thinking mode is the configuration Qwen warns about, because
a trace can loop until it hits the cap. That arrives here as a length stop, which
the adapter raises as `Truncated` and the runner counts separately, and a run
over 5 percent failures is quarantined rather than scored. Muse ran the same way
and landed at 4.0 percent on ICD-10 full, just inside the ceiling. If this
quarantines on truncation, raise `max_tokens` and rerun that step, and record
that the step used a different value from the rest of the board.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from coding_bench.eval_remote import image as eval_image

RESULTS_MOUNT = "/results"
GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-chain-qwen38")

results_volume = modal.Volume.from_name("coding-benchmark-results", create_if_missing=True)
gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

# Four in flight, matching MAX_NUM_SEQS on the server and what the other self
# hosted runs used. Asking for more would queue at the door rather than widen
# the batch; asking for fewer would leave the card idle between notes.
CONCURRENCY = 4

CHAIN: list[dict] = [
    # Deployment condition first, larger task first.
    {"task": "icd10", "model": "qwen3.8-27b-fp8", "candidate_space": "full"},
    {"task": "cpt", "model": "qwen3.8-27b-fp8", "candidate_space": "full"},
    # The recall ceilings, which complete the set every other model has.
    {"task": "icd10", "model": "qwen3.8-27b-fp8", "candidate_space": "gold"},
    {"task": "cpt", "model": "qwen3.8-27b-fp8", "candidate_space": "gold"},
]

DEFAULTS = {
    "approach": "llm",
    # What the board's runs used. Also in the cache key.
    "max_tokens": 16384,
    # Thinking on, matching the Qwen3.6 row. See the docstring.
    "reasoning_strength": "medium",
    "cache": "auto",
    "recompute_empty": False,
    "concurrency": CONCURRENCY,
}


@app.function(
    image=eval_image,
    volumes={RESULTS_MOUNT: results_volume, GOLD_MOUNT: gold_volume},
    # No note leaves Modal on this chain, but the app declares the same secret
    # the others do because Modal resolves every secret in an app at startup and
    # an app that declares none here would drift from the rest.
    secrets=[modal.Secret.from_name("groq-api")],
    # A day. Each step commits its own record, so hitting this ceiling costs one
    # step rather than the queue.
    timeout=86400,
    cpu=4.0,
    memory=8192,
)
def run_qwen38_chain(chain: list[dict] | None = None, git_sha: str | None = None) -> list[dict]:
    from coding_bench.run_chain import _evaluate_inline

    queue = chain if chain is not None else CHAIN
    summaries = []

    for index, step in enumerate(queue, start=1):
        config = {**DEFAULTS, **step, "git_sha": git_sha}
        label = f"{config['task']}/{config['model']}/{config['candidate_space']}"
        print(f"\n[{index}/{len(queue)}] starting {label}", flush=True)

        try:
            record = _evaluate_inline(config)
        except Exception as exc:  # noqa: BLE001
            # One failing condition must not abandon the rest. Three of four
            # conditions is still a usable row; nothing banked is not.
            print(f"[{index}/{len(queue)}] FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"step": label, "status": "failed", "error": str(exc)[:300]})
            continue

        manifest = record["manifest"]
        metrics = record["metrics"]
        ops = metrics["operational"]
        error_rate = ops.get("error_rate", 0.0)

        out_path = Path(RESULTS_MOUNT) / f"{manifest['run_id']}.json"
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
        results_volume.commit()

        summaries.append(
            {
                "step": label,
                "status": "ok",
                "run_id": manifest["run_id"],
                "n_notes": manifest["n_notes"],
                "micro_f1": round(metrics["core"]["micro_f1"], 4),
                # The band that decides whether this model is usable, printed in
                # the summary so it is not something you have to go and look up.
                "tail_f1": round(metrics["bands"].get("tail", {}).get("micro_f1", 0.0), 4),
                "error_rate": round(error_rate, 4),
                "valid": error_rate <= 0.05,
                # The number to read first if `valid` is false.
                "truncation_rate": round(ops.get("truncation_rate", 0.0), 4),
                "latency_mean_s": round(ops.get("latency_mean_s", 0.0), 2),
                "cache_hits": ops.get("cache_hits", 0),
            }
        )
        print(f"[{index}/{len(queue)}] done {label}: {json.dumps(summaries[-1])}", flush=True)

    print("\n=== qwen3.8 chain complete ===", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(
        "\nBank the records and rebuild the board with:\n"
        "  modal run coding_bench/run_chain.py::fetch\n"
        "  python -m coding_bench.bench.reporting",
        flush=True,
    )
    return summaries


def _spawn() -> modal.FunctionCall:
    """Create the call server side and hand back a handle to it.

    Deliberately a spawn against the *deployed* function rather than a call on
    the local object. A spawn is created inside the deployed app and belongs to
    it, so there is no client whose death can cancel it. `modal run --detach`
    is not the same thing and does not survive: killing the launcher produced
    "Received a cancellation signal" and lost a step at note 100 of 312.
    """
    from coding_bench.eval_remote import local_git_state

    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; these runs will not be exactly reproducible.")

    print(f"Queueing {len(CHAIN)} runs:")
    for step in CHAIN:
        print(f"  {step['task']:6} {step['model']:20} {step['candidate_space']}")

    fn = modal.Function.from_name(app.name, "run_qwen38_chain")
    call = fn.spawn(git_sha=sha)
    print(f"\nSpawned as call {call.object_id}")
    return call


@app.local_entrypoint()
def launch():
    """Spawn the queue and return, leaving nothing on this machine holding it up."""
    _spawn()
    print("Expect three to six hours. Watch or collect with:")
    print(f"  modal app logs {app.name}")
    print("  modal run coding_bench/run_chain.py::fetch")


@app.local_entrypoint()
def run_now(timeout_s: int = 18000):
    """Spawn the queue and then watch it, for a caller that wants to bank it.

    This is what CI runs on the commit that adds the model, so the row lands on
    the board without anybody remembering to come back and collect it.

    **Watching is not driving.** The call was created server side by `_spawn`
    and belongs to the deployed app, so this process is only holding a handle to
    it. If the runner is cancelled, the job hits its own ceiling, or the network
    goes, the chain carries on and every finished step is already committed to
    the results volume. The worst case is that nobody has banked it yet, which a
    dispatch of `action: collect` fixes at any point afterwards.

    That is also why the timeout here is not an error. Five hours is under both
    the six hour ceiling a GitHub hosted runner has and the estimate for this
    queue, so a slow run is expected to hit it sometimes. It returns rather than
    raising, so the collect step that follows still banks whatever finished.
    """
    from modal.exception import OutputExpiredError

    call = _spawn()
    print(f"Watching for up to {timeout_s / 3600:.1f}h. The run does not depend on this process.")

    try:
        summaries = call.get(timeout=timeout_s)
    # The builtin TimeoutError, deliberately. Modal exports a TimeoutError of
    # its own from modal.exception, it is what several of its errors derive
    # from, and it is **not** a subclass of the builtin. `FunctionCall.get`
    # raises the builtin on wait expiry, so catching modal's would look more
    # correct and would never fire, turning every slow run into a failed job.
    except (TimeoutError, OutputExpiredError) as exc:
        still_running = isinstance(exc, TimeoutError)
        print(
            f"\n{'Still running after' if still_running else 'Lost the result handle after'} "
            f"{timeout_s / 3600:.1f}h, which is not a failure of the run.\n"
            f"  modal app logs {app.name}\n"
            "  modal run coding_bench/run_chain.py::fetch   # banks the finished steps\n"
            "Every finished step is already on the results volume, so nothing is lost "
            "and a later dispatch of `action: collect` picks up the rest."
        )
        return

    print(json.dumps(summaries, indent=2))
    failed = [s for s in summaries if s.get("status") != "ok"]
    quarantined = [s for s in summaries if s.get("status") == "ok" and not s.get("valid")]
    if failed:
        print(f"\n{len(failed)} step(s) failed: {[s['step'] for s in failed]}")
    if quarantined:
        # Not raised. A quarantined run is a result about the configuration and
        # the board has a section for it, so it must still be banked.
        print(
            f"\n{len(quarantined)} step(s) exceeded the 5% failure ceiling and will be "
            f"quarantined rather than scored: {[s['step'] for s in quarantined]}\n"
            "Check truncation_rate first: greedy decoding in thinking mode is the "
            "configuration this is expected to fail in."
        )
