"""Is a configuration already banked, at this exact code and these parameters?

The prediction cache answers "has this note been paid for" one note at a time,
after a container has started, a dataset has loaded and a run has begun. It is
the right mechanism when something genuinely changed. It is the wrong one when
nothing did, because arriving at the answer still costs a container start per
configuration, and any note the cache misses is re-inferred at full price.

That gap has a measured cost. On 2026-09-07 a push whose only real work was four
new Qwen3.8 cells also walked twelve settled ones and re-inferred 383 notes
across them, 139 on qwen3.6 and 244 on gpt-oss, for $1.90 on top of the run it
was actually there to do. Those cells had valid committed records the whole
time, under an identical adapter hash and identical parameters.

So this asks the cheap question first, before anything is deployed or loaded: is
there already a committed record for this configuration, produced by the code
that is checked out now, that passed the failure ceiling? If there is, the run
set can skip the cell entirely and read the number off the record.

What the check covers, and what it leans on. Directly: the model, the task, the
candidate space, the adapter file's current hash, the token budget and the
reasoning strength, all of which are in the prediction cache key and any of
which changing means the answer could differ. Indirectly: the prompt and the
gold set, which are also in that key but cannot be hashed without the restricted
data. Those are covered by the workflow's own guard, which refuses a push that
touches `approaches/llm.py` or `data/manifests/` and asks for an explicit
`confirm_full_rerun`. If that guard is ever removed, this check becomes unsafe
and must grow a prompt hash comparison.

A quarantined record does not count as settled. A cell that failed its way past
the ceiling has no usable number, so it is worth trying again.

    python -m coding_bench.bench.settled --task icd10 --model qwen3.6-27b \
        --space gold --max-tokens 16384

Exit 0 means settled, so skip it. Exit 1 means it has to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_bench.bench.runner import ERROR_RATE_INVALIDATES, RUNS_DIR, file_sha256

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"


def model_id_for(model: str) -> str:
    """The id a run record carries for a `--model` name.

    Imported lazily and per family, the way build_predictor does it, so that a
    missing optional SDK cannot stop the question being asked.
    """
    from coding_bench.adapters.api_groq import MODELS as GROQ_MODELS

    if model in GROQ_MODELS:
        return GROQ_MODELS[model]

    from coding_bench.adapters.api_groq_qwen38 import MODELS as QWEN38_MODELS

    if model in QWEN38_MODELS:
        return QWEN38_MODELS[model]

    from coding_bench.adapters.api_anthropic import MODELS as ANTHROPIC_MODELS

    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model]

    self_hosted = {
        "muse-glimmer-30b-gguf": "coding_bench.adapters.modal_muse_gguf",
        "muse-glimmer-30b": "coding_bench.adapters.modal_muse",
        "gemma4-26b-a4b-gguf": "coding_bench.adapters.modal_gemma4_gguf",
    }
    if model in self_hosted:
        import importlib

        return importlib.import_module(self_hosted[model]).MODEL_ID

    raise ValueError(f"Unknown model {model!r}")


def settled_record(
    task: str,
    model: str,
    space: str,
    max_tokens: int,
    reasoning_strength: str = "medium",
    runs_dir: Path = RUNS_DIR,
) -> dict | None:
    """The committed record that makes this configuration settled, if any."""
    from coding_bench.eval_remote import ADAPTER_FILES

    model_id = model_id_for(model)
    adapter_hash = file_sha256(ADAPTER_DIR / ADAPTER_FILES[model])

    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf8"))
        except (json.JSONDecodeError, OSError):
            continue
        m = record.get("manifest", {})
        if m.get("model_id") != model_id or m.get("task") != task:
            continue
        if str(m.get("candidate_space")) != space:
            continue
        # The adapter hash is the whole point: an edited adapter is a different
        # model as far as any answer it produced is concerned.
        if m.get("adapter_sha256") != adapter_hash:
            continue
        params = m.get("parameters") or {}
        if params.get("max_tokens") != max_tokens:
            continue
        if params.get("reasoning_strength", "medium") != reasoning_strength:
            continue
        error_rate = record.get("metrics", {}).get("operational", {}).get("error_rate", 0.0)
        if error_rate > ERROR_RATE_INVALIDATES:
            continue
        return record
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--reasoning-strength", default="medium")
    args = ap.parse_args(argv)

    record = settled_record(
        args.task, args.model, args.space, args.max_tokens, args.reasoning_strength
    )
    if record is None:
        print(f"{args.task} {args.model} {args.space}: not banked at this code, running it")
        return 1

    m, op = record["manifest"], record["metrics"]["operational"]
    print(
        f"{args.task} {args.model} {args.space}: settled by {m['run_id']} "
        f"(n={m['n_notes']}, errors {op.get('error_rate', 0.0):.1%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
