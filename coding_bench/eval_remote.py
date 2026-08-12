"""Run the benchmark inside Modal, where the restricted data already is.

Tier 2 note text is never copied to a runner, a laptop, or a log. The gold
volume is mounted read only, its checksum is checked against the committed
manifest before anything is scored, and what comes back is aggregate metrics
plus predictions that carry code lists and note ids only.

    modal run coding_bench/eval_remote.py --task icd10 --model qwen3.6-27b --limit 25
    modal run coding_bench/eval_remote.py --task cpt --model muse-glimmer-30b --candidate-space full

A note on external APIs. What the code still does is record
which provider saw the text, in the run manifest, so any run can be audited
later without anyone having to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "coding_bench"

GOLD_MOUNT = "/gold"

app = modal.App("coding-bench-eval")

gold_volume = modal.Volume.from_name("coding-benchmark-gold", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=15.0.0",
        "numpy>=1.26",
        "groq>=0.11.0",
        "modal>=1.0.0",
        "sentence-transformers>=3.0.0",
    )
    .env({"CODING_BENCH_GOLD_ROOT": GOLD_MOUNT, "PYTHONPATH": "/root"})
    .add_local_dir(
        PACKAGE_DIR,
        remote_path="/root/coding_bench",
        ignore=modal.FilePatternMatcher(
            "**/gold/**",
            "**/*.parquet",
            "**/__pycache__/**",
            "**/results/**",
        ),
    )
)

# Which models send note text to an outside inference provider.
# Recorded per run rather than blocked, so
# that the provenance of any number is answerable after the fact.
EXTERNAL_PROVIDERS = {
    "qwen3.6-27b": "groq",
    "gpt-oss-120b": "groq",
}

# Which adapter file backs each model. Its hash goes into the run manifest and
# into the prediction cache key, so pointing at the wrong file would both
# misattribute a result and let an edited adapter serve stale cached answers.
ADAPTER_FILES = {
    "qwen3.6-27b": "api_groq.py",
    "gpt-oss-120b": "api_groq.py",
    "muse-glimmer-30b-gguf": "modal_muse_gguf.py",
    "muse-glimmer-30b": "modal_muse.py",
}


def local_git_state() -> tuple[str, bool]:
    """The launching machine's commit, resolved without importing the package.

    The local entrypoint runs in whatever environment the modal CLI lives in,
    which is not the benchmark's environment and generally has neither numpy nor
    pyarrow. Importing `runner` here to reuse its git helper looks tidy and
    breaks every run with a ModuleNotFoundError.
    """
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def build_predictor(
    approach: str,
    model: str,
    code_system: str,
    max_tokens: int = 16384,
    reasoning_strength: str = "medium",
):
    """Assemble an approach from its parts."""
    from coding_bench.adapters.api_groq import MODELS as GROQ_MODELS, GroqClient
    from coding_bench.approaches.llm import LLMPredictor
    from coding_bench.approaches.retrieval import RetrievalPredictor
    from coding_bench.approaches.retr_llm import RetrievalThenLLM

    if approach == "retrieval":
        return RetrievalPredictor(top_k=10)

    if model in GROQ_MODELS:
        client = GroqClient(model_id=GROQ_MODELS[model])
    elif model == "muse-glimmer-30b-gguf":
        from coding_bench.adapters.modal_muse_gguf import MuseGGUFClient

        # Quantised weights, named as such in the run manifest so this can never
        # be tabulated next to a bf16 number as though they were one model.
        client = MuseGGUFClient(reasoning_strength=reasoning_strength)
    elif model == "muse-glimmer-30b":
        from coding_bench.adapters.modal_muse import MuseClient

        # Reasoning strength drives generation length, and generation length is
        # the entire wall clock of a self hosted 30B on one GPU. At "high" this
        # model runs its trace out to the token cap on clinical notes, which is
        # minutes per note. It is a recorded run parameter, not a hidden default.
        client = MuseClient(reasoning_strength=reasoning_strength)
    else:
        raise ValueError(
            f"Unknown model {model!r}. Known: "
            f"{sorted(GROQ_MODELS) + ['muse-glimmer-30b-gguf', 'muse-glimmer-30b']}"
        )

    llm = LLMPredictor(client=client, code_system=code_system, max_tokens=max_tokens)
    if approach == "llm":
        return llm
    if approach == "retr_llm":
        return RetrievalThenLLM(retriever=RetrievalPredictor(), llm=llm, shortlist=50)
    raise ValueError(f"Unknown approach {approach!r}")


@app.function(
    image=image,
    volumes={GOLD_MOUNT: gold_volume},
    secrets=[modal.Secret.from_name("groq-api")],
    timeout=14400,
    memory=8192,
    cpu=4.0,
)
def evaluate(
    task: str = "icd10",
    approach: str = "llm",
    model: str = "qwen3.6-27b",
    candidate_space: str = "gold",
    limit: int | None = None,
    max_tokens: int = 16384,
    concurrency: int = 1,
    reasoning_strength: str = "medium",
    cache: str = "auto",
    git_sha: str | None = None,
) -> dict:
    """Score one approach against one task. Returns a text free run record."""
    from coding_bench.bench import loaders, runner

    provider = EXTERNAL_PROVIDERS.get(model)
    if provider and task in ("cpt", "icd10"):
        print(
            f"Note text for {task} will be sent to {provider}, which is cleared for "
            f"MIMIC derived data and is recorded in the run manifest.",
            flush=True,
        )

    if task in ("cpt", "icd10"):
        stamp = loaders.verify_volume(Path(GOLD_MOUNT))
        print(f"Volume verified against the committed manifests: {stamp}", flush=True)

    dataset = loaders.load(task)
    print(f"Loaded {len(dataset)} notes for {task} ({dataset.code_system})", flush=True)

    space: int | str | None = candidate_space
    if candidate_space == "full":
        space = None
    elif candidate_space.isdigit():
        space = int(candidate_space)

    predictor = build_predictor(
        approach, model, dataset.code_system, max_tokens, reasoning_strength
    )
    adapter_path = Path("/root/coding_bench/adapters") / ADAPTER_FILES.get(model, "api_groq.py")
    prompt_text = None
    if hasattr(predictor, "build_prompt"):
        prompt_text = predictor.build_prompt(dataset.notes[0], [])[0]

    record = runner.run(
        predictor,
        dataset,
        candidate_space=space,
        limit=limit,
        parameters={"max_tokens": max_tokens, "approach": approach, "concurrency": concurrency,
                    "reasoning_strength": reasoning_strength},
        adapter_path=adapter_path,
        prompt_text=prompt_text,
        external_provider=provider,
        # The retriever caches encoded candidates on the instance, so it is not
        # safe to share across threads.
        concurrency=1 if approach in ("retrieval", "retr_llm") else concurrency,
        cache=cache,
        git_sha=git_sha,
    )
    print(runner.summarise(record), flush=True)
    return record


@app.local_entrypoint()
def main(
    task: str = "icd10",
    approach: str = "llm",
    model: str = "qwen3.6-27b",
    candidate_space: str = "gold",
    limit: int = 0,
    max_tokens: int = 16384,
    concurrency: int = 1,
    reasoning_strength: str = "medium",
    cache: str = "auto",
):
    sha, dirty = local_git_state()
    if dirty:
        print("Working tree is dirty; this run will not be exactly reproducible.")
    record = evaluate.remote(
        task=task,
        approach=approach,
        model=model,
        candidate_space=candidate_space,
        limit=limit or None,
        max_tokens=max_tokens,
        concurrency=concurrency,
        reasoning_strength=reasoning_strength,
        cache=cache,
        git_sha=sha,
    )

    runs_dir = PACKAGE_DIR / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record['manifest']['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
    print(f"\nRun record written to {path.relative_to(REPO_ROOT)}")
