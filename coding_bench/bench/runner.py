"""The evaluation loop and the run record writer.

Two rules are enforced here rather than remembered:

  1. A per note output is {note_id, gold, pred, latency} and nothing else. The
     writer raises on a text key. Public Actions logs are public, and a run
     record that carried note text would be a disclosure.
  2. Every run records what produced it: model, adapter hash, prompt hash,
     dataset manifest checksum, parameters, and git SHA. A number nobody can
     reproduce does not belong on the leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from coding_bench.approaches.base import Candidate, Note, Predictor, Truncated
from coding_bench.bench import cache as cache_module
from coding_bench.bench import metrics as m
from coding_bench.bench.loaders import Dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

# Candidate space sizes for the scaling stress. None means the full catalogue.
SCALING_SIZES = ("gold", 200, 1000, None)

# Keys that must never appear in a per note record.
FORBIDDEN_KEYS = {"text", "note_text", "note", "covered_text", "raw", "prompt", "response"}

# Recorded in the run manifest for reproducibility, but deliberately excluded
# from the cache key: they affect how a run executes, never what it answers.
NON_SEMANTIC_PARAMETERS = {"concurrency", "recompute_empty"}


class ResultWriterError(RuntimeError):
    """Raised when a record would leak something it must not carry."""


@dataclass
class RunManifest:
    """Everything needed to reproduce one number."""

    run_id: str
    task: str
    code_system: str
    tier: int
    approach: str
    approach_version: str
    model_id: str
    candidate_space: str
    n_notes: int
    dataset_manifest_sha256: str | None
    adapter_sha256: str | None
    prompt_sha256: str | None
    git_sha: str
    git_dirty: bool
    # The hosted provider that saw the note text, or None when the run stayed
    # inside Modal. Recorded so the provenance of a Tier 2 number is answerable
    # long after the run.
    external_provider: str | None = None
    # The approach module that parsed the responses, hashed. Deliberately NOT
    # part of the cache key: a parser change cannot alter a prediction that is
    # already stored, only how a fresh response would be read, so keying on it
    # would throw away every paid answer in the workspace to fix a handful of
    # them. Recorded here instead, so "which parser produced this number" is
    # answerable without a rerun, and `recompute_empty` regenerates the notes
    # that a parser fix could actually change.
    approach_sha256: str | None = None
    parameters: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    python: str = field(default_factory=platform.python_version)


def git_state() -> tuple[str, bool]:
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


def approach_file(predictor) -> str | None:
    """The source file of the approach, found through the object rather than named.

    Naming it would mean a second registry to keep in step with the first, and
    the failure mode of a stale one is a manifest that points at the wrong code.
    """
    import sys

    module = sys.modules.get(type(predictor).__module__)
    return getattr(module, "__file__", None)


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_candidates(
    note: Note, label_space: dict[str, str | None], size: int | str | None, seed: int = 0
) -> list[Candidate]:
    """The codes offered to the model for one note.

    "gold" offers only the right answers, which is the ceiling condition. An
    integer pads the gold codes with deterministically chosen distractors up to
    that size. None offers the whole catalogue, which is the deployment
    condition. Distractor choice is seeded by note id so the same note gets the
    same distractors on every rerun and across models.
    """
    gold = list(note.gold_codes)
    if size == "gold":
        chosen = gold
    elif size is None:
        chosen = list(label_space)
    else:
        pool = sorted(set(label_space) - set(gold))
        rng = random.Random(f"{seed}:{note.note_id}")
        n_distractors = max(0, int(size) - len(gold))
        chosen = gold + rng.sample(pool, min(n_distractors, len(pool)))

    rng = random.Random(f"order:{seed}:{note.note_id}")
    ordered = sorted(set(chosen))
    rng.shuffle(ordered)
    return [Candidate(code=code, description=label_space.get(code)) for code in ordered]


_ACCOUNT_ID = re.compile(r"\b(org|acct|user)_[A-Za-z0-9]{8,}\b")


def sanitise_error(message: str | None, limit: int = 200) -> str | None:
    """Trim a provider error down to what a reader of the leaderboard needs.

    Provider errors are quoted verbatim into run records, which are committed,
    so they must not carry account identifiers or grow long enough to hide
    something that came back from the request.
    """
    if not message:
        return None
    return _ACCOUNT_ID.sub("<account>", message)[:limit]


def _rebuild(note: Note, offered: set[str], cached: dict) -> tuple[m.NoteResult, dict]:
    """Turn a cache hit back into the pair a fresh call would have produced."""
    codes = {
        code: (tuple(span) if span else None) for code, span in (cached.get("codes") or {}).items()
    }
    truncated = bool(cached.get("truncated"))
    latency = float(cached.get("latency_s") or 0.0)
    usage = dict(cached.get("usage") or {})
    salvaged = bool(cached.get("salvaged"))

    result = m.NoteResult(
        note_id=note.note_id,
        gold=list(note.gold_codes),
        pred=list(codes),
        pred_spans={code: span for code, span in codes.items() if span is not None},
        gold_spans=note.gold_spans,
        latency_s=latency,
        truncated=truncated,
        candidates=offered,
    )
    record = check_record(
        {
            "note_id": note.note_id,
            "gold": sorted(note.gold_codes),
            "pred": sorted(codes),
            "pred_spans": {code: list(span) for code, span in codes.items() if span},
            "latency_s": round(latency, 3),
            "truncated": truncated,
            "n_candidates": len(offered),
            "usage": usage,
            "error": None,
            "cached": True,
            "salvaged": salvaged,
        }
    )
    return result, record


def check_record(record: dict) -> dict:
    """Refuse to write a per note record that carries note text."""
    offending = sorted(FORBIDDEN_KEYS & set(record))
    if offending:
        raise ResultWriterError(
            f"Per note records may not carry {offending}. Records are "
            f"{{note_id, gold, pred, latency}} and nothing else, because run "
            f"records are committed to a public repo."
        )
    for key, value in record.items():
        if isinstance(value, str) and len(value) > 500:
            raise ResultWriterError(
                f"Field {key!r} is {len(value)} characters, which is note shaped. "
                f"Per note records carry codes and numbers, not prose."
            )
    return record


def run(
    predictor: Predictor,
    dataset: Dataset,
    candidate_space: int | str | None = "gold",
    limit: int | None = None,
    seed: int = 0,
    parameters: dict | None = None,
    adapter_path: Path | str | None = None,
    prompt_text: str | None = None,
    external_provider: str | None = None,
    concurrency: int = 1,
    cache: str = "auto",
    git_sha: str | None = None,
    progress: bool = True,
    recompute_empty: bool = False,
) -> dict:
    """Evaluate one approach over one dataset and return the full run record.

    concurrency > 1 runs notes through a thread pool. For an API backed model
    almost all of the wall clock is waiting on a socket, so this is the
    difference between an evaluation that takes minutes and one that takes
    hours. Results are reassembled in dataset order regardless, so the run
    record does not depend on which request finished first. Leave it at 1 for
    approaches that hold mutable state across notes, such as the retriever with
    its encoded candidate cache.
    """
    notes = dataset.notes[:limit] if limit else dataset.notes
    started = datetime.now(timezone.utc)

    adapter_hash = file_sha256(adapter_path)
    prompt_hash = text_sha256(prompt_text)
    # Only parameters that can change an answer belong in the cache key.
    # Concurrency changes how fast the run goes and nothing else, so including
    # it would throw the whole cache away every time the rate limit forces a
    # different worker count, which is precisely when reuse matters most.
    cache_parameters = {
        key: value for key, value in (parameters or {}).items() if key not in NON_SEMANTIC_PARAMETERS
    }

    # Rebuild the candidate set a past run offered a note, so its committed
    # prediction can be keyed identically. Only reconstructible when that run
    # used the same candidate space and seed, which the manifest records.
    def candidates_for_seed(past_manifest: dict, row: dict) -> set[str] | None:
        if past_manifest.get("candidate_space") != space_label:
            return None
        note = notes_by_id.get(row["note_id"])
        if note is None:
            return None
        offered = {c.code for c in build_candidates(note, dataset.label_space, candidate_space, seed)}
        # The record stores how many codes were offered. If that disagrees, the
        # past run asked a different question and its answer is not reusable.
        return offered if len(offered) == row.get("n_candidates") else None

    notes_by_id = {note.note_id: note for note in dataset.notes}
    space_label = "full" if candidate_space is None else str(candidate_space)
    store = cache_module.open_cache(
        cache,
        seed=(
            cache_module.seed_from_records(RUNS_DIR, manifest_lookup=candidates_for_seed)
            if cache != "off"
            else None
        ),
    )

    def key_for(note: Note, offered: set[str]) -> str:
        return cache_module.cache_key(
            dataset_manifest_sha256=dataset.manifest_sha256,
            note_id=note.note_id,
            model_id=getattr(predictor, "model_id", "n/a"),
            approach=getattr(predictor, "name", type(predictor).__name__),
            approach_version=getattr(predictor, "version", "0"),
            adapter_sha256=adapter_hash,
            prompt_sha256=prompt_hash,
            candidates=offered,
            parameters=cache_parameters,
        )

    def evaluate_note(note: Note) -> tuple[m.NoteResult, dict]:
        candidates = build_candidates(note, dataset.label_space, candidate_space, seed)
        offered = {candidate.code for candidate in candidates}

        key = key_for(note, offered)
        cached = store.get(key)
        # An empty cached prediction is the one answer a parser fix can change,
        # and the cache stores the parse rather than the response, so it cannot
        # be reinterpreted after the fact. `recompute_empty` pays to generate
        # exactly those notes again and reuses every note that produced codes.
        # It steers execution rather than the answer, so it stays out of the
        # cache key, like concurrency.
        if cached is not None and not (recompute_empty and not cached.get("codes")):
            return _rebuild(note, offered, cached)

        start = time.perf_counter()
        truncated = False
        salvaged = False
        error = None
        try:
            prediction = predictor.predict(note, candidates)
            codes = dict(prediction.codes)
            truncated = prediction.truncated
            salvaged = getattr(prediction, "salvaged", False)
            latency = prediction.latency_s or (time.perf_counter() - start)
            usage = prediction.usage
        except Truncated as exc:
            # Truncation is a distinct outcome from an empty prediction and is
            # scored as such: no codes, but counted in the truncation rate.
            codes, truncated, error = {}, True, str(exc)
            latency = time.perf_counter() - start
            usage = {}
        except Exception as exc:  # noqa: BLE001
            codes, error = {}, f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            usage = {}

        result = m.NoteResult(
            note_id=note.note_id,
            gold=list(note.gold_codes),
            pred=list(codes),
            pred_spans={code: span for code, span in codes.items() if span is not None},
            gold_spans=note.gold_spans,
            latency_s=latency,
            truncated=truncated,
            candidates=offered,
        )
        record = check_record(
            {
                "note_id": note.note_id,
                "gold": sorted(note.gold_codes),
                "pred": sorted(codes),
                "pred_spans": {code: list(span) for code, span in codes.items() if span},
                "latency_s": round(latency, 3),
                "truncated": truncated,
                "n_candidates": len(offered),
                "usage": usage,
                "error": sanitise_error(error),
                "salvaged": salvaged,
            }
        )

        # Only successes are cached. A rate limit or a timeout says nothing about
        # the model, and freezing one in would make the next run inherit tonight's
        # bad luck forever. Truncation is a genuine model outcome, so it caches.
        if error is None:
            store.put(
                key,
                {
                    "codes": {code: list(span) if span else None for code, span in codes.items()},
                    "truncated": truncated,
                    "usage": usage,
                    "latency_s": latency,
                    "salvaged": salvaged,
                },
            )
        return result, record

    pairs: list[tuple[m.NoteResult, dict]] = []
    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for pair in pool.map(evaluate_note, notes):
                pairs.append(pair)
                done += 1
                if progress and done % 25 == 0:
                    print(f"  {done}/{len(notes)} notes", flush=True)
    else:
        for index, note in enumerate(notes, start=1):
            pairs.append(evaluate_note(note))
            if progress and index % 25 == 0:
                print(f"  {index}/{len(notes)} notes", flush=True)

    results = [result for result, _ in pairs]
    records = [record for _, record in pairs]

    finished = datetime.now(timezone.utc)
    # Inside a Modal container there is no repository to interrogate, so the
    # caller passes the SHA it launched from. Falling back to a local lookup
    # would silently record "unknown" for every restricted run, which is
    # precisely the run whose provenance matters most.
    if git_sha:
        git_dirty = False
    else:
        git_sha, git_dirty = git_state()
    space_label = "full" if candidate_space is None else str(candidate_space)

    manifest = RunManifest(
        run_id=_run_id(dataset.task, predictor, space_label, started),
        task=dataset.task,
        code_system=dataset.code_system,
        tier=dataset.tier,
        approach=getattr(predictor, "name", type(predictor).__name__),
        approach_version=getattr(predictor, "version", "0"),
        model_id=getattr(predictor, "model_id", "n/a"),
        candidate_space=space_label,
        n_notes=len(notes),
        dataset_manifest_sha256=dataset.manifest_sha256,
        adapter_sha256=file_sha256(adapter_path),
        prompt_sha256=text_sha256(prompt_text),
        approach_sha256=file_sha256(approach_file(predictor)),
        git_sha=git_sha,
        git_dirty=git_dirty,
        external_provider=external_provider,
        parameters=parameters or {},
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )

    report = m.evaluate(results)
    report["operational"]["error_rate"] = sum(bool(r["error"]) for r in records) / max(len(records), 1)
    report["operational"]["cache_hits"] = sum(bool(r.get("cached")) for r in records)
    # How many of those hits were answered by a different revision of the
    # adapter under a declared equivalence. Zero for an ordinary run. Non zero
    # means part of this record was produced by code that is not the code the
    # manifest names, which a reader is entitled to know without diffing hashes.
    report["operational"]["cache_carried_forward"] = getattr(store, "carried", 0)
    # Notes whose codes were recovered from unparseable JSON. They carry no
    # evidence spans, so a high count here explains an evidence coverage that
    # would otherwise look like the model having stopped citing its work.
    report["operational"]["salvaged"] = sum(bool(r.get("salvaged")) for r in records)

    return {"manifest": asdict(manifest), "metrics": report, "predictions": records}


def _run_id(task: str, predictor, space: str, when: datetime) -> str:
    name = getattr(predictor, "name", type(predictor).__name__)
    model = getattr(predictor, "model_id", "none").replace("/", "-")
    return f"{task}__{name}__{model}__cand{space}__{when.strftime('%Y%m%dT%H%M%SZ')}"


def save_run(record: dict, runs_dir: Path | None = None) -> Path:
    """Write the run record, checking every per note row on the way out."""
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    for row in record.get("predictions", []):
        check_record(row)

    path = runs_dir / f"{record['manifest']['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
    return path


# Above this share of failed notes, the run is measuring the harness rather than
# the model, and its scores are not a finding about anything.
ERROR_RATE_INVALIDATES = 0.05


def summarise(record: dict) -> str:
    """A short, log safe summary of a run.

    The error rate is printed first and loudly when it is non trivial. A run
    where most requests were rejected still produces a full set of plausible
    looking metrics, because a failed note scores as an empty prediction, and
    that number will be read as a model result unless something says otherwise.
    """
    manifest, report = record["manifest"], record["metrics"]
    core, uncertainty = report["core"], report["uncertainty"]
    operational = report["operational"]
    low, high = uncertainty["micro_f1_ci95"]
    error_rate = operational.get("error_rate", 0.0)

    lines = [
        f"{manifest['approach']} / {manifest['model_id']} on {manifest['task']} "
        f"(candidates: {manifest['candidate_space']}, n={manifest['n_notes']})",
    ]
    if error_rate > ERROR_RATE_INVALIDATES:
        failed = round(error_rate * manifest["n_notes"])
        lines.append(
            f"  *** NOT A VALID RESULT: {failed} of {manifest['n_notes']} notes failed "
            f"({error_rate:.1%}). Failed notes score as empty predictions, so every "
            f"number below understates the model. Fix the failures and rerun. ***"
        )
    lines += [
        f"  micro F1 {core['micro_f1']:.3f} [{low:.3f}, {high:.3f}]   "
        f"macro F1 {core['macro_f1']:.3f}   exact {core['exact_match_ratio']:.3f}",
        f"  cardinality ratio {core['label_cardinality_ratio']:.2f}   "
        f"truncation {operational['truncation_rate']:.1%}   "
        f"errors {error_rate:.1%}   "
        f"latency {operational['latency_mean_s']:.2f}s",
        m.format_band_table(report["bands"]),
    ]
    return "\n".join(lines)
