"""Recompute metrics from predictions already paid for. No model is called.

The gap this closes. A run record holds two different things: the predictions,
which cost money and cannot be regenerated for free, and the metrics, which are
arithmetic over those predictions. `reporting.py` reads the **stored** metrics
for the results table, the bands and the quarantine decision, so changing
`metrics.py` left every committed number stale. Worse, the head to head section
recomputes from predictions rather than reading them, so one page would show a
new significance test against an old F1.

Nothing about that requires inference. The predictions are on disk, in git, and
a changed metric is a question about them, not about the model:

    python -m coding_bench.bench.rescore            # rewrite the stored metrics
    python -m coding_bench.bench.rescore --check    # fail if any are stale

## What can be recomputed, and what cannot

A per note record is `{note_id, gold, pred, pred_spans, latency, truncated,
n_candidates}`, because that is the most a record may carry: note text is
refused by the writer and the offered candidate list is a few hundred codes per
note that nobody wanted in git. That is enough for most of the report and not
all of it:

| Block | Recomputed | Why |
|---|---|---|
| `core` | yes | micro, macro, exact match, Jaccard, cardinality are functions of gold and pred |
| `uncertainty` | yes | bootstrap resamples the same note level results |
| `bands` | yes | head, torso and tail are frequency bands over the gold mentions in the run itself |
| `operational` | partly | counts are recomputed; latency is preserved, because the record rounds it |
| `evidence` | **no** | needs MDACE's gold spans, which are not in the record |
| `scaling` | **no** | needs the offered candidate list, of which only the size is stored |

The two that cannot be are **preserved, not recomputed and not dropped**. A
rescore that silently zeroed evidence coverage would be worse than a stale
number, because it would look like a model that stopped citing its work. If
those metrics change, they need a rerun against the gold set, and this says so
rather than pretending otherwise.

Fields the runner adds after scoring, such as the error rate and the cache
counters, are carried across untouched. They are facts about how the run went,
not arithmetic over its predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_bench.bench import metrics as m
from coding_bench.bench.runner import RUNS_DIR

# Recomputed from the stored predictions.
DERIVED = ("core", "uncertainty", "bands")
# Written by the runner from things only the run knew, or measured at full
# precision that the record then rounds away. Carried across as is.
#
# The latency figures are the subtle one. A per note record stores `latency_s`
# rounded to three decimals, so a mean recomputed from records is not the same
# number the runner measured, and it is the *less* accurate of the two. Keeping
# the runner's value is not laziness: recomputing here would quietly degrade a
# measurement and then report the disagreement as a stale metric forever.
RUNNER_FIELDS = (
    "error_rate",
    "cache_hits",
    "cache_carried_forward",
    "salvaged",
    "latency_mean_s",
    "latency_p95_s",
)
# Cannot be recomputed from a record. Preserved untouched, and named here so the
# reason is one edit away from the code that relies on it.
NEEDS_GOLD_SET = ("evidence", "scaling")


def results_from(record: dict) -> list[m.NoteResult]:
    """Rebuild the note level results a run produced, from its own record."""
    return [
        m.NoteResult(
            note_id=row["note_id"],
            gold=row["gold"],
            pred=row["pred"],
            pred_spans={code: tuple(span) for code, span in (row.get("pred_spans") or {}).items()},
            latency_s=row.get("latency_s") or 0.0,
            truncated=bool(row.get("truncated")),
        )
        for row in record.get("predictions", [])
    ]


def rescored(record: dict) -> dict:
    """The record's metrics block, recomputed where the predictions allow."""
    results = results_from(record)
    if not results:
        return record["metrics"]

    fresh = m.evaluate(results)
    stored = record["metrics"]

    updated = {key: fresh[key] for key in DERIVED}
    updated["operational"] = {
        **fresh["operational"],
        # Facts about the run rather than about its predictions.
        **{k: stored.get("operational", {})[k] for k in RUNNER_FIELDS
           if k in stored.get("operational", {})},
    }
    for key in NEEDS_GOLD_SET:
        if key in stored:
            updated[key] = stored[key]
    # Anything a future metrics.py adds and this file has not been taught about
    # is kept rather than dropped, so an unfamiliar block survives a rescore.
    for key, value in stored.items():
        updated.setdefault(key, value)
    return updated


def run(runs_dir: Path | None = None, check: bool = False) -> int:
    runs_dir = runs_dir or RUNS_DIR
    stale: list[str] = []
    rewritten = 0

    for path in sorted(runs_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping unreadable run record {path.name}: {exc}", file=sys.stderr)
            continue

        fresh = rescored(record)
        if fresh == record["metrics"]:
            continue

        if check:
            stale.append(path.name)
            continue

        record["metrics"] = fresh
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf8")
        rewritten += 1

    if check:
        if stale:
            print(
                f"{len(stale)} run record(s) carry metrics that disagree with the current "
                f"metrics code:\n  " + "\n  ".join(stale) + "\n\nRegenerate them from the "
                f"predictions already on disk, which costs nothing and calls no model:\n"
                f"  python -m coding_bench.bench.rescore\n"
                f"  python -m coding_bench.bench.reporting",
                file=sys.stderr,
            )
            return 1
        print("Every run record's metrics match the current metrics code")
        return 0

    print(f"Rescored {rewritten} run record(s) with no inference")
    if rewritten:
        print("Now regenerate the leaderboard: python -m coding_bench.bench.reporting")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit non zero if any stored metrics are stale"
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args()
    return run(runs_dir=args.runs_dir, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
