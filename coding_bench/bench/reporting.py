"""Generate LEADERBOARD.md from run records. Never maintained by hand.

Two rules shape this file, both learned the hard way in one night:

  1. A run whose notes largely failed is not a result. It still produces a full
     set of plausible numbers, because a failed note scores as an empty
     prediction, so it is quarantined into its own section with the failure
     rate stated, rather than being tabulated next to real scores.

  2. The candidate space is part of the number. At gold-only candidates
     precision is 1.000 by construction, so the F1 is a recall ceiling and not
     a deployment estimate. The table says so, every time, because nobody
     reading a leaderboard six months from now will infer it.

    python -m coding_bench.bench.reporting            # rewrite LEADERBOARD.md
    python -m coding_bench.bench.reporting --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_bench.bench.runner import ERROR_RATE_INVALIDATES, RESULTS_DIR, RUNS_DIR

LEADERBOARD = RESULTS_DIR / "LEADERBOARD.md"


def load_runs(runs_dir: Path | None = None) -> list[dict]:
    runs_dir = runs_dir or RUNS_DIR
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf8")))
        except json.JSONDecodeError as exc:
            print(f"Skipping unreadable run record {path.name}: {exc}", file=sys.stderr)
    return runs


def is_valid(run: dict) -> bool:
    return run["metrics"]["operational"].get("error_rate", 0.0) <= ERROR_RATE_INVALIDATES


def _row(run: dict) -> str:
    manifest, metrics = run["manifest"], run["metrics"]
    core, ops = metrics["core"], metrics["operational"]
    low, high = metrics["uncertainty"]["micro_f1_ci95"]
    bands = metrics["bands"]
    return (
        f"| {manifest['model_id']} "
        f"| {manifest['task']} "
        f"| {manifest['candidate_space']} "
        f"| {manifest['n_notes']} "
        f"| {core['micro_f1']:.3f} [{low:.3f}, {high:.3f}] "
        f"| {core['macro_f1']:.3f} "
        f"| {bands.get('head', {}).get('micro_f1', 0):.3f} "
        f"| {bands.get('tail', {}).get('micro_f1', 0):.3f} "
        f"| {core['exact_match_ratio']:.3f} "
        f"| {ops['truncation_rate']:.1%} "
        f"| {ops['latency_mean_s']:.1f}s |"
    )


# Below this many shared notes a paired interval is too wide to mean anything,
# so the pair is dropped rather than reported with a misleading verdict.
MIN_PAIRED_NOTES = 50


def _shared_notes(a: dict, b: dict) -> set[str]:
    """Note ids both runs actually scored.

    Runs get truncated for budget, so a 150 note run and a 578 note run are
    routine. Because the runner slices the dataset in sorted note id order, the
    shorter run's notes are a prefix of the longer one's, and comparing on the
    overlap is exactly right. Requiring identical sets instead would silently
    refuse to compare any model that had to be cut short, which is most of the
    interesting ones.
    """
    keys = ("task", "candidate_space")
    if any(a["manifest"][key] != b["manifest"][key] for key in keys):
        return set()
    return {row["note_id"] for row in a["predictions"]} & {
        row["note_id"] for row in b["predictions"]
    }


def _comparable(a: dict, b: dict) -> bool:
    return len(_shared_notes(a, b)) >= MIN_PAIRED_NOTES


def _paired_section(valid: list[dict]) -> list[str]:
    """Head to head deltas, with the significance rule applied.

    A difference in F1 of a couple of points looks decisive in a table and is
    routinely noise on 578 notes. Nothing is called a win here unless the paired
    interval excludes zero.
    """
    from coding_bench.bench import metrics as m

    pairs = [
        (a, b)
        for i, a in enumerate(valid)
        for b in valid[i + 1 :]
        if _comparable(a, b) and a["predictions"] and b["predictions"]
    ]
    if not pairs:
        return []

    lines = [
        "## Head to head",
        "",
        "Paired bootstrap over the same notes. A difference counts as real only "
        "when the 95% interval excludes zero; anything else is resampling noise "
        "however large the gap looks in the table above.",
        "",
        "| A | B | Candidates | n | Statistic | Delta (A - B) | 95% CI | Real? |",
        "|---|---|---|---:|---|---:|---|---|",
    ]

    def results_of(run: dict, keep: set[str]):
        """Only the shared notes, so the pairing is genuinely note for note."""
        return [
            m.NoteResult(note_id=row["note_id"], gold=row["gold"], pred=row["pred"])
            for row in run["predictions"]
            if row["note_id"] in keep
        ]

    for a, b in pairs:
        shared = _shared_notes(a, b)
        rows_a, rows_b = results_of(a, shared), results_of(b, shared)
        for statistic in ("micro_f1", "macro_f1", "exact_match"):
            outcome = m.paired_bootstrap(rows_a, rows_b, statistic=statistic)
            lines.append(
                f"| {a['manifest']['model_id']} | {b['manifest']['model_id']} "
                f"| {a['manifest']['candidate_space']} | {len(shared)} | {statistic} "
                f"| {outcome['delta']:+.4f} "
                f"| [{outcome['ci_low']:+.4f}, {outcome['ci_high']:+.4f}] "
                f"| {'**yes**' if outcome['significant'] else 'no'} |"
            )
    lines.append("")
    return lines


def deduplicate(runs: list[dict]) -> list[dict]:
    """Keep one run per model, task and candidate space: the best covered one.

    Reruns are routine, because every step is cache seeded and re-running a
    finished configuration costs almost nothing, so records for the same
    configuration accumulate. Left alone they all reach the results table, and
    worse they all reach the head to head, where a model gets paired against
    itself and reports a delta of exactly zero with a tight interval. That reads
    like a finding rather than like the same run twice.

    Coverage decides, not recency. A 578 note run supersedes the 150 note run
    that preceded it whichever order they happened in, because the runner slices
    the dataset in sorted note id order, so the short run's notes are a prefix of
    the long one's and its numbers carry strictly less information. Recency only
    breaks ties between runs of equal length.

    Nothing is deleted. The superseded records stay on disk and stay listed under
    Provenance, so the attempt is still visible; it just stops being counted
    twice in the tables.
    """
    best: dict[tuple, tuple] = {}
    for run in runs:
        manifest = run["manifest"]
        key = (manifest["model_id"], manifest["task"], str(manifest["candidate_space"]))
        # `started_at` only breaks ties between runs of equal length, so a
        # record without one still sorts correctly on coverage. Reading it
        # directly took the whole leaderboard down with a KeyError instead,
        # which is a steep price for a field that is only a tiebreak.
        rank = (manifest["n_notes"], manifest.get("started_at", ""))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, run)
    return [run for _rank, run in best.values()]


def render(runs: list[dict]) -> str:
    valid = deduplicate([run for run in runs if is_valid(run)])
    invalid = [run for run in runs if not is_valid(run)]

    valid.sort(key=lambda run: -run["metrics"]["core"]["micro_f1"])

    lines = [
        "# Leaderboard",
        "",
        "Generated by `coding_bench/bench/reporting.py`. Do not edit by hand:",
        "CI regenerates this file and fails if the committed copy differs.",
        "",
    ]

    if valid:
        lines += ["## Results", ""]
        # One table per task and candidate space, each ranked best first.
        # A single ranked table put a cpt number next to an icd10 one and a
        # gold number next to a full one, which is the comparison the note
        # below says cannot be made, and it is the reading that picks a model
        # on a recall ceiling. `full` leads, because it is the condition that
        # predicts deployment.
        groups: dict[tuple[str, str], list[dict]] = {}
        for run in valid:
            key = (run["manifest"]["task"], run["manifest"]["candidate_space"])
            groups.setdefault(key, []).append(run)

        for task, space in sorted(groups, key=lambda key: (key[1] != "full", key[0])):
            ranked = sorted(
                groups[(task, space)],
                key=lambda run: -run["metrics"]["core"]["micro_f1"],
            )
            lines += [
                f"### {task}, {space} candidates",
                "",
                "| Model | Task | Candidates | n | Micro F1 [95% CI] | Macro F1 | Head F1 | Tail F1 | Exact | Trunc | Latency |",
                "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
            lines += [_row(run) for run in ranked]
            lines += [""]

        lines += [
            "**Reading the candidate space.** `gold` offers only the note's correct "
            "codes, so precision is 1.000 by construction and the F1 is a recall "
            "ceiling, not a deployment estimate. `full` offers the whole catalogue "
            "and is the condition that predicts real behaviour. A number from one "
            "cannot be compared against a number from the other.",
            "",
            "**Head and tail** are frequency bands over the gold label space: the ten "
            "most frequent codes, and codes with five or fewer gold mentions. The gap "
            "between them is the long-tail collapse a single F1 hides.",
            "",
        ]
    else:
        lines += ["## Results", "", "_No valid runs yet._", ""]

    if invalid:
        lines += [
            "## Quarantined runs",
            "",
            f"These runs exceeded the {ERROR_RATE_INVALIDATES:.0%} failure ceiling. Failed "
            "notes score as empty predictions, so their metrics understate the model by "
            "an unknown amount and are not results. They are listed so the attempt is "
            "not silently forgotten.",
            "",
            "| Model | Task | Candidates | n | Failed | Truncated | Reported micro F1 (not valid) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for run in invalid:
            manifest, ops = run["manifest"], run["metrics"]["operational"]
            lines.append(
                f"| {manifest['model_id']} | {manifest['task']} | {manifest['candidate_space']} "
                f"| {manifest['n_notes']} | {ops['error_rate']:.1%} | {ops['truncation_rate']:.1%} "
                f"| {run['metrics']['core']['micro_f1']:.3f} |"
            )
        lines.append("")

    lines += _paired_section(valid)

    if valid:
        lines += ["## Frequency bands", ""]
        for run in valid:
            manifest = run["manifest"]
            lines += [
                f"### {manifest['model_id']}, {manifest['task']}, candidates {manifest['candidate_space']}",
                "",
                "| Band | Codes | Gold mentions | Micro P | Micro R | Micro F1 | Macro F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for band in ("head", "torso", "tail"):
                row = run["metrics"]["bands"].get(band)
                if not row:
                    continue
                lines.append(
                    f"| {band} | {row['n_codes']} | {row['n_gold_mentions']} | "
                    f"{row['micro_precision']:.3f} | {row['micro_recall']:.3f} | "
                    f"{row['micro_f1']:.3f} | {row['macro_f1']:.3f} |"
                )
            lines.append("")

    lines += ["## Provenance", ""]
    for run in runs:
        manifest = run["manifest"]
        provider = manifest.get("external_provider") or "modal only"
        lines.append(
            f"- `{manifest['run_id']}`: dataset "
            f"`{(manifest.get('dataset_manifest_sha256') or 'n/a')[:12]}`, "
            f"git `{manifest['git_sha'][:12]}`, provider {provider}"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit non zero if the committed file is stale")
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args()

    rendered = render(load_runs(args.runs_dir))

    if args.check:
        current = LEADERBOARD.read_text(encoding="utf8") if LEADERBOARD.exists() else ""
        if current != rendered:
            print(
                "LEADERBOARD.md is stale. Regenerate it with "
                "`python -m coding_bench.bench.reporting` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("LEADERBOARD.md is up to date")
        return 0

    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(rendered, encoding="utf8")
    print(f"Wrote {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
