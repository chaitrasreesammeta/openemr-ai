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

# The repository README carries the four winning rows, and only those. It is
# read by people who will never open the leaderboard, and a table copied there
# by hand goes stale the first run that lands, quietly and in the most visible
# file in the repo. So it is generated with the board and checked with the
# board, between markers, leaving the rest of that file alone.
REPO_README = RESULTS_DIR.parent.parent / "README.md"
SUMMARY_START = "<!-- coding-bench-summary:start -->"
SUMMARY_END = "<!-- coding-bench-summary:end -->"

# What each model is called in a table cell, and how it was served.
#
# The manifest carries the exact model id, and it has to: it is what a rerun
# must match character for character, and it is the key the board groups on.
# It is also unreadable at a glance.
# `meta-models/Muse-Glimmer-30B-GGUF:kquant-dynamic` is forty eight characters
# of repository path to say "Muse Glimmer 30B", and it sets the width of the
# first column in every table, which pushes the scores it should be introducing
# off to the right. So the tables carry the short name and the `## Models` key
# carries the id: readable at the top, exact underneath.
#
# The serving string is the second half of the answer and belongs next to the
# id rather than in every row. A quantised build is not the released model, a
# point the gemma adapter makes at length, and it is also most of why one row
# reports 119 seconds a note and another reports 5.
#
# The map is explicit rather than parsed out of the path, because a parser has
# to guess which trailing segment is a size, which is a quantisation and which
# is an organisation, and it guesses wrong the first time a vendor names
# something differently. An unmapped id falls through verbatim, so a new model
# appears as its raw path; `test_every_committed_model_is_named` turns that into
# a failed check rather than something nobody notices.
MODEL_NAMES: dict[str, tuple[str, str]] = {
    "claude-sonnet-5": ("Claude Sonnet 5", "Anthropic API"),
    "google/gemma-4-26B-A4B-it:qat-UD-Q4_K_XL": (
        "Gemma 4 26B-A4B",
        "GGUF QAT UD-Q4_K_XL, self-hosted llama.cpp",
    ),
    "meta-models/Muse-Glimmer-30B-GGUF:kquant-dynamic": (
        "Muse Glimmer 30B",
        "GGUF dynamic K-quant, self-hosted llama.cpp",
    ),
    "openai/gpt-oss-120b": ("GPT-OSS 120B", "Groq API"),
    "qwen/qwen3.6-27b": ("Qwen3.6 27B", "Groq API"),
    # No build tag, unlike the GGUF entries and unlike the FP8 arm this replaced.
    # Groq does not publish its serving precision, so a tag here would be a claim
    # about somebody else's stack. Named the same way the 3.6 row above is.
    "qwen/qwen3.8-27b": ("Qwen3.8 27B", "Groq API"),
}


def display_name(model_id: str) -> str:
    """The table label for a model id, or the id itself if it has no entry."""
    return MODEL_NAMES.get(model_id, (model_id, ""))[0]


def _models_section(runs: list[dict]) -> list[str]:
    """The key from short name back to the exact id, for every model shown.

    Every model in `runs`, not just the tabulated ones: a quarantined run names
    its model too, and a reader who meets `Gemma 4 26B-A4B` under Quarantined
    has the same question as one who meets a name in Results.
    """
    seen = {run["manifest"]["model_id"] for run in runs}
    if not seen:
        return []

    lines = [
        "## Models",
        "",
        "The tables above name the model; the exact id is what reproduces it. A "
        "quantised build is not the released model and is never tabulated as "
        "though it were, so the build is named here too.",
        "",
        "| Model | Served as | Exact id |",
        "|---|---|---|",
    ]
    rows = [(display_name(model_id), MODEL_NAMES.get(model_id, (model_id, "n/a"))[1], model_id)
            for model_id in seen]
    # Case insensitive, or `GPT-OSS` sorts above `Gemma` on the capital and the
    # list stops looking alphabetical to anyone reading it.
    for name, served, model_id in sorted(rows, key=lambda row: (row[0].lower(), row[2])):
        lines.append(f"| {name} | {served} | `{model_id}` |")
    lines.append("")
    return lines


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
        f"| {display_name(manifest['model_id'])} "
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


def _note_results(run: dict, keep: set[str]):
    """Only the shared notes, so a pairing is genuinely note for note."""
    from coding_bench.bench import metrics as m

    return [
        m.NoteResult(note_id=row["note_id"], gold=row["gold"], pred=row["pred"])
        for row in run["predictions"]
        if row["note_id"] in keep
    ]


def _separated(leader: dict, other: dict) -> bool:
    """Does a paired bootstrap actually put `leader` ahead of `other`?

    False means the two are inside each other's noise, which is the only honest
    reading of a gap the interval spans. Pairs too short to compare come back
    True rather than tied: a tie is a claim, and fewer than `MIN_PAIRED_NOTES`
    shared notes cannot support one.
    """
    from coding_bench.bench import metrics as m

    if not (leader["predictions"] and other["predictions"]):
        return True
    if not _comparable(leader, other):
        return True
    shared = _shared_notes(leader, other)
    outcome = m.paired_bootstrap(
        _note_results(leader, shared), _note_results(other, shared), statistic="micro_f1"
    )
    return bool(outcome["significant"])


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

    for a, b in pairs:
        shared = _shared_notes(a, b)
        rows_a, rows_b = _note_results(a, shared), _note_results(b, shared)
        for statistic in ("micro_f1", "macro_f1", "exact_match"):
            outcome = m.paired_bootstrap(rows_a, rows_b, statistic=statistic)
            lines.append(
                f"| {display_name(a['manifest']['model_id'])} "
                f"| {display_name(b['manifest']['model_id'])} "
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

    Coverage decides first. A 578 note run supersedes the 150 note run that
    preceded it whichever order they happened in, because the runner slices the
    dataset in sorted note id order, so the short run's notes are a prefix of
    the long one's and its numbers carry strictly less information.

    Then the failure rate, and only then recency. Ranking equal length runs on
    recency alone let a noisier rerun represent a configuration, and that is not
    a cosmetic preference. A failed note scores as an empty prediction, so it
    costs recall, so the record with more transient provider errors reports a
    lower F1 for the same answers. Qwen3.6 on icd10 at full candidates had
    eleven records spanning 0.5132 to 0.5283, and the most recent carried
    fourteen rate limited notes against the cleanest one's five. On the 564
    notes both of those runs answered, both score 0.5335 exactly: the
    predictions are cache identical and only the error set differs. Picking on
    recency published the noisiest of them, which was enough to hand icd10 full
    to Qwen3.8 on a gap of 0.0028 that the head to head table on this same page
    called insignificant, and that reverses once the errored notes come out.

    Recency still breaks ties between runs of equal length and equal error rate,
    where by this argument the numbers cannot differ anyway.

    Nothing is deleted. The superseded records stay on disk and stay listed under
    Provenance, so the attempt is still visible; it just stops being counted
    twice in the tables.
    """
    best: dict[tuple, tuple] = {}
    for run in runs:
        manifest = run["manifest"]
        key = (manifest["model_id"], manifest["task"], str(manifest["candidate_space"]))
        # Every term is read defensively. `started_at` only breaks ties between
        # runs of equal length, so a record without one still sorts correctly on
        # coverage; reading it directly took the whole leaderboard down with a
        # KeyError instead, which is a steep price for a field that is only a
        # tiebreak. The error rate is negated so that fewer failures rank
        # higher under the same `>` comparison, and a record that never stated
        # one is read as clean, matching `is_valid`.
        error_rate = run["metrics"]["operational"].get("error_rate", 0.0)
        rank = (manifest["n_notes"], -error_rate, manifest.get("started_at", ""))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, run)
    return [run for _rank, run in best.values()]


def conditions(valid: list[dict]) -> list[tuple[tuple[str, str], list[dict]]]:
    """Runs grouped by task and candidate space, `full` first, each ranked.

    One table per task and candidate space. A single ranked table put a cpt
    number next to an icd10 one and a gold number next to a full one, which is
    the comparison this file says twice cannot be made, and it is the reading
    that picks a model on a recall ceiling. `full` leads, because it is the
    condition that predicts deployment.

    Shared with the README summary block, so the row the README calls the winner
    of a condition is the row at the top of that condition's table, by
    construction rather than by anybody checking.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in valid:
        key = (run["manifest"]["task"], run["manifest"]["candidate_space"])
        groups.setdefault(key, []).append(run)
    return [
        (key, sorted(groups[key], key=lambda run: -run["metrics"]["core"]["micro_f1"]))
        for key in sorted(groups, key=lambda key: (key[1] != "full", key[0]))
    ]


def summary_block(runs: list[dict]) -> str:
    """The winner of each condition, for the repository README.

    Four rows and a caveat. Anything longer belongs on the board, which the
    block links to; the README's job is to say what the benchmark found, not to
    reproduce it.

    A row names every model the leader is not separated from, because ranking
    on micro F1 alone will crown one on a gap its own interval spans. That is
    not hypothetical: icd10 at full candidates was handed to Qwen3.8 on 0.0028
    over Qwen3.6 while the head to head table two files away marked the same
    comparison insignificant. The README is the most read file in the
    repository and the one place a reader will not go looking for a caveat, so
    the tie is stated here rather than left to be discovered.
    """
    valid = deduplicate([run for run in runs if is_valid(run)])

    lines = [
        SUMMARY_START,
        "",
        "<!-- Generated by coding_bench/bench/reporting.py. Edits between these "
        "markers are overwritten; CI fails if the committed copy is stale. -->",
        "",
    ]

    if not valid:
        lines += ["_No valid runs yet._", "", SUMMARY_END]
        return "\n".join(lines)

    lines += [
        "| Task | Candidates | Best model | n | Micro F1 [95% CI] | Latency |",
        "|---|---|---|---:|---|---:|",
    ]
    tied_anywhere = False
    for (task, space), ranked in conditions(valid):
        run = ranked[0]
        tied = [other for other in ranked[1:] if not _separated(run, other)]
        tied_anywhere = tied_anywhere or bool(tied)
        names = ", ".join(
            display_name(r["manifest"]["model_id"]) for r in (run, *tied)
        )
        if tied:
            names += " (tied)"
        core, ops = run["metrics"]["core"], run["metrics"]["operational"]
        low, high = run["metrics"]["uncertainty"]["micro_f1_ci95"]
        lines.append(
            f"| {task} | {space} | {names} "
            f"| {run['manifest']['n_notes']} "
            f"| {core['micro_f1']:.3f} [{low:.3f}, {high:.3f}] "
            f"| {ops['latency_mean_s']:.1f}s |"
        )
    if tied_anywhere:
        lines += [
            "",
            "Models marked tied are not separated from the first named by a "
            "paired bootstrap over the notes they share, so the ordering "
            "between them is resampling noise and not a result. The interval, "
            "n and latency in such a row describe the first named model only.",
        ]
    lines += ["", SUMMARY_END]
    return "\n".join(lines)


def with_summary(readme: str, block: str) -> str:
    """Swap the delimited block into a README, leaving every other line alone.

    A README with no markers comes back unchanged rather than raising. The
    markers are checked by a test instead, so losing them fails somewhere that
    names the problem rather than in the middle of a run that had numbers to
    write.
    """
    start = readme.find(SUMMARY_START)
    end = readme.find(SUMMARY_END)
    if start == -1 or end == -1 or end < start:
        return readme
    return readme[:start] + block + readme[end + len(SUMMARY_END) :]


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
        for (task, space), ranked in conditions(valid):
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

    # After Results and before Quarantined, so it sits between the two places a
    # short name is met. It also closes the Results section, which the CI
    # summary reads by slicing to the next `## `; that slice used to end at
    # Quarantined and now ends here, with the same rows inside it.
    lines += _models_section(runs)

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
                f"| {display_name(manifest['model_id'])} | {manifest['task']} "
                f"| {manifest['candidate_space']} "
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
                f"### {display_name(manifest['model_id'])}, {manifest['task']}, "
                f"candidates {manifest['candidate_space']}",
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

    runs = load_runs(args.runs_dir)
    rendered = render(runs)

    # The README is regenerated from its own committed copy, so the only thing
    # that can differ is the block between the markers.
    readme = REPO_README.read_text(encoding="utf8") if REPO_README.exists() else ""
    readme_wanted = with_summary(readme, summary_block(runs)) if readme else ""

    if args.check:
        stale = []
        current = LEADERBOARD.read_text(encoding="utf8") if LEADERBOARD.exists() else ""
        if current != rendered:
            stale.append(str(LEADERBOARD))
        if readme and readme != readme_wanted:
            stale.append(str(REPO_README))
        if stale:
            print(
                f"Stale: {', '.join(stale)}. Regenerate with "
                "`python -m coding_bench.bench.reporting` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("LEADERBOARD.md and the README summary are up to date")
        return 0

    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(rendered, encoding="utf8")
    print(f"Wrote {LEADERBOARD}")
    if readme and readme != readme_wanted:
        REPO_README.write_text(readme_wanted, encoding="utf8")
        print(f"Updated the summary block in {REPO_README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
