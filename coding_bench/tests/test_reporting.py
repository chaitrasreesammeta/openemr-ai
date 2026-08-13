"""The leaderboard must never present a failed run as a result."""

from __future__ import annotations

import json

from coding_bench.bench import reporting


def make_run(run_id: str, micro: float, error_rate: float = 0.0, candidates: str = "gold") -> dict:
    return {
        "manifest": {
            "run_id": run_id,
            "task": "icd10",
            "model_id": f"model-{run_id}",
            "candidate_space": candidates,
            "n_notes": 578,
            "dataset_manifest_sha256": "abc123def456",
            "git_sha": "0123456789ab",
            "external_provider": "groq",
        },
        "metrics": {
            "core": {"micro_f1": micro, "macro_f1": micro / 2, "exact_match_ratio": 0.2},
            "uncertainty": {"micro_f1_ci95": [micro - 0.02, micro + 0.02]},
            "bands": {
                "head": {"n_codes": 10, "n_gold_mentions": 627, "micro_precision": 1.0,
                         "micro_recall": 0.8, "micro_f1": 0.9, "macro_f1": 0.9},
                "torso": {"n_codes": 147, "n_gold_mentions": 1979, "micro_precision": 1.0,
                          "micro_recall": 0.6, "micro_f1": 0.79, "macro_f1": 0.67},
                "tail": {"n_codes": 516, "n_gold_mentions": 1032, "micro_precision": 1.0,
                         "micro_recall": 0.48, "micro_f1": 0.65, "macro_f1": 0.49},
            },
            "operational": {
                "error_rate": error_rate,
                "truncation_rate": 0.01,
                "latency_mean_s": 2.5,
            },
        },
        "predictions": [],
    }


def test_a_failed_run_is_quarantined_not_tabulated():
    """The whole point: 74 percent failures must not read as a model score."""
    rendered = reporting.render([make_run("failed", 0.138, error_rate=0.74)])
    assert "Quarantined runs" in rendered
    assert "_No valid runs yet._" in rendered
    # The number still appears, but only under the quarantine heading.
    quarantine = rendered.split("## Quarantined runs")[1]
    assert "0.138" in quarantine
    assert "74" in quarantine


def test_a_clean_run_is_tabulated():
    rendered = reporting.render([make_run("clean", 0.779, error_rate=0.005)])
    assert "Quarantined runs" not in rendered
    results = rendered.split("## Results")[1]
    assert "0.779" in results


def test_the_error_ceiling_is_the_boundary():
    just_over = reporting.render([make_run("over", 0.5, error_rate=0.06)])
    just_under = reporting.render([make_run("under", 0.5, error_rate=0.04)])
    assert "Quarantined" in just_over
    assert "Quarantined" not in just_under


def test_valid_and_invalid_runs_are_separated():
    rendered = reporting.render(
        [make_run("good", 0.779, 0.0), make_run("bad", 0.9, 0.5)]
    )
    results = rendered.split("## Results")[1].split("## Quarantined")[0]
    assert "model-good" in results
    # A higher score does not buy a failed run a place in the table.
    assert "model-bad" not in results


def test_results_are_ranked_by_score():
    rendered = reporting.render([make_run("low", 0.4), make_run("high", 0.8)])
    results = rendered.split("## Results")[1]
    assert results.index("model-high") < results.index("model-low")


def test_the_candidate_space_caveat_is_always_stated():
    """Nobody reading this later will infer that gold-only pins precision at 1."""
    rendered = reporting.render([make_run("clean", 0.779)])
    assert "recall ceiling" in rendered
    assert "cannot be compared" in rendered


def test_provenance_is_recorded_for_every_run_including_failed_ones():
    rendered = reporting.render([make_run("good", 0.8, 0.0), make_run("bad", 0.5, 0.9)])
    assert "run_id" not in rendered  # the id itself, not the key name
    assert "good" in rendered.split("## Provenance")[1]
    assert "bad" in rendered.split("## Provenance")[1]


def test_empty_leaderboard_is_honest():
    rendered = reporting.render([])
    assert "_No valid runs yet._" in rendered


def test_unreadable_run_records_do_not_break_generation(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf8")
    (tmp_path / "fine.json").write_text(json.dumps(make_run("fine", 0.7)), encoding="utf8")
    runs = reporting.load_runs(tmp_path)
    assert len(runs) == 1


def test_render_is_deterministic():
    runs = [make_run("a", 0.7), make_run("b", 0.6)]
    assert reporting.render(runs) == reporting.render(runs)


def with_predictions(run: dict, note_ids: list[str], correct: bool) -> dict:
    run = json.loads(json.dumps(run))
    run["predictions"] = [
        {"note_id": n, "gold": ["A"], "pred": ["A"] if correct else ["B"]} for n in note_ids
    ]
    return run


def test_head_to_head_only_pairs_runs_over_the_same_notes():
    a = with_predictions(make_run("a", 0.9), ["1", "2", "3"], True)
    b = with_predictions(make_run("b", 0.1), ["4", "5", "6"], False)
    assert "Head to head" not in reporting.render([a, b])


def test_head_to_head_refuses_to_pair_across_candidate_spaces():
    a = with_predictions(make_run("a", 0.9, candidates="gold"), ["1", "2"], True)
    b = with_predictions(make_run("b", 0.5, candidates="full"), ["1", "2"], False)
    assert "Head to head" not in reporting.render([a, b])


def test_a_clear_difference_is_marked_real():
    ids = [str(i) for i in range(80)]
    a = with_predictions(make_run("a", 0.9), ids, True)
    b = with_predictions(make_run("b", 0.1), ids, False)
    section = reporting.render([a, b]).split("## Head to head")[1]
    assert "**yes**" in section


def test_no_difference_is_not_marked_real():
    ids = [str(i) for i in range(80)]
    a = with_predictions(make_run("a", 0.9), ids, True)
    b = with_predictions(make_run("b", 0.9), ids, True)
    section = reporting.render([a, b]).split("## Head to head")[1]
    assert "**yes**" not in section


def test_runs_of_different_sizes_are_paired_on_the_overlap():
    """Budget cuts truncate runs; comparing the shared prefix is still valid."""
    long_run = with_predictions(make_run("long", 0.8), [str(i) for i in range(200)], True)
    short_run = with_predictions(make_run("short", 0.3), [str(i) for i in range(60)], False)
    section = reporting.render([long_run, short_run]).split("## Head to head")[1]
    assert "Head to head" not in section or "model-long" in section
    # The comparison must state how many notes it actually used.
    assert "| 60 |" in section


def test_too_few_shared_notes_is_not_compared():
    """A handful of shared notes gives an interval too wide to mean anything."""
    long_run = with_predictions(make_run("long", 0.8), [str(i) for i in range(200)], True)
    tiny = with_predictions(make_run("tiny", 0.3), [str(i) for i in range(10)], False)
    assert "Head to head" not in reporting.render([long_run, tiny])


def test_deduplication_survives_a_record_with_no_start_time():
    """Coverage decides which run wins, and a missing tiebreak is not fatal.

    `started_at` only separates two runs of the same length. Requiring it took
    the entire leaderboard down with a KeyError on any record that predated the
    field or was written by hand.
    """
    small = make_run("small", 0.4)
    small["manifest"]["n_notes"] = 150
    large = make_run("large", 0.9)
    for run in (small, large):
        run["manifest"]["model_id"] = "same-model"
        run["manifest"].pop("started_at", None)

    rendered = reporting.render([small, large])
    assert "| 578 |" in rendered
    assert "| 150 |" not in rendered, "the better covered run supersedes the short one"
