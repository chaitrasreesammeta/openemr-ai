"""Metrics tests against cases computed by hand.

The worked example used by most of these tests:

    n1  gold {A, B}   pred {A, C}
    n2  gold {B}      pred {B}
    n3  gold {C, D}   pred {}

    micro   TP=2 (A in n1, B in n2), FP=1 (C in n1), FN=3 (B in n1, C and D in n3)
            P = 2/3, R = 2/5, F1 = 2TP/(2TP+FP+FN) = 4/8 = 0.5
    macro   A: F1 1.0, B: P 1.0 R 0.5 F1 2/3, C: 0.0, D: 0.0  ->  mean 5/12
    sample  n1 0.5, n2 1.0, n3 0.0  ->  mean 0.5
"""

from __future__ import annotations

import math

import pytest

from coding_bench.bench import metrics as m


def note(note_id, gold, pred, **kwargs):
    return m.NoteResult(note_id=note_id, gold=list(gold), pred=list(pred), **kwargs)


@pytest.fixture
def worked_example():
    return [
        note("n1", ["A", "B"], ["A", "C"]),
        note("n2", ["B"], ["B"]),
        note("n3", ["C", "D"], []),
    ]


def close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


def test_note_result_sorts_and_deduplicates():
    result = note("n", ["B", "A", "A"], ["C", "C"])
    assert result.gold == ["A", "B"]
    assert result.pred == ["C"]


def test_micro(worked_example):
    precision, recall, f1 = m.micro_f1(worked_example)
    assert close(precision, 2 / 3)
    assert close(recall, 2 / 5)
    assert close(f1, 0.5)


def test_macro(worked_example):
    precision, recall, f1 = m.macro_f1(worked_example)
    assert close(precision, 0.5)          # (1 + 1 + 0 + 0) / 4
    assert close(recall, 0.375)           # (1 + 0.5 + 0 + 0) / 4
    assert close(f1, (1 + 2 / 3) / 4)     # 5/12


def test_sample(worked_example):
    assert close(m.sample_f1(worked_example)[2], 0.5)


def test_exact_match_and_jaccard(worked_example):
    assert close(m.exact_match_ratio(worked_example), 1 / 3)
    # n1 1/3, n2 1, n3 0
    assert close(m.jaccard(worked_example), (1 / 3 + 1 + 0) / 3)


def test_label_cardinality_ratio(worked_example):
    assert close(m.label_cardinality_ratio(worked_example), 3 / 5)


def test_empty_prediction_set_scores_zero_not_error():
    results = [note("n1", ["A"], [])]
    assert m.micro_f1(results)[2] == 0.0
    assert m.macro_f1(results)[2] == 0.0
    assert m.jaccard(results) == 0.0


def test_empty_gold_and_empty_pred_is_a_perfect_note():
    """"Nothing is codeable here" is a correct answer and must score as one."""
    results = [note("n1", [], [])]
    assert m.exact_match_ratio(results) == 1.0
    assert m.jaccard(results) == 1.0


def test_macro_restricted_to_a_label_subset(worked_example):
    # Restricted to {A}: one true positive, nothing else.
    assert close(m.macro_f1(worked_example, labels={"A"})[2], 1.0)
    # Restricted to {D}: never predicted.
    assert close(m.macro_f1(worked_example, labels={"D"})[2], 0.0)


def test_frequency_bands_split_by_gold_frequency(monkeypatch):
    monkeypatch.setattr(m, "HEAD_SIZE", 1)
    monkeypatch.setattr(m, "TAIL_MAX_FREQUENCY", 1)
    results = (
        [note(f"h{i}", ["X"], []) for i in range(10)]
        + [note(f"m{i}", ["Y"], []) for i in range(3)]
        + [note("t0", ["Z"], [])]
    )
    bands = m.frequency_bands(results)
    assert bands["head"] == {"X"}
    assert bands["torso"] == {"Y"}
    assert bands["tail"] == {"Z"}


def test_band_table_covers_every_gold_code(worked_example):
    table = m.band_table(worked_example)
    assert set(table) == {"head", "torso", "tail"}
    assert sum(row["n_codes"] for row in table.values()) == 4
    assert sum(row["n_gold_mentions"] for row in table.values()) == 5


def test_tail_collapse_is_visible_in_the_band_table(monkeypatch):
    """A model that only gets head codes right must not look good in the tail."""
    monkeypatch.setattr(m, "HEAD_SIZE", 1)
    monkeypatch.setattr(m, "TAIL_MAX_FREQUENCY", 1)
    results = [note(f"h{i}", ["X"], ["X"]) for i in range(10)] + [note("t0", ["Z"], [])]
    table = m.band_table(results)
    assert close(table["head"]["micro_f1"], 1.0)
    assert close(table["tail"]["micro_f1"], 0.0)


def test_distractor_false_positive_rate(worked_example):
    candidates = {"A", "B", "C", "D", "E"}
    # offered distractors: n1 {C,D,E}=3, n2 {A,C,D,E}=4, n3 {A,B,E}=3 -> 10
    # wrongly selected: C in n1 -> 1
    assert close(m.distractor_false_positive_rate(worked_example, candidates), 0.1)


def test_evidence_overlap():
    results = [
        note(
            "n1", ["A"], ["A"],
            pred_spans={"A": (15, 25)},
            gold_spans={"A": [(10, 20)]},
        )
    ]
    scores = m.evidence_scores(results)
    assert scores["n_scored"] == 1
    assert close(scores["evidence_coverage"], 1.0)
    assert close(scores["evidence_hit_rate"], 1.0)
    # overlap 5 characters, union 15
    assert close(scores["evidence_mean_iou"], 1 / 3)


def test_evidence_coverage_counts_codes_cited_without_a_span():
    results = [
        note("n1", ["A"], ["A"], pred_spans={"A": (10, 20)}, gold_spans={"A": [(10, 20)]}),
        note("n2", ["A"], ["A"], pred_spans={}, gold_spans={"A": [(0, 5)]}),
    ]
    scores = m.evidence_scores(results)
    assert close(scores["evidence_coverage"], 0.5)
    assert close(scores["evidence_hit_rate"], 1.0)


def test_evidence_ignores_wrong_codes():
    """A cited span for a code that is not in gold earns nothing."""
    results = [note("n1", ["A"], ["B"], pred_spans={"B": (0, 10)}, gold_spans={"A": [(0, 10)]})]
    assert m.evidence_scores(results)["n_scored"] == 0


def test_non_overlapping_span_is_a_miss():
    results = [note("n1", ["A"], ["A"], pred_spans={"A": (100, 110)}, gold_spans={"A": [(0, 10)]})]
    scores = m.evidence_scores(results)
    assert close(scores["evidence_coverage"], 1.0)
    assert close(scores["evidence_hit_rate"], 0.0)


def test_bootstrap_is_deterministic_and_brackets_the_point(worked_example):
    point, low, high = m.bootstrap_ci(worked_example, "micro_f1", rounds=200)
    assert close(point, 0.5)
    assert low <= point <= high
    assert (point, low, high) == m.bootstrap_ci(worked_example, "micro_f1", rounds=200)


def test_paired_bootstrap_finds_no_difference_between_identical_systems(worked_example):
    result = m.paired_bootstrap(worked_example, worked_example, rounds=200)
    assert result["delta"] == 0.0
    assert result["significant"] is False


def test_paired_bootstrap_finds_a_clear_difference():
    good = [note(f"n{i}", ["A"], ["A"]) for i in range(50)]
    bad = [note(f"n{i}", ["A"], ["B"]) for i in range(50)]
    result = m.paired_bootstrap(good, bad, rounds=200)
    assert close(result["delta"], 1.0)
    assert result["significant"] is True


def test_paired_bootstrap_refuses_mismatched_note_sets():
    a = [note("n1", ["A"], ["A"])]
    b = [note("n2", ["A"], ["A"])]
    with pytest.raises(ValueError, match="identical note sets"):
        m.paired_bootstrap(a, b)


def test_operational_counts_truncation():
    results = [
        note("n1", ["A"], ["A"], latency_s=1.0, truncated=False),
        note("n2", ["A"], [], latency_s=3.0, truncated=True),
    ]
    ops = m.operational(results)
    assert close(ops["truncation_rate"], 0.5)
    assert close(ops["latency_mean_s"], 2.0)


def test_evaluate_report_shape(worked_example):
    report = m.evaluate(worked_example, candidates={"A", "B", "C", "D", "E"})
    assert close(report["core"]["micro_f1"], 0.5)
    assert set(report["bands"]) == {"head", "torso", "tail"}
    assert report["uncertainty"]["micro_f1_ci95"][0] <= report["core"]["micro_f1"]
    assert close(report["scaling"]["distractor_fp_rate"], 0.1)
    assert report["operational"]["n_notes"] == 3


def test_format_band_table_is_markdown(worked_example):
    rendered = m.format_band_table(m.band_table(worked_example))
    assert rendered.startswith("| Band |")
    assert "head" in rendered
