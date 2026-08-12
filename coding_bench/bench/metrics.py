"""Metrics for multi-label medical coding.

The design constraint here is that a single F1 hides the failure modes that
decide whether a coding system is deployable. So every report carries, as first
class results and not footnotes:

  * the frequency band table, because macro F1 averages away tail collapse
  * bootstrap confidence intervals, because 312 notes is a small evaluation set
  * evidence overlap, because a code without defensible evidence is not billable

Definitions used throughout, stated because the literature is inconsistent:

  micro       pooled over all (note, code) decisions
  macro       unweighted mean over labels, taken over the label space given
  sample      per note, then averaged over notes
  exact match fraction of notes where the predicted code set equals gold exactly
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

BOOTSTRAP_SEED = 0
BOOTSTRAP_ROUNDS = 1000
HEAD_SIZE = 10
TAIL_MAX_FREQUENCY = 5


@dataclass
class NoteResult:
    """One note's outcome. Note text is deliberately not representable here."""

    note_id: str
    gold: list[str]
    pred: list[str]
    # code -> (begin, end) character span the model cited, where it cited one
    pred_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    # code -> list of gold spans from MDACE
    gold_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    latency_s: float | None = None
    truncated: bool = False
    # The candidate codes this note was actually offered, when the run varied
    # the candidate space per note. Used by the scaling stress.
    candidates: set[str] | None = None

    def __post_init__(self):
        # Order must never affect a score, and duplicates must never inflate one.
        self.gold = sorted(set(self.gold))
        self.pred = sorted(set(self.pred))


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _counts(results: Sequence[NoteResult], labels: set[str] | None = None):
    """Per label true positive, false positive, false negative counts."""
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for result in results:
        gold = set(result.gold)
        pred = set(result.pred)
        if labels is not None:
            gold &= labels
            pred &= labels
        for code in pred & gold:
            tp[code] += 1
        for code in pred - gold:
            fp[code] += 1
        for code in gold - pred:
            fn[code] += 1
    return tp, fp, fn


def micro_f1(results: Sequence[NoteResult], labels: set[str] | None = None) -> tuple[float, float, float]:
    tp, fp, fn = _counts(results, labels)
    return _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))


def macro_f1(results: Sequence[NoteResult], labels: set[str] | None = None) -> tuple[float, float, float]:
    """Unweighted mean over labels.

    The label space is every code that appears in gold, plus, when the caller
    passes one, whatever else it names. Codes a model hallucinates outside that
    space still count as false positives in micro, but they cannot dilute macro
    by adding zero scored labels that were never part of the task.
    """
    tp, fp, fn = _counts(results, labels)
    space = labels if labels is not None else {code for result in results for code in result.gold}
    if not space:
        return 0.0, 0.0, 0.0
    scores = [_prf(tp[code], fp[code], fn[code]) for code in sorted(space)]
    return tuple(float(np.mean([s[i] for s in scores])) for i in range(3))  # type: ignore[return-value]


def sample_f1(results: Sequence[NoteResult]) -> tuple[float, float, float]:
    per_note = []
    for result in results:
        gold, pred = set(result.gold), set(result.pred)
        per_note.append(_prf(len(pred & gold), len(pred - gold), len(gold - pred)))
    if not per_note:
        return 0.0, 0.0, 0.0
    return tuple(float(np.mean([s[i] for s in per_note])) for i in range(3))  # type: ignore[return-value]


def exact_match_ratio(results: Sequence[NoteResult]) -> float:
    if not results:
        return 0.0
    return sum(set(r.pred) == set(r.gold) for r in results) / len(results)


def jaccard(results: Sequence[NoteResult]) -> float:
    if not results:
        return 0.0
    scores = []
    for result in results:
        gold, pred = set(result.gold), set(result.pred)
        union = gold | pred
        scores.append(len(gold & pred) / len(union) if union else 1.0)
    return float(np.mean(scores))


def label_cardinality_ratio(results: Sequence[NoteResult]) -> float:
    """Mean predicted codes per note over mean gold codes per note.

    Above 1 means the model over-codes, which in billing terms is the expensive
    direction of error.
    """
    gold_total = sum(len(r.gold) for r in results)
    pred_total = sum(len(r.pred) for r in results)
    if not gold_total:
        return 0.0
    return pred_total / gold_total


def frequency_bands(results: Sequence[NoteResult]) -> dict[str, set[str]]:
    """Split the gold label space into head, torso and tail by gold frequency."""
    frequency = Counter(code for result in results for code in result.gold)
    ordered = [code for code, _ in frequency.most_common()]
    head = set(ordered[:HEAD_SIZE])
    tail = {code for code, count in frequency.items() if count <= TAIL_MAX_FREQUENCY} - head
    torso = set(frequency) - head - tail
    return {"head": head, "torso": torso, "tail": tail}


def band_table(results: Sequence[NoteResult]) -> dict[str, dict]:
    """Per band scores. Mandatory in every report."""
    frequency = Counter(code for result in results for code in result.gold)
    table = {}
    for band, codes in frequency_bands(results).items():
        micro = micro_f1(results, labels=codes)
        macro = macro_f1(results, labels=codes)
        table[band] = {
            "n_codes": len(codes),
            "n_gold_mentions": sum(frequency[code] for code in codes),
            "micro_precision": micro[0],
            "micro_recall": micro[1],
            "micro_f1": micro[2],
            "macro_f1": macro[2],
        }
    return table


def distractor_false_positive_rate(
    results: Sequence[NoteResult], candidates: set[str] | None = None
) -> float:
    """Share of offered distractors the model wrongly selected.

    A distractor is a candidate offered to the model that is not in that note's
    gold set. This is the number to watch as the candidate space grows from
    gold-only to the full catalogue, because it is the deployment risk that
    scales with catalogue size.

    Per note candidate sets win over the run wide set when a run varied what it
    offered each note, which is exactly what the scaling stress does.
    """
    offered = 0
    selected = 0
    for result in results:
        offered_here = result.candidates if result.candidates is not None else candidates
        if offered_here is None:
            continue
        gold = set(result.gold)
        distractors = set(offered_here) - gold
        offered += len(distractors)
        selected += len(set(result.pred) & distractors)
    return selected / offered if offered else 0.0


def evidence_scores(results: Sequence[NoteResult]) -> dict[str, float]:
    """Score cited spans against MDACE gold evidence.

    Only correctly predicted codes are scored. Evidence for a wrong code is not
    partially right, it is a wrong code, and micro F1 already counts it.
    """
    considered = 0
    cited = 0
    overlapping = 0
    ious: list[float] = []

    for result in results:
        for code in set(result.pred) & set(result.gold):
            gold_spans = result.gold_spans.get(code)
            if not gold_spans:
                continue
            considered += 1
            span = result.pred_spans.get(code)
            if span is None:
                continue
            cited += 1
            best = max((_iou(span, gold) for gold in gold_spans), default=0.0)
            ious.append(best)
            if best > 0:
                overlapping += 1

    if not considered:
        return {"evidence_coverage": 0.0, "evidence_hit_rate": 0.0, "evidence_mean_iou": 0.0, "n_scored": 0}

    return {
        # of correct codes that have gold evidence, how many did the model cite at all
        "evidence_coverage": cited / considered,
        # of those cited, how many touched the right text
        "evidence_hit_rate": overlapping / cited if cited else 0.0,
        "evidence_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "n_scored": considered,
    }


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union else 0.0


def operational(results: Sequence[NoteResult]) -> dict[str, float]:
    latencies = [r.latency_s for r in results if r.latency_s is not None]
    return {
        "n_notes": len(results),
        "truncation_rate": sum(r.truncated for r in results) / len(results) if results else 0.0,
        "latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def bootstrap_ci(
    results: Sequence[NoteResult],
    statistic="micro_f1",
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over notes. Returns (point, low, high).

    Resampling is over notes, not over (note, code) decisions, because codes
    within a note are not independent.
    """
    fn = _STATISTICS[statistic] if isinstance(statistic, str) else statistic
    point = fn(results)
    if not results:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = len(results)
    samples = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        samples[i] = fn([results[j] for j in idx])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(point), float(low), float(high)


def paired_bootstrap(
    a: Sequence[NoteResult],
    b: Sequence[NoteResult],
    statistic="micro_f1",
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Is a better than b, on the same notes, beyond resampling noise.

    The leaderboard may only call a difference real when the interval excludes
    zero. Both runs must cover the same note ids, or the pairing is a lie.
    """
    by_id_a = {r.note_id: r for r in a}
    by_id_b = {r.note_id: r for r in b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if len(shared) != len(by_id_a) or len(shared) != len(by_id_b):
        raise ValueError(
            f"Paired comparison needs identical note sets: {len(by_id_a)} vs {len(by_id_b)}, "
            f"{len(shared)} shared"
        )

    fn = _STATISTICS[statistic] if isinstance(statistic, str) else statistic
    rows_a = [by_id_a[i] for i in shared]
    rows_b = [by_id_b[i] for i in shared]
    point = fn(rows_a) - fn(rows_b)

    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        deltas[i] = fn([rows_a[j] for j in idx]) - fn([rows_b[j] for j in idx])
    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": bool(low > 0 or high < 0),
    }


_STATISTICS = {
    "micro_f1": lambda rs: micro_f1(rs)[2],
    "macro_f1": lambda rs: macro_f1(rs)[2],
    "sample_f1": lambda rs: sample_f1(rs)[2],
    "exact_match": exact_match_ratio,
    "jaccard": jaccard,
}


def evaluate(results: Sequence[NoteResult], candidates: set[str] | None = None) -> dict:
    """The full report for one run."""
    micro = micro_f1(results)
    macro = macro_f1(results)
    sample = sample_f1(results)

    micro_point, micro_low, micro_high = bootstrap_ci(results, "micro_f1")
    macro_point, macro_low, macro_high = bootstrap_ci(results, "macro_f1")

    report = {
        "core": {
            "micro_precision": micro[0],
            "micro_recall": micro[1],
            "micro_f1": micro[2],
            "macro_precision": macro[0],
            "macro_recall": macro[1],
            "macro_f1": macro[2],
            "sample_f1": sample[2],
            "exact_match_ratio": exact_match_ratio(results),
            "jaccard": jaccard(results),
            "label_cardinality_ratio": label_cardinality_ratio(results),
        },
        "uncertainty": {
            "micro_f1_ci95": [micro_low, micro_high],
            "macro_f1_ci95": [macro_low, macro_high],
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "bands": band_table(results),
        "evidence": evidence_scores(results),
        "operational": operational(results),
    }
    per_note_candidates = [r.candidates for r in results if r.candidates is not None]
    if candidates or per_note_candidates:
        sizes = [len(c) for c in per_note_candidates]
        report["scaling"] = {
            "candidate_space_size": (
                float(np.mean(sizes)) if sizes else len(candidates or ())
            ),
            "distractor_fp_rate": distractor_false_positive_rate(results, candidates),
        }
    return report


def format_band_table(table: dict[str, dict]) -> str:
    """Markdown for the band table, for the leaderboard and for run summaries."""
    lines = [
        "| Band | Codes | Gold mentions | Micro P | Micro R | Micro F1 | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for band in ("head", "torso", "tail"):
        row = table.get(band)
        if not row:
            continue
        lines.append(
            f"| {band} | {row['n_codes']} | {row['n_gold_mentions']} | "
            f"{row['micro_precision']:.3f} | {row['micro_recall']:.3f} | "
            f"{row['micro_f1']:.3f} | {row['macro_f1']:.3f} |"
        )
    return "\n".join(lines)
