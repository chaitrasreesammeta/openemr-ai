"""End to end checks over the Tier 0 set, plus the response parsing.

No network, no GPU, no restricted data. This is what public CI runs.
"""

from __future__ import annotations

import json
import time

import pytest

from coding_bench.approaches.base import Candidate, Note, Prediction, Truncated, locate_span
from coding_bench.approaches.llm import balanced_objects, extract_json, normalise_code, parse_codes
from coding_bench.bench import loaders, runner


# --------------------------------------------------------------------------
# Tier 0 loading


def test_smoke_set_loads_for_both_code_systems():
    for system, expected in (("icd10", "ICD-10-CM"), ("cpt", "CPT")):
        dataset = loaders.load_smoke(system)
        assert dataset.tier == 0
        assert dataset.code_system == expected
        assert len(dataset) == 20
        assert dataset.manifest_sha256


def test_every_smoke_note_has_gold_and_evidence():
    dataset = loaders.load_smoke("icd10")
    for note in dataset.notes:
        assert note.gold_codes, f"{note.note_id} has no gold codes"
        for code in note.gold_codes:
            assert code in note.gold_spans, f"{note.note_id} has no evidence for {code}"


def test_smoke_evidence_offsets_point_at_real_text():
    """An offset that does not land on the note would corrupt the evidence metric."""
    dataset = loaders.load_smoke("icd10")
    for note in dataset.notes:
        for code, spans in note.gold_spans.items():
            for begin, end in spans:
                assert 0 <= begin < end <= len(note.text)
                assert note.text[begin:end].strip()


def test_smoke_gold_codes_are_inside_the_label_space():
    for system in ("icd10", "cpt"):
        dataset = loaders.load_smoke(system)
        assert dataset.gold_codes <= set(dataset.label_space)


def test_load_dispatches_by_task_name():
    assert loaders.load("smoke_cpt").code_system == "CPT"
    with pytest.raises(loaders.DataError, match="Unknown task"):
        loaders.load("not_a_task")


# --------------------------------------------------------------------------
# Candidate spaces


def test_gold_only_candidate_space_offers_exactly_the_answers():
    note = Note(note_id="n1", text="x", gold_codes=("A", "B"))
    candidates = runner.build_candidates(note, {"A": None, "B": None, "C": None}, "gold")
    assert {c.code for c in candidates} == {"A", "B"}


def test_sized_candidate_space_always_contains_gold():
    note = Note(note_id="n1", text="x", gold_codes=("A", "B"))
    space = {f"C{i}": None for i in range(500)} | {"A": None, "B": None}
    candidates = runner.build_candidates(note, space, 50)
    codes = {c.code for c in candidates}
    assert len(codes) == 50
    assert {"A", "B"} <= codes


def test_full_candidate_space_is_the_whole_catalogue():
    note = Note(note_id="n1", text="x", gold_codes=("A",))
    space = {f"C{i}": None for i in range(30)} | {"A": None}
    assert len(runner.build_candidates(note, space, None)) == 31


def test_candidate_choice_is_reproducible_across_runs():
    note = Note(note_id="n1", text="x", gold_codes=("A",))
    space = {f"C{i}": None for i in range(200)} | {"A": None}
    first = [c.code for c in runner.build_candidates(note, space, 20)]
    second = [c.code for c in runner.build_candidates(note, space, 20)]
    assert first == second


def test_different_notes_get_different_distractors():
    space = {f"C{i}": None for i in range(200)} | {"A": None}
    a = {c.code for c in runner.build_candidates(Note("n1", "x", ("A",)), space, 20)}
    b = {c.code for c in runner.build_candidates(Note("n2", "x", ("A",)), space, 20)}
    assert a != b


# --------------------------------------------------------------------------
# Response parsing


def test_extract_json_from_a_bare_object():
    assert extract_json('{"codes": []}') == {"codes": []}


def test_extract_json_from_a_fenced_block_after_reasoning():
    response = 'Let me think about this.\n\n```json\n{"codes": [{"code": "I10"}]}\n```\n'
    assert extract_json(response) == {"codes": [{"code": "I10"}]}


def test_extract_json_survives_prose_around_the_answer():
    response = 'Here is my answer: {"codes": [{"code": "I10", "quote": "hypertension"}]} Hope that helps.'
    assert extract_json(response)["codes"][0]["code"] == "I10"


def test_extract_json_takes_the_answer_not_the_thinking():
    """A reasoning model restates JSON while thinking. The last one is the answer."""
    response = (
        'We need to code this note. Maybe {"codes": [{"code": "I10"}]} is right.\n'
        'Actually the note also documents diabetes.\n'
        'Final answer: {"codes": [{"code": "I10"}, {"code": "E11.9"}]}'
    )
    payload = extract_json(response)
    assert [entry["code"] for entry in payload["codes"]] == ["I10", "E11.9"]


def test_extract_json_survives_a_channel_prefixed_reasoning_trace():
    """Muse Glimmer emits its trace inline, including to=self channel markers."""
    response = (
        ' to=selfThe user wants codes. Let me think.\n\n'
        'The note says hypertension. So {"ok": true} style output is wanted.\n'
        '{"codes": [{"code": "I10", "quote": "hypertension"}]}'
    )
    payload = extract_json(response)
    assert payload["codes"][0]["code"] == "I10"


def test_extract_json_handles_nested_objects():
    response = 'Thinking...\n{"codes": [{"code": "I10", "meta": {"confidence": 0.9}}]}'
    assert extract_json(response)["codes"][0]["meta"]["confidence"] == 0.9


def test_extract_json_ignores_braces_inside_quoted_text():
    response = '{"codes": [{"code": "I10", "quote": "BP was 140/90 {see flowsheet}"}]}'
    assert extract_json(response)["codes"][0]["code"] == "I10"


def test_balanced_objects_finds_each_object_separately():
    assert balanced_objects('{"a": 1} noise {"b": {"c": 2}}') == ['{"a": 1}', '{"b": {"c": 2}}']


def test_extract_json_returns_none_on_garbage():
    assert extract_json("I could not determine any codes.") is None
    assert extract_json("") is None


def test_parse_codes_locates_the_quoted_span():
    text = "Assessment: Essential hypertension, above goal. Continue lisinopril."
    codes = parse_codes('{"codes":[{"code":"I10","quote":"Essential hypertension"}]}', text)
    begin, end = codes["I10"]
    assert text[begin:end] == "Essential hypertension"


def test_parse_codes_reports_no_span_when_the_quote_is_invented():
    """A hallucinated quote must not become a plausible looking offset."""
    text = "Assessment: acute bronchitis."
    codes = parse_codes('{"codes":[{"code":"J20.9","quote":"pneumonia on chest x-ray"}]}', text)
    assert codes == {"J20.9": None}


def test_parse_codes_accepts_a_plain_list_of_codes():
    assert set(parse_codes('{"codes":["I10","E11.9"]}', "text")) == {"I10", "E11.9"}


def test_parse_codes_returns_empty_for_an_unparseable_response():
    assert parse_codes("no codes apply", "text") == {}


def test_parse_codes_restricted_to_candidates_drops_outsiders():
    codes = parse_codes('{"codes":["I10","Z99.9"]}', "text", restrict_to={"I10"})
    assert set(codes) == {"I10"}


def test_normalise_repairs_a_missing_decimal_point_only_against_the_offered_list():
    assert normalise_code("E119", {"E11.9"}) == "E11.9"
    # With nothing offered it must not invent a format.
    assert normalise_code("E119", None) == "E119"
    # And it must never conjure a code that was not offered.
    assert normalise_code("X999", {"E11.9"}) == "X999"


def test_locate_span_tolerates_whitespace_differences():
    text = "Assessment:\n1. Acute  bronchitis\n"
    span = locate_span(text, "Acute bronchitis")
    assert span is None or text[span[0] : span[1]].strip()


# --------------------------------------------------------------------------
# The evaluation loop


class PerfectPredictor:
    name, version, model_id = "oracle", "v1", "none"

    def predict(self, note, candidates=None):
        return Prediction(codes={code: note.gold_spans.get(code, [(0, 1)])[0] for code in note.gold_codes})


class SilentPredictor:
    name, version, model_id = "silent", "v1", "none"

    def predict(self, note, candidates=None):
        return Prediction(codes={})


class TruncatingPredictor:
    name, version, model_id = "truncating", "v1", "test-model"

    def predict(self, note, candidates=None):
        raise Truncated("test-model", "length", produced_tokens=4096)


class ExplodingPredictor:
    name, version, model_id = "exploding", "v1", "none"

    def predict(self, note, candidates=None):
        raise ValueError("provider returned nonsense")


def test_a_perfect_run_scores_one():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(PerfectPredictor(), dataset, candidate_space="gold", progress=False)
    assert record["metrics"]["core"]["micro_f1"] == 1.0
    assert record["metrics"]["core"]["exact_match_ratio"] == 1.0
    assert record["metrics"]["evidence"]["evidence_hit_rate"] == 1.0


def test_a_silent_run_scores_zero_without_crashing():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(SilentPredictor(), dataset, candidate_space="gold", progress=False)
    assert record["metrics"]["core"]["micro_f1"] == 0.0
    assert record["metrics"]["operational"]["truncation_rate"] == 0.0


def test_truncation_is_recorded_and_never_looks_like_an_empty_answer():
    dataset = loaders.load_smoke("icd10")
    truncated = runner.run(TruncatingPredictor(), dataset, candidate_space="gold", progress=False)
    silent = runner.run(SilentPredictor(), dataset, candidate_space="gold", progress=False)

    assert truncated["metrics"]["core"]["micro_f1"] == silent["metrics"]["core"]["micro_f1"] == 0.0
    # Same score, and yet the reports must not be mistakable for each other.
    assert truncated["metrics"]["operational"]["truncation_rate"] == 1.0
    assert silent["metrics"]["operational"]["truncation_rate"] == 0.0


def test_a_failing_predictor_is_recorded_rather_than_killing_the_run():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(ExplodingPredictor(), dataset, candidate_space="gold", limit=3, progress=False)
    assert record["metrics"]["operational"]["error_rate"] == 1.0
    assert all("provider returned nonsense" in row["error"] for row in record["predictions"])


def test_run_manifest_carries_everything_needed_to_reproduce_it():
    dataset = loaders.load_smoke("cpt")
    record = runner.run(
        PerfectPredictor(), dataset, candidate_space=200, limit=2,
        parameters={"temperature": 0.0}, prompt_text="a prompt", progress=False,
    )
    manifest = record["manifest"]
    for key in ("run_id", "task", "git_sha", "dataset_manifest_sha256", "prompt_sha256", "candidate_space"):
        assert manifest[key], f"{key} is missing from the run manifest"
    assert manifest["parameters"] == {"temperature": 0.0}


def test_run_records_never_contain_note_text():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(PerfectPredictor(), dataset, candidate_space="gold", limit=5, progress=False)
    serialised = json.dumps(record)
    for note in dataset.notes[:5]:
        assert note.text[:60] not in serialised


def test_distractors_are_scored_when_the_candidate_space_grows():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(SilentPredictor(), dataset, candidate_space=20, limit=5, progress=False)
    assert record["metrics"]["scaling"]["candidate_space_size"] == 20
    assert record["metrics"]["scaling"]["distractor_fp_rate"] == 0.0


class SlowJitteryPredictor:
    """Finishes notes out of order, to prove ordering does not depend on timing."""

    name, version, model_id = "jittery", "v1", "none"

    def predict(self, note, candidates=None):
        # Later note ids return sooner, so completion order is reversed.
        time.sleep(0.02 * (20 - int(note.note_id.split("-")[1])))
        return Prediction(codes={code: None for code in note.gold_codes})


def test_concurrent_run_matches_the_sequential_one_exactly():
    """Concurrency is a speed change, never a results change."""
    dataset = loaders.load_smoke("icd10")
    sequential = runner.run(PerfectPredictor(), dataset, candidate_space="gold", progress=False)
    parallel = runner.run(
        PerfectPredictor(), dataset, candidate_space="gold", concurrency=8, progress=False
    )
    assert sequential["metrics"]["core"] == parallel["metrics"]["core"]
    assert [r["note_id"] for r in sequential["predictions"]] == [
        r["note_id"] for r in parallel["predictions"]
    ]


def test_concurrent_results_stay_in_dataset_order_despite_completion_order():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(
        SlowJitteryPredictor(), dataset, candidate_space="gold", concurrency=8, progress=False
    )
    ids = [r["note_id"] for r in record["predictions"]]
    assert ids == sorted(ids)
    # And every note is still scored against its own gold, not a neighbour's.
    for row, note in zip(record["predictions"], dataset.notes):
        assert row["note_id"] == note.note_id
        assert row["gold"] == sorted(note.gold_codes)


def test_concurrent_run_still_records_failures_per_note():
    dataset = loaders.load_smoke("icd10")
    record = runner.run(
        ExplodingPredictor(), dataset, candidate_space="gold", limit=6, concurrency=4, progress=False
    )
    assert record["metrics"]["operational"]["error_rate"] == 1.0
    assert len(record["predictions"]) == 6


def test_save_and_summarise_a_run(tmp_path):
    dataset = loaders.load_smoke("icd10")
    record = runner.run(PerfectPredictor(), dataset, candidate_space="gold", limit=3, progress=False)
    path = runner.save_run(record, runs_dir=tmp_path)
    assert path.exists()
    assert json.loads(path.read_text())["manifest"]["run_id"] == record["manifest"]["run_id"]
    assert "micro F1 1.000" in runner.summarise(record)
