"""The checks that make the August 2026 leak structurally impossible to repeat.

A parquet of MIMIC discharge summaries reached a public repo once. These tests
cover the two mechanisms that stop it happening again: the pre-commit and CI
scan over changed files, and the run record writer that refuses to serialise
note text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_restricted  # noqa: E402

from coding_bench.bench.runner import ResultWriterError, check_record, save_run  # noqa: E402

# Assembled at runtime so this file does not itself contain a run of markers,
# which is exactly what the scanner is looking for.
MARKER = "[" + "**" + "Known lastname 4291" + "**" + "]"
DATE_MARKER = "[" + "**" + "2101-4-6" + "**" + "]"


def test_parquet_is_rejected_whatever_it_contains(tmp_path):
    path = tmp_path / "all_icd10.parquet"
    path.write_bytes(b"PAR1")
    assert check_restricted.check([str(path)])


def test_pickle_is_rejected(tmp_path):
    path = tmp_path / "cache.pkl"
    path.write_bytes(b"\x80\x04")
    assert check_restricted.check([str(path)])


def test_oversized_file_is_rejected(tmp_path):
    path = tmp_path / "big.json"
    path.write_bytes(b"x" * (check_restricted.MAX_BYTES + 1))
    assert check_restricted.check([str(path)])


def test_deid_markers_are_caught_in_any_container(tmp_path):
    """This is the check that matters. Note text is note text in any file type."""
    for name in ("notes.json", "fixture.py", "analysis.md", "notebook.ipynb", "data.csv"):
        path = tmp_path / name
        path.write_text(
            f"Patient {MARKER} was admitted on {DATE_MARKER} with chest pain.",
            encoding="utf8",
        )
        problems = check_restricted.check([str(path)])
        assert problems, f"{name} should have been rejected"
        assert "de-identification" in problems[0]


def test_a_single_marker_in_prose_is_allowed(tmp_path):
    """Documentation has to be able to describe the pattern it is guarding against."""
    path = tmp_path / "README.md"
    path.write_text(f"MIMIC replaces names with markers such as {MARKER}.", encoding="utf8")
    assert not check_restricted.check([str(path)])


def test_ordinary_source_passes(tmp_path):
    path = tmp_path / "metrics.py"
    path.write_text("def f1(p, r):\n    return 2 * p * r / (p + r)\n", encoding="utf8")
    assert not check_restricted.check([str(path)])


def test_the_committed_tree_is_clean():
    """The repo as it stands must pass its own scanner."""
    root = Path(__file__).resolve().parent.parent
    paths = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".venv" not in path.parts
    ]
    assert not check_restricted.check(paths)


def test_result_writer_rejects_note_text():
    with pytest.raises(ResultWriterError, match="may not carry"):
        check_record({"note_id": "1", "gold": [], "pred": [], "text": "Patient is a 58 year old"})


def test_result_writer_rejects_every_text_shaped_key():
    for key in ("note_text", "covered_text", "raw", "prompt", "response"):
        with pytest.raises(ResultWriterError):
            check_record({"note_id": "1", key: "anything"})


def test_result_writer_rejects_long_strings_under_an_innocent_key():
    """Renaming the field must not be a way around the rule."""
    with pytest.raises(ResultWriterError, match="note shaped"):
        check_record({"note_id": "1", "notes_for_reviewer": "x" * 501})


def test_provider_errors_are_stripped_of_account_identifiers():
    """Run records are committed, so a quoted provider error must not carry ids."""
    from coding_bench.bench.runner import sanitise_error

    message = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        "`org_01k9wzfvafepdbz6wh2rvcaw2m` service tier `on_demand`"
    )
    cleaned = sanitise_error(message)
    assert "org_01k9wzfvafepdbz6wh2rvcaw2m" not in cleaned
    assert "<account>" in cleaned
    # The useful part survives, so the record is still diagnosable.
    assert "Rate limit reached" in cleaned


def test_provider_errors_are_length_capped():
    from coding_bench.bench.runner import sanitise_error

    assert len(sanitise_error("x" * 5000)) == 200
    assert sanitise_error(None) is None


def test_result_writer_accepts_a_legitimate_record():
    record = {
        "note_id": "12345",
        "gold": ["I10", "E11.9"],
        "pred": ["I10"],
        "pred_spans": {"I10": [10, 32]},
        "latency_s": 1.4,
        "truncated": False,
    }
    assert check_record(record) == record


def test_save_run_checks_every_row(tmp_path):
    record = {
        "manifest": {"run_id": "test-run"},
        "metrics": {},
        "predictions": [
            {"note_id": "1", "gold": [], "pred": []},
            {"note_id": "2", "text": "leaked"},
        ],
    }
    with pytest.raises(ResultWriterError):
        save_run(record, runs_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))
