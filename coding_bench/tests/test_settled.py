"""The cell level skip: what counts as settled, and what must not.

A false negative here costs money, which is the whole point of the check. A
false positive is worse: it would read a number off a record that the current
code could no longer reproduce, and nothing downstream would notice.
"""

from __future__ import annotations

import json

import pytest

modal = pytest.importorskip("modal", reason="settled imports the adapter registry")

from coding_bench.bench.settled import settled_record  # noqa: E402


def write_record(runs_dir, *, model_id, task, space, adapter_sha, max_tokens=16384,
                 reasoning_strength="medium", error_rate=0.0, run_id="r1"):
    record = {
        "manifest": {
            "run_id": run_id, "task": task, "model_id": model_id,
            "candidate_space": space, "n_notes": 312, "adapter_sha256": adapter_sha,
            "parameters": {"max_tokens": max_tokens, "reasoning_strength": reasoning_strength},
        },
        "metrics": {"operational": {"error_rate": error_rate}},
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(record), encoding="utf8")


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


def current_hash():
    from pathlib import Path

    from coding_bench.bench.runner import file_sha256
    from coding_bench.bench.settled import ADAPTER_DIR

    return file_sha256(Path(ADAPTER_DIR) / "api_groq.py")


def test_a_matching_valid_record_settles_the_cell(runs_dir):
    write_record(runs_dir, model_id="qwen/qwen3.6-27b", task="cpt", space="gold",
                 adapter_sha=current_hash())
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir)


def test_an_edited_adapter_unsettles_it(runs_dir):
    """The case the note level cache also catches, caught earlier and cheaper."""
    write_record(runs_dir, model_id="qwen/qwen3.6-27b", task="cpt", space="gold",
                 adapter_sha="0" * 64)
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir) is None


def test_a_quarantined_record_does_not_settle_anything(runs_dir):
    """It has no usable number, so it is worth paying to try again."""
    write_record(runs_dir, model_id="qwen/qwen3.6-27b", task="cpt", space="gold",
                 adapter_sha=current_hash(), error_rate=0.44)
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir) is None


@pytest.mark.parametrize("field,value", [("max_tokens", 8192), ("reasoning_strength", "low")])
def test_a_different_parameter_unsettles_it(runs_dir, field, value):
    """Both are in the prediction cache key, so both change the answer."""
    write_record(runs_dir, model_id="qwen/qwen3.6-27b", task="cpt", space="gold",
                 adapter_sha=current_hash(), **{field: value})
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir) is None


def test_the_wrong_cell_never_settles_another(runs_dir):
    """Two models share api_groq.py, so the hash alone cannot identify a row."""
    write_record(runs_dir, model_id="openai/gpt-oss-120b", task="cpt", space="gold",
                 adapter_sha=current_hash())
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir) is None
    assert settled_record("cpt", "gpt-oss-120b", "gold", 16384, runs_dir=runs_dir)


def test_an_empty_runs_dir_settles_nothing(runs_dir):
    assert settled_record("cpt", "qwen3.6-27b", "gold", 16384, runs_dir=runs_dir) is None
