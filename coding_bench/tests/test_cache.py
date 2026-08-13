"""The prediction cache must save money without ever changing an answer.

Two failure modes would be worse than having no cache at all: serving a stale
answer after something that matters has changed, and remembering a transient
provider failure as though it were a model result.
"""

from __future__ import annotations

import pytest

from coding_bench.approaches.base import Prediction, Truncated
from coding_bench.bench import cache as cache_module
from coding_bench.bench import loaders, runner

BASE = dict(
    dataset_manifest_sha256="dataset-a",
    note_id="1",
    model_id="model-a",
    approach="llm",
    approach_version="v1",
    adapter_sha256="adapter-a",
    prompt_sha256="prompt-a",
    candidates={"A", "B"},
    parameters={"max_tokens": 4096},
)


def test_identical_configuration_gives_the_same_key():
    assert cache_module.cache_key(**BASE) == cache_module.cache_key(**BASE)


def test_candidate_order_does_not_change_the_key():
    other = BASE | {"candidates": {"B", "A"}}
    assert cache_module.cache_key(**other) == cache_module.cache_key(**BASE)


@pytest.mark.parametrize(
    "field, value",
    [
        ("dataset_manifest_sha256", "dataset-b"),
        ("note_id", "2"),
        ("model_id", "model-b"),
        ("approach_version", "v2"),
        ("adapter_sha256", "adapter-b"),
        ("prompt_sha256", "prompt-b"),
        ("candidates", {"A", "B", "C"}),
        ("parameters", {"max_tokens": 8192}),
    ],
)
def test_anything_that_can_change_the_answer_changes_the_key(field, value):
    assert cache_module.cache_key(**(BASE | {field: value})) != cache_module.cache_key(**BASE)


class CountingPredictor:
    """Counts how many times the provider would actually have been paid."""

    name, version, model_id = "counting", "v1", "counted-model"

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def predict(self, note, candidates=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("429 rate limited")
        return Prediction(codes={code: None for code in note.gold_codes})


@pytest.fixture
def shared_store(monkeypatch):
    """Pin one cache across runs, the way the Modal Dict persists in production."""
    store = cache_module.MemoryCache()
    monkeypatch.setattr(cache_module, "open_cache", lambda kind="auto", seed=None: store)
    return store


def test_second_run_costs_nothing(shared_store):
    dataset = loaders.load_smoke("icd10")
    predictor = CountingPredictor()

    first = runner.run(predictor, dataset, cache="memory", progress=False)
    assert predictor.calls == len(dataset)

    runner.run(predictor, dataset, cache="memory", progress=False)
    assert predictor.calls == len(dataset), "a warm cache must not pay the provider again"
    assert first["metrics"]["core"]["micro_f1"] == 1.0


def test_a_cached_run_scores_identically_to_a_fresh_one(shared_store):
    dataset = loaders.load_smoke("icd10")
    fresh = runner.run(CountingPredictor(), dataset, cache="memory", progress=False)
    warm = runner.run(CountingPredictor(), dataset, cache="memory", progress=False)

    assert fresh["metrics"]["core"] == warm["metrics"]["core"]
    assert warm["metrics"]["operational"]["cache_hits"] == len(dataset)


def test_failures_are_never_cached_so_a_rerun_retries_them(shared_store):
    """Tonight's rate limit must not become a permanent zero."""
    dataset = loaders.load_smoke("icd10")
    predictor = CountingPredictor(fail_first=len(dataset))

    failed = runner.run(predictor, dataset, cache="memory", progress=False)
    assert failed["metrics"]["operational"]["error_rate"] == 1.0

    # Every note failed, so nothing was cached and the retry does real work.
    retried = runner.run(predictor, dataset, cache="memory", progress=False)

    assert retried["metrics"]["operational"]["error_rate"] == 0.0
    assert retried["metrics"]["core"]["micro_f1"] == 1.0


def test_cache_off_always_recomputes():
    dataset = loaders.load_smoke("icd10")
    predictor = CountingPredictor()
    runner.run(predictor, dataset, cache="off", limit=5, progress=False)
    runner.run(predictor, dataset, cache="off", limit=5, progress=False)
    assert predictor.calls == 10


def test_an_unreachable_modal_dict_degrades_to_no_cache(monkeypatch):
    """A cache outage must slow a run down, never fail it."""
    def explode(name=cache_module.CACHE_NAME):
        raise RuntimeError("modal is unreachable")

    monkeypatch.setattr(cache_module, "ModalDictCache", explode)
    store = cache_module.open_cache("auto")
    # Degrades to an in process store rather than failing the run. It will not
    # persist across runs, but the run itself still completes.
    assert isinstance(store, (cache_module.NullCache, cache_module.MemoryCache))


def test_null_cache_reports_nothing_stored():
    store = cache_module.NullCache()
    store.put("k", {"codes": {}})
    assert store.get("k") is None


# --------------------------------------------------------------------------
# Adapter equivalences


def seed_record(tmp_path, adapter: str, rows: list[dict]) -> dict:
    """One run record on disk, and the seed rebuilt from it."""
    import json

    record = {
        "manifest": {
            "run_id": "r",
            "task": "cpt",
            "model_id": "model-a",
            "approach": "llm",
            "approach_version": "v1",
            "candidate_space": "gold",
            "dataset_manifest_sha256": "dataset-a",
            "adapter_sha256": adapter,
            "prompt_sha256": "prompt-a",
            "parameters": {"max_tokens": 4096, "concurrency": 4},
        },
        "predictions": rows,
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf8")
    return cache_module.seed_from_records(tmp_path, manifest_lookup=lambda m, row: {"A", "B"})


def key_for(adapter: str, note_id: str) -> str:
    return cache_module.cache_key(**(BASE | {"adapter_sha256": adapter, "note_id": note_id}))


def test_an_equivalence_carries_an_answered_note_onto_the_new_adapter(monkeypatch, tmp_path):
    """The point of the whole mechanism: do not pay twice for an unchanged answer."""
    monkeypatch.setattr(
        cache_module,
        "ADAPTER_EQUIVALENCES",
        (cache_module.AdapterEquivalence(before="old", after="new", why="test"),),
    )
    seed = seed_record(tmp_path, "old", [
        {"note_id": "1", "pred": ["A"], "pred_spans": {}, "n_candidates": 2, "usage": {}},
    ])
    assert key_for("old", "1") in seed
    carried = seed[key_for("new", "1")]
    assert carried["codes"] == {"A": None}
    assert carried["carried_from"] == "old", "a borrowed answer has to say so"


@pytest.mark.parametrize(
    "row, why",
    [
        ({"note_id": "1", "pred": [], "n_candidates": 2}, "silent notes are what the fix was for"),
        ({"note_id": "1", "pred": ["A"], "truncated": True, "n_candidates": 2}, "truncated"),
    ],
)
def test_an_equivalence_refuses_to_carry_a_failed_note(monkeypatch, tmp_path, row, why):
    monkeypatch.setattr(
        cache_module,
        "ADAPTER_EQUIVALENCES",
        (cache_module.AdapterEquivalence(before="old", after="new", why="test"),),
    )
    seed = seed_record(tmp_path, "old", [row])
    assert key_for("new", "1") not in seed, why


def test_an_equivalence_only_applies_to_the_adapter_it_names(monkeypatch, tmp_path):
    """Otherwise one declaration would quietly excuse every adapter in the repo."""
    monkeypatch.setattr(
        cache_module,
        "ADAPTER_EQUIVALENCES",
        (cache_module.AdapterEquivalence(before="old", after="new", why="test"),),
    )
    seed = seed_record(tmp_path, "unrelated", [
        {"note_id": "1", "pred": ["A"], "pred_spans": {}, "n_candidates": 2, "usage": {}},
    ])
    assert key_for("new", "1") not in seed
    assert key_for("unrelated", "1") in seed


def test_carried_hits_are_counted_separately_from_ordinary_ones():
    seed = {
        "borrowed": {"codes": {"A": None}, "carried_from": "old"},
        "ours": {"codes": {"B": None}},
    }
    store = cache_module.RecordSeededCache(seed, cache_module.MemoryCache())
    store.get("borrowed")
    store.get("ours")
    store.get("absent")
    assert (store.hits, store.misses, store.carried) == (2, 1, 1)


def test_the_declared_equivalences_name_the_adapters_that_are_actually_here():
    """A stale `after` hash is the failure mode this table has.

    `before` may name an adapter that was never pushed, which is the situation
    it exists for. `after` must be a file in this checkout, because the whole
    guarantee is that these bytes are the ones that will run.
    """
    from pathlib import Path
    import hashlib

    here = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(cache_module.__file__).parent.parent / "adapters").glob("*.py")
    }
    for rule in cache_module.ADAPTER_EQUIVALENCES:
        assert rule.after in here, f"{rule.why}: `after` names no adapter in this checkout"


# --------------------------------------------------------------------------
# Recomputing only the empty answers


def test_recompute_empty_reruns_the_silent_notes_and_keeps_the_rest(shared_store):
    """A parser fix cannot be applied to the cache, only to a fresh response.

    The cache stores the parse, not the text, so an answer that was discarded as
    unreadable is remembered as an empty prediction and no amount of fixing the
    parser will change it. This is the flag that pays to generate exactly those
    notes again while every note that produced codes stays free.
    """
    dataset = loaders.load_smoke("icd10")

    class SometimesSilent:
        name, version, model_id = "llm", "v1", "test"

        def __init__(self, silent_on: set[str]):
            self.silent_on = silent_on
            self.calls: list[str] = []

        def predict(self, note, candidates=None):
            self.calls.append(note.note_id)
            if note.note_id in self.silent_on:
                return Prediction(codes={})
            return Prediction(codes={code: None for code in note.gold_codes})

    silent = {dataset.notes[0].note_id, dataset.notes[1].note_id}

    first = SometimesSilent(silent)
    runner.run(first, dataset, candidate_space="gold", progress=False)
    assert len(first.calls) == len(dataset)

    # Nothing changed, so nothing is asked again.
    second = SometimesSilent(silent)
    runner.run(second, dataset, candidate_space="gold", progress=False)
    assert second.calls == []

    # Now only the two that came back empty are paid for a second time.
    third = SometimesSilent(set())
    record = runner.run(
        third, dataset, candidate_space="gold", progress=False, recompute_empty=True
    )
    assert sorted(third.calls) == sorted(silent)
    assert record["metrics"]["operational"]["cache_hits"] == len(dataset) - len(silent)


def test_recompute_empty_stays_out_of_the_cache_key():
    """It steers execution, not the answer, so it must not invalidate anything."""
    assert "recompute_empty" in runner.NON_SEMANTIC_PARAMETERS
    with_flag = dict(BASE, parameters={"max_tokens": 4096})
    assert cache_module.cache_key(**with_flag) == cache_module.cache_key(**BASE)


def test_the_manifest_names_the_parser_without_the_cache_keying_on_it():
    """The reuse policy, as a test rather than as a convention.

    Reusing a paid prediction is right whenever nothing that could change the
    answer changed. A parser edit cannot change an answer already stored, so it
    must not invalidate the cache. It can change how the next response is read,
    so it has to be on the record. Both halves matter and they pull in opposite
    directions, which is why this is pinned.
    """
    import hashlib
    from pathlib import Path

    from coding_bench.approaches import llm

    dataset = loaders.load_smoke("cpt")

    class Predictor:
        name, version, model_id = "llm", "v1", "test"

        def predict(self, note, candidates=None):
            return Prediction(codes={code: None for code in note.gold_codes})

    record = runner.run(Predictor(), dataset, candidate_space="gold", progress=False)
    assert record["manifest"]["approach_sha256"] is not None

    # This test's predictor lives in the test module, so the hash is of that
    # file. What matters is that the mechanism finds the defining module.
    assert record["manifest"]["approach_sha256"] == hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    assert runner.approach_file(llm.LLMPredictor(client=None)).endswith("llm.py")

    # And the key has no room for it, so it cannot creep in by accident.
    with pytest.raises(TypeError):
        cache_module.cache_key(**BASE, approach_sha256="anything")
