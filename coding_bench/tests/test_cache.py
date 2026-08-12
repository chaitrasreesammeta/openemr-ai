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
