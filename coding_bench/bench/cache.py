"""A per note prediction cache, so nothing is paid for twice.

The unit of caching is one note under one exact configuration, not one run. That
matters because runs get interrupted, rate limited, and rerun constantly, and a
run level cache would throw away 400 good notes because 20 failed.

The key covers everything that can change an answer:

  the dataset      via the gold manifest checksum, so rebuilding the gold set
                   invalidates every entry derived from it
  the note         via its id inside that dataset
  the model        via its id
  the code         via the adapter file hash and the prompt hash
  the parameters   temperature, max tokens, reasoning strength, and the exact
                   candidate list the note was offered

Change any of those and the key changes, so a rerun genuinely re-runs. Change
none of them and the answer cannot differ, so paying a provider again buys
nothing.

**Failures are never cached.** A rate limit or a timeout is a fact about the
afternoon, not about the model, and caching it would freeze a transient error
into the results permanently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

CACHE_NAME = "coding-bench-predictions"


def cache_key(
    *,
    dataset_manifest_sha256: str | None,
    note_id: str,
    model_id: str,
    approach: str,
    approach_version: str,
    adapter_sha256: str | None,
    prompt_sha256: str | None,
    candidates: set[str],
    parameters: dict,
) -> str:
    """A stable digest of everything that determines the answer."""
    payload = {
        "dataset": dataset_manifest_sha256,
        "note_id": note_id,
        "model_id": model_id,
        "approach": approach,
        "approach_version": approach_version,
        "adapter": adapter_sha256,
        "prompt": prompt_sha256,
        # The offered list, not just its size: two notes at candidate space 200
        # get different distractors and can legitimately answer differently.
        "candidates": sorted(candidates),
        "parameters": {k: parameters[k] for k in sorted(parameters)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, value: Any) -> None: ...


class MemoryCache:
    """For tests and for Tier 0, where nothing costs anything."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        value = self._store.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value


class ModalDictCache:
    """Persistent across runs, shared across everyone in the workspace."""

    def __init__(self, name: str = CACHE_NAME):
        import modal

        self._dict = modal.Dict.from_name(name, create_if_missing=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        try:
            value = self._dict[key]
        except KeyError:
            self.misses += 1
            return None
        except Exception:  # noqa: BLE001
            # A cache that is misbehaving must never fail a run. Treat any
            # trouble as a miss and pay for the call.
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        try:
            self._dict[key] = value
        except Exception:  # noqa: BLE001
            pass


class NullCache:
    """Explicitly disabled, for when a rerun must actually re-run."""

    hits = 0
    misses = 0

    def get(self, key: str):
        return None

    def put(self, key: str, value: Any) -> None:
        return None


class RecordSeededCache:
    """A cache warmed from the committed run records, backed by a live store.

    The run records already hold every per note prediction, keyed by everything
    the cache key needs, and they are committed to git. So a fresh checkout on a
    CI runner has thousands of paid answers sitting right there, and asking the
    provider for them again is buying something already owned.

    Reads check the seeded layer first and fall through to the live store.
    Writes go only to the live store: the records are written by the runner at
    the end of a run and are not ours to edit.
    """

    def __init__(self, seed: dict[str, dict], backing):
        self._seed = seed
        self._backing = backing
        self.hits = 0
        self.misses = 0
        self.seeded = len(seed)

    def get(self, key: str):
        value = self._seed.get(key)
        if value is not None:
            self.hits += 1
            return value
        value = self._backing.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value) -> None:
        self._backing.put(key, value)


def seed_from_records(runs_dir, manifest_lookup=None) -> dict[str, dict]:
    """Rebuild cache entries from committed run records.

    A record stores the run manifest once and the predictions per note, which is
    exactly the material the key is built from. Failed notes are skipped: an
    error describes an afternoon, not a model, and must always be retried.
    """
    import json
    from pathlib import Path

    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return {}

    seed: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf8"))
        except (json.JSONDecodeError, OSError):
            continue

        manifest = record.get("manifest", {})
        parameters = dict(manifest.get("parameters") or {})
        # Mirror the runner: parameters that steer execution rather than the
        # answer are excluded from the key.
        parameters.pop("concurrency", None)

        for row in record.get("predictions", []):
            if row.get("error") or "n_candidates" not in row:
                continue
            # The offered list is not stored per note, only its size, so a
            # record can only seed runs whose candidate set is reconstructible.
            candidates = manifest_lookup(manifest, row) if manifest_lookup else None
            if candidates is None:
                continue
            key = cache_key(
                dataset_manifest_sha256=manifest.get("dataset_manifest_sha256"),
                note_id=row["note_id"],
                model_id=manifest.get("model_id", ""),
                approach=manifest.get("approach", ""),
                approach_version=manifest.get("approach_version", ""),
                adapter_sha256=manifest.get("adapter_sha256"),
                prompt_sha256=manifest.get("prompt_sha256"),
                candidates=candidates,
                parameters=parameters,
            )
            seed[key] = {
                "codes": {
                    code: (list(span) if span else None)
                    for code, span in (row.get("pred_spans") or {}).items()
                }
                | {code: None for code in row.get("pred", []) if code not in (row.get("pred_spans") or {})},
                "truncated": bool(row.get("truncated")),
                "usage": row.get("usage") or {},
                "latency_s": row.get("latency_s") or 0.0,
            }
    return seed


def open_cache(kind: str = "auto", seed: dict[str, dict] | None = None):
    """Pick a backend. Falls back rather than failing when Modal is absent."""
    if kind == "off":
        return NullCache()
    if kind == "memory":
        backing = MemoryCache()
    else:
        try:
            backing = ModalDictCache()
        except Exception as exc:  # noqa: BLE001
            print(f"Prediction cache unavailable ({exc}), running without it", flush=True)
            backing = MemoryCache()

    if seed:
        print(f"Cache seeded with {len(seed)} predictions from committed run records", flush=True)
        return RecordSeededCache(seed, backing)
    return backing
