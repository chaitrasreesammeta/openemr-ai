"""Tier aware dataset loading with integrity checks.

Tier 0 (smoke) is committed and loads anywhere. Tier 2 (gold) lives on the
private Modal volume and loads only where that volume is mounted. Either way the
loader refuses to hand back data it cannot match to the committed manifest, so a
stale or tampered volume fails the run instead of quietly changing the numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from coding_bench.approaches.base import Candidate, Note

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
LABEL_DIR = DATA_DIR / "labels"
SMOKE_DIR = DATA_DIR / "smoke"

# Where the gold volume is mounted inside Modal. Overridable for a local build.
GOLD_ROOT = Path(os.environ.get("CODING_BENCH_GOLD_ROOT", "/gold"))

TASKS = ("cpt", "icd10", "smoke_cpt", "smoke_icd10")


class DataError(RuntimeError):
    """Raised when data cannot be verified against the committed manifest."""


@dataclass
class Dataset:
    task: str
    code_system: str
    tier: int
    notes: list[Note]
    # code -> description, where we are allowed to have one
    label_space: dict[str, str | None]
    manifest_sha256: str | None

    @property
    def candidates(self) -> list[Candidate]:
        return [Candidate(code=code, description=desc) for code, desc in sorted(self.label_space.items())]

    @property
    def gold_codes(self) -> set[str]:
        return {code for note in self.notes for code in note.gold_codes}

    def __len__(self) -> int:
        return len(self.notes)


def _canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(task: str) -> dict:
    path = MANIFEST_DIR / f"{task}.json"
    if not path.exists():
        raise DataError(
            f"No manifest for task {task!r} at {path}. Build it with "
            f"`modal run coding_bench/data/build_remote.py` and commit the result."
        )
    return json.loads(path.read_text(encoding="utf8"))


def manifest_checksum(task: str) -> str:
    """The checksum CI compares the volume against."""
    return hashlib.sha256(_canonical_bytes(load_manifest(task))).hexdigest()


def load_label_space(task: str, allow_restricted_descriptions: bool = False) -> dict[str, str | None]:
    """The label space for a task.

    CPT descriptors are AMA copyrighted, so the committed CPT label space has
    codes and no descriptions. When running inside Modal, the descriptions come
    off the restricted volume instead, which is why the flag exists.
    """
    path = LABEL_DIR / f"{task}.json"
    if not path.exists():
        raise DataError(f"No label space for {task!r} at {path}")
    spec = json.loads(path.read_text(encoding="utf8"))
    labels: dict[str, str | None] = dict(spec["codes"])

    if allow_restricted_descriptions and not spec.get("descriptions_committed", True):
        volume_descriptions = GOLD_ROOT / "labels" / f"{task}_descriptions.json"
        if volume_descriptions.exists():
            extra = json.loads(volume_descriptions.read_text(encoding="utf8"))
            labels = {code: extra.get(code, labels.get(code)) for code in labels}

    return labels


def load_gold(task: str, root: Path | None = None, verify: bool = True) -> Dataset:
    """Load a Tier 2 gold set. Only ever call this where the volume is mounted."""
    import pyarrow.parquet as pq

    root = root or GOLD_ROOT
    path = root / "gold" / f"{task}.parquet"
    if not path.exists():
        raise DataError(
            f"No gold set at {path}. Tier 2 data lives on the coding-benchmark-gold "
            f"volume; run this inside Modal or point CODING_BENCH_GOLD_ROOT at a local build."
        )

    manifest = load_manifest(task)
    expected = {note["note_id"]: note for note in manifest["notes"]}

    table = pq.read_table(path)
    notes: list[Note] = []
    mismatches: list[int] = []

    for row in table.to_pylist():
        note_id = int(row["note_id"])
        if verify:
            reference = expected.get(note_id)
            if reference is None:
                mismatches.append(note_id)
                continue
            if _sha256(row["text"]) != reference["text_sha256"]:
                mismatches.append(note_id)
                continue
            if sorted(row["gold_codes"]) != reference["gold_codes"]:
                mismatches.append(note_id)
                continue

        gold_spans: dict[str, list[tuple[int, int]]] = {}
        for span in row["evidence"] or []:
            gold_spans.setdefault(span["code"], []).append((span["begin"], span["end"]))

        notes.append(
            Note(
                note_id=str(note_id),
                text=row["text"],
                gold_codes=tuple(sorted(row["gold_codes"])),
                gold_spans=gold_spans,
                category=row.get("category") or "",
                description=row.get("description") or "",
            )
        )

    if verify:
        missing = sorted(set(expected) - {int(note.note_id) for note in notes} - set(mismatches))
        if mismatches or missing:
            raise DataError(
                f"Gold set at {path} does not match the committed manifest for {task!r}: "
                f"{len(mismatches)} notes differ, {len(missing)} are absent. "
                f"Rebuild the volume before scoring anything against it."
            )

    notes.sort(key=lambda note: int(note.note_id))
    return Dataset(
        task=task,
        code_system=manifest["code_system"],
        tier=2,
        notes=notes,
        label_space=load_label_space(task, allow_restricted_descriptions=True),
        manifest_sha256=manifest_checksum(task),
    )


def load_smoke(code_system: str = "icd10") -> Dataset:
    """Load the Tier 0 synthetic set. Safe in public CI, safe in a classroom."""
    path = SMOKE_DIR / "notes.json"
    if not path.exists():
        raise DataError(f"No smoke set at {path}")

    spec = json.loads(path.read_text(encoding="utf8"))
    key = "cpt" if code_system == "cpt" else "icd10"

    notes = []
    for record in spec["notes"]:
        gold = sorted(record.get(key, []))
        spans = {
            code: [(span[0], span[1]) for span in spans]
            for code, spans in (record.get(f"{key}_evidence") or {}).items()
        }
        notes.append(
            Note(
                note_id=record["note_id"],
                text=record["text"],
                gold_codes=tuple(gold),
                gold_spans=spans,
                category=record.get("category", ""),
                description=record.get("description", ""),
            )
        )

    notes.sort(key=lambda note: note.note_id)
    label_space = spec["label_spaces"][key]
    return Dataset(
        task=f"smoke_{key}",
        code_system="CPT" if key == "cpt" else "ICD-10-CM",
        tier=0,
        notes=notes,
        label_space=label_space,
        manifest_sha256=hashlib.sha256(_canonical_bytes(spec)).hexdigest(),
    )


def load(task: str, **kwargs) -> Dataset:
    """Load any task by name."""
    if task.startswith("smoke"):
        return load_smoke(task.replace("smoke_", "") or "icd10")
    if task not in ("cpt", "icd10"):
        raise DataError(f"Unknown task {task!r}, expected one of {TASKS}")
    return load_gold(task, **kwargs)


def verify_volume(root: Path | None = None) -> dict[str, str]:
    """Check the volume's stamp against the committed manifests.

    This is the check that runs before any restricted evaluation, so that a
    stale volume fails loudly rather than producing numbers nobody can trace.
    """
    root = root or GOLD_ROOT
    stamp_path = root / "CHECKSUMS.json"
    if not stamp_path.exists():
        raise DataError(f"Volume at {root} carries no CHECKSUMS.json, so it cannot be trusted")

    stamp = json.loads(stamp_path.read_text(encoding="utf8"))
    problems = []
    for task, volume_sum in sorted(stamp.items()):
        repo_sum = manifest_checksum(task)
        if volume_sum != repo_sum:
            problems.append(f"{task}: volume {volume_sum[:12]}, repo {repo_sum[:12]}")

    if problems:
        raise DataError(
            "Volume and repo manifests disagree, refusing to score:\n  " + "\n  ".join(problems)
        )
    return stamp
