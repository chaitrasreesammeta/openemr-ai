"""Deterministic builder for the Tier 2 gold sets.

Joins MDACE Profee annotations (public) to MIMIC-III NOTEEVENTS (restricted) and
emits, per task:

  gold/<task>.parquet      Tier 2. Note text. Never committed, never logged.
  manifests/<task>.json    Tier 1. note_id, gold codes, text_sha256, char_len.
  labels/<task>.json       Tier 1. The label space.

The manifest is the integrity contract. It carries a one way hash of each note
instead of the note, so it is public and reviewable, and any rebuild that does
not reproduce it byte for byte is rejected before it can be scored against.

Both inputs are required arguments. There is deliberately no default path to
anyone's machine.

Usage:
    python -m coding_bench.data.build \
        --mdace ~/benchmark-data/MDACE \
        --noteevents /path/to/NOTEEVENTS.csv.gz \
        --out-dir coding_bench/data

    # rebuild and check against what is already committed, emitting nothing
    python -m coding_bench.data.build --mdace ... --noteevents ... --verify-only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# The MDACE release these manifests were built from. Pinned so that a rebuild
# two years from now reproduces the same gold set.
MDACE_COMMIT = "bf438d0d3a0dd905439011bcf423de66ee85d3ce"
MDACE_REPO = "https://github.com/solventum-oss/MDACE.git"

# MDACE lays out annotations as data/<subset>/<code version>/<release>. The
# Profee ICD-10 release carries both the CPT and the ICD-10-CM annotations, so
# one directory feeds both tasks.
MDACE_SUBSET = "data/Profee/ICD-10/1.0"

# Bumping this invalidates every committed manifest on purpose. Change it only
# when the join or the normalisation changes, never for cosmetic edits.
BUILDER_VERSION = 1

TASKS = {"cpt": "CPT", "icd10": "ICD-10-CM"}

# NOTEEVENTS.csv column order in MIMIC-III v1.4.
_ROW_ID = 0
_HADM_ID = 2
_CATEGORY = 6
_DESCRIPTION = 7
_TEXT = 10


class BuildError(RuntimeError):
    """Raised when the build cannot produce a gold set it can vouch for."""


@dataclass
class NoteAnnotations:
    """Every annotation MDACE records for one note under one code system."""

    note_id: int
    hadm_id: int
    category: str
    description: str
    codes: set[str] = field(default_factory=set)
    # (code, begin, end) evidence spans. Offsets are public in MDACE; the text
    # they cover is not, so covered_text is never carried here.
    spans: set[tuple[str, int, int]] = field(default_factory=set)
    descriptions: dict[str, str] = field(default_factory=dict)


def text_sha256(text: str) -> str:
    """Hash a note exactly as MDACE offsets index it.

    No normalisation of any kind. MDACE begin/end offsets index into the raw
    NOTEEVENTS text, so stripping or rewrapping would silently invalidate every
    evidence span in the dataset.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_bytes(obj) -> bytes:
    """Byte representation a checksum can be taken over reproducibly."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def checksum(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def collect_annotations(mdace_dir: Path) -> dict[str, dict[int, NoteAnnotations]]:
    """Read the MDACE chart JSONs into per task, per note annotations."""
    annotation_dir = mdace_dir / MDACE_SUBSET
    chart_files = sorted(annotation_dir.glob("*.json"))
    if not chart_files:
        raise BuildError(
            f"No MDACE chart JSONs under {annotation_dir}. Clone {MDACE_REPO} "
            f"and check out {MDACE_COMMIT[:12]}."
        )

    per_task: dict[str, dict[int, NoteAnnotations]] = {task: {} for task in TASKS}

    for chart_file in chart_files:
        chart = json.loads(chart_file.read_text(encoding="utf8"))
        hadm_id = int(chart["hadm_id"])
        for note in chart["notes"]:
            note_id = int(note["note_id"])
            for annotation in note.get("annotations", []):
                task = _task_for(annotation["code_system"])
                if task is None:
                    continue
                bucket = per_task[task]
                entry = bucket.get(note_id)
                if entry is None:
                    entry = NoteAnnotations(
                        note_id=note_id,
                        hadm_id=hadm_id,
                        category=note.get("category", ""),
                        description=note.get("description", ""),
                    )
                    bucket[note_id] = entry
                elif entry.hadm_id != hadm_id:
                    raise BuildError(
                        f"Note {note_id} appears under admissions {entry.hadm_id} "
                        f"and {hadm_id}; the join key is no longer unique."
                    )
                code = annotation["code"]
                entry.codes.add(code)
                entry.spans.add((code, int(annotation["begin"]), int(annotation["end"])))
                desc = (annotation.get("description") or "").strip()
                if desc:
                    entry.descriptions.setdefault(code, desc)

    return per_task


def _task_for(code_system: str) -> str | None:
    for task, system in TASKS.items():
        if system == code_system:
            return task
    return None


def load_note_texts(noteevents: Path, needed: set[int]) -> dict[int, str]:
    """Stream NOTEEVENTS and return the text of just the notes we need.

    Streaming rather than loading the frame keeps peak memory near the size of
    the notes we actually keep, which is a few hundred, not two million.
    """
    if not noteevents.exists():
        raise BuildError(f"NOTEEVENTS not found at {noteevents}")

    csv.field_size_limit(sys.maxsize)
    opener = gzip.open if noteevents.suffix == ".gz" else open
    texts: dict[int, str] = {}

    with opener(noteevents, "rt", encoding="utf8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[_ROW_ID].strip('"').upper() != "ROW_ID" or header[_TEXT].strip('"').upper() != "TEXT":
            raise BuildError(
                f"Unexpected NOTEEVENTS header {header[:3]}...; expected MIMIC-III v1.4 column order"
            )
        for row in reader:
            row_id = int(row[_ROW_ID])
            if row_id in needed:
                texts[row_id] = row[_TEXT]
                if len(texts) == len(needed):
                    break

    return texts


def build_rows(
    annotations: dict[int, NoteAnnotations], texts: dict[int, str]
) -> list[dict]:
    """Join annotations to text, sorted so the output does not depend on walk order."""
    rows = []
    missing = []
    for note_id in sorted(annotations):
        entry = annotations[note_id]
        text = texts.get(note_id)
        if text is None:
            missing.append(note_id)
            continue
        rows.append(
            {
                "note_id": note_id,
                "hadm_id": entry.hadm_id,
                "category": entry.category,
                "description": entry.description,
                "text": text,
                "gold_codes": sorted(entry.codes),
                "evidence": [
                    {"code": code, "begin": begin, "end": end}
                    for code, begin, end in sorted(entry.spans)
                ],
                "text_sha256": text_sha256(text),
                "char_len": len(text),
            }
        )

    if missing:
        raise BuildError(
            f"{len(missing)} annotated notes had no NOTEEVENTS match "
            f"(first few: {missing[:5]}). The NOTEEVENTS copy is incomplete or "
            f"is not MIMIC-III v1.4."
        )
    return rows


def make_manifest(task: str, rows: list[dict]) -> dict:
    """The committable, text free description of one gold set."""
    codes = sorted({code for row in rows for code in row["gold_codes"]})
    return {
        "task": task,
        "code_system": TASKS[task],
        "builder_version": BUILDER_VERSION,
        "mdace_commit": MDACE_COMMIT,
        "mdace_subset": MDACE_SUBSET,
        "text_source": "MIMIC-III v1.4 NOTEEVENTS.csv, joined on ROW_ID, verbatim",
        "n_notes": len(rows),
        "n_codes": len(codes),
        "n_spans": sum(len(row["evidence"]) for row in rows),
        "notes": [
            {
                "note_id": row["note_id"],
                "gold_codes": row["gold_codes"],
                "text_sha256": row["text_sha256"],
                "char_len": row["char_len"],
            }
            for row in rows
        ],
    }


def make_label_space(task: str, annotations: dict[int, NoteAnnotations]) -> dict:
    """The committable label space.

    ICD-10-CM descriptors are public domain (CDC/CMS) and ship with the code
    list. CPT descriptors are AMA copyrighted, so the committed CPT label space
    is codes only; descriptions for CPT stay on the restricted volume and are
    loaded at run time. See the copyright note in the design doc.
    """
    descriptions: dict[str, str] = {}
    for entry in annotations.values():
        for code, desc in entry.descriptions.items():
            descriptions.setdefault(code, desc)

    codes = sorted({code for entry in annotations.values() for code in entry.codes})
    committed = TASKS[task] != "CPT"
    return {
        "code_system": TASKS[task],
        "n_codes": len(codes),
        "descriptions_committed": committed,
        "codes": {code: (descriptions.get(code) if committed else None) for code in codes},
    }


def full_descriptions(annotations: dict[int, NoteAnnotations]) -> dict[str, str]:
    """Every descriptor MDACE supplies, for the restricted volume only."""
    out: dict[str, str] = {}
    for entry in annotations.values():
        for code, desc in entry.descriptions.items():
            out.setdefault(code, desc)
    return dict(sorted(out.items()))


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("note_id", pa.int64()),
            ("hadm_id", pa.int64()),
            ("category", pa.string()),
            ("description", pa.string()),
            ("text", pa.string()),
            ("gold_codes", pa.list_(pa.string())),
            (
                "evidence",
                pa.list_(
                    pa.struct(
                        [("code", pa.string()), ("begin", pa.int32()), ("end", pa.int32())]
                    )
                ),
            ),
            ("text_sha256", pa.string()),
            ("char_len", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def verify_manifest(built: dict, committed_path: Path) -> None:
    """Refuse to proceed unless the rebuild reproduces what is committed."""
    if not committed_path.exists():
        raise BuildError(f"No committed manifest at {committed_path} to verify against")

    committed = json.loads(committed_path.read_text(encoding="utf8"))
    problems = []

    for key in ("code_system", "builder_version", "mdace_commit", "n_notes", "n_codes", "n_spans"):
        if built[key] != committed.get(key):
            problems.append(f"{key}: rebuilt {built[key]!r}, committed {committed.get(key)!r}")

    built_notes = {note["note_id"]: note for note in built["notes"]}
    committed_notes = {note["note_id"]: note for note in committed.get("notes", [])}

    only_built = sorted(set(built_notes) - set(committed_notes))
    only_committed = sorted(set(committed_notes) - set(built_notes))
    if only_built:
        problems.append(f"{len(only_built)} notes not in the manifest (first: {only_built[:3]})")
    if only_committed:
        problems.append(f"{len(only_committed)} manifest notes missing (first: {only_committed[:3]})")

    hash_mismatch = [
        note_id
        for note_id in sorted(set(built_notes) & set(committed_notes))
        if built_notes[note_id]["text_sha256"] != committed_notes[note_id]["text_sha256"]
    ]
    if hash_mismatch:
        problems.append(
            f"{len(hash_mismatch)} notes hash differently, so the note text is not "
            f"the text the benchmark was defined on (first: {hash_mismatch[:3]})"
        )

    code_mismatch = [
        note_id
        for note_id in sorted(set(built_notes) & set(committed_notes))
        if built_notes[note_id]["gold_codes"] != committed_notes[note_id]["gold_codes"]
    ]
    if code_mismatch:
        problems.append(f"{len(code_mismatch)} notes have different gold codes (first: {code_mismatch[:3]})")

    if problems:
        raise BuildError(
            "Rebuild does not match the committed manifest:\n  " + "\n  ".join(problems)
        )


def build(
    mdace_dir: Path,
    noteevents: Path,
    out_dir: Path,
    tasks: list[str] | None = None,
    verify_only: bool = False,
    write_manifests: bool = False,
) -> dict[str, dict]:
    """Build every task and return a text free report per task."""
    tasks = tasks or list(TASKS)
    per_task = collect_annotations(mdace_dir)

    needed = {note_id for task in tasks for note_id in per_task[task]}
    print(f"Loading {len(needed)} note texts from {noteevents.name}", file=sys.stderr)
    texts = load_note_texts(noteevents, needed)
    print(f"Matched {len(texts)}/{len(needed)} notes", file=sys.stderr)

    report: dict[str, dict] = {}
    for task in tasks:
        annotations = per_task[task]
        rows = build_rows(annotations, texts)
        manifest = make_manifest(task, rows)
        manifest_path = out_dir / "manifests" / f"{task}.json"

        if verify_only:
            verify_manifest(manifest, manifest_path)
            print(f"{task}: verified against {manifest_path}", file=sys.stderr)
        else:
            if manifest_path.exists():
                verify_manifest(manifest, manifest_path)
            write_parquet(rows, out_dir / "gold" / f"{task}.parquet")
            if write_manifests or not manifest_path.exists():
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_bytes(canonical_bytes(manifest))
                label_path = out_dir / "labels" / f"{task}.json"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_bytes(canonical_bytes(make_label_space(task, annotations)))

        report[task] = {
            "n_notes": manifest["n_notes"],
            "n_codes": manifest["n_codes"],
            "n_spans": manifest["n_spans"],
            "manifest_sha256": checksum(manifest),
            "mean_char_len": round(sum(r["char_len"] for r in rows) / max(len(rows), 1)),
        }

    if not verify_only:
        _write_checksums(out_dir, report)

    return report


def _write_checksums(out_dir: Path, report: dict[str, dict]) -> None:
    path = out_dir / "manifests" / "checksums.json"
    existing = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    existing.update({task: info["manifest_sha256"] for task, info in report.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(dict(sorted(existing.items()))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mdace", type=Path, required=True, help="Path to a checkout of the MDACE repo")
    parser.add_argument("--noteevents", type=Path, required=True, help="Path to MIMIC-III NOTEEVENTS.csv or .csv.gz")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Where gold/, manifests/ and labels/ live")
    parser.add_argument("--tasks", nargs="*", choices=list(TASKS), default=None)
    parser.add_argument("--verify-only", action="store_true", help="Rebuild in memory and check the committed manifests, write nothing")
    parser.add_argument("--write-manifests", action="store_true", help="Overwrite the committed manifests and label spaces")
    args = parser.parse_args()

    try:
        report = build(
            mdace_dir=args.mdace,
            noteevents=args.noteevents,
            out_dir=args.out_dir,
            tasks=args.tasks,
            verify_only=args.verify_only,
            write_manifests=args.write_manifests,
        )
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
