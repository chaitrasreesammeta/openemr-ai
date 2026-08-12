"""Refuse to let restricted data reach git.

Three checks, run over files added or modified in a commit or a pull request:

  1. Nothing over a size ceiling, because gold sets are large and source is not.
  2. No parquet, pickle, feather, or arrow, whatever the file is named.
  3. No MIMIC de-identification markers, which look like [** ... **]. This is the
     backstop that catches note text pasted into a notebook, a fixture, a test,
     or a markdown file, in any container format.

Check 3 is the important one. Checks 1 and 2 only catch the obvious shape of the
mistake that already happened once.

Usage:
    python coding_bench/scripts/check_restricted.py --staged        # pre-commit
    python coding_bench/scripts/check_restricted.py --diff origin/main   # CI
    python coding_bench/scripts/check_restricted.py path/to/file ...
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000

BLOCKED_SUFFIXES = {".parquet", ".pkl", ".pickle", ".feather", ".arrow", ".npy", ".npz"}

# MIMIC replaces PHI with [**Known lastname 123**], [**2101-4-6**] and similar.
# Two or more hits in one file is note text; one can be a doc example.
DEID_MARKER = re.compile(rb"\[\*\*.{0,120}?\*\*\]", re.DOTALL)
DEID_THRESHOLD = 2

# Files that legitimately describe the marker pattern rather than contain data.
ALLOWLIST = {
    "coding_bench/scripts/check_restricted.py",
    "coding_bench/tests/test_guard_rails.py",
}


def is_allowlisted(name: str) -> bool:
    """Match whether the caller passed a repo relative path or an absolute one."""
    normalised = Path(name).as_posix()
    return any(normalised == entry or normalised.endswith("/" + entry) for entry in ALLOWLIST)


def changed_files(staged: bool, diff_base: str | None) -> list[str]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    elif diff_base:
        cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{diff_base}...HEAD"]
    else:
        return []
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def check(paths: list[str]) -> list[str]:
    problems = []
    for name in paths:
        if is_allowlisted(name):
            continue
        path = Path(name)
        if not path.is_file():
            continue

        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"{name}: {path.suffix} files may never be committed")
            continue

        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{name}: {size / 1e6:.1f} MB exceeds the {MAX_BYTES / 1e6:.0f} MB ceiling")

        try:
            blob = path.read_bytes()
        except OSError as exc:
            problems.append(f"{name}: unreadable ({exc})")
            continue

        hits = DEID_MARKER.findall(blob)
        if len(hits) >= DEID_THRESHOLD:
            problems.append(
                f"{name}: {len(hits)} MIMIC de-identification markers found, "
                f"this looks like restricted note text"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Explicit paths to check")
    parser.add_argument("--staged", action="store_true", help="Check files staged for commit")
    parser.add_argument("--diff", dest="diff_base", help="Check files changed against this ref")
    args = parser.parse_args()

    paths = list(args.paths) + changed_files(args.staged, args.diff_base)
    if not paths:
        print("Nothing to check")
        return 0

    problems = check(paths)
    if problems:
        print("Restricted data check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nTier 2 data belongs on the coding-benchmark-gold Modal volume, "
            "not in git. See coding_bench/README.md.",
            file=sys.stderr,
        )
        return 1

    print(f"Restricted data check passed over {len(paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
