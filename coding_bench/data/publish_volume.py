"""Upload a locally built gold set to the private Modal volume.

Only needed for the local build path. The Modal build in build_remote.py writes
the volume directly and this script never runs. It exists because the design's
canonical path is that any credentialed member can rebuild from scratch, and a
rebuild is worthless if it cannot replace what CI reads.

The upload is stamped with the manifest checksums, which is the same stamp the
loader checks before scoring, so a half finished upload fails the next run
rather than silently changing the numbers.

    python coding_bench/data/publish_volume.py --gold-dir coding_bench/data/gold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VOLUME_NAME = "coding-benchmark-gold"


def main() -> int:
    import modal

    from coding_bench.data import build

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", type=Path, required=True, help="Directory holding the built parquets")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent, help="Where manifests/ lives")
    parser.add_argument("--volume", default=VOLUME_NAME)
    args = parser.parse_args()

    parquets = sorted(args.gold_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquets under {args.gold_dir}", file=sys.stderr)
        return 1

    checksums = {}
    for parquet in parquets:
        task = parquet.stem
        manifest_path = args.data_dir / "manifests" / f"{task}.json"
        if not manifest_path.exists():
            print(f"No committed manifest for {task}, refusing to publish", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        checksums[task] = build.checksum(manifest)

    stamp_path = args.gold_dir / "CHECKSUMS.json"
    stamp_path.write_bytes(build.canonical_bytes(checksums))

    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for parquet in parquets:
            print(f"  gold/{parquet.name}  {parquet.stat().st_size / 1e6:.1f} MB")
            batch.put_file(parquet, f"/gold/{parquet.name}")
        for descriptions in sorted(args.gold_dir.glob("*_descriptions.json")):
            batch.put_file(descriptions, f"/labels/{descriptions.name}")
        batch.put_file(stamp_path, "/CHECKSUMS.json")

    print(f"Published to {args.volume} with stamp {json.dumps(checksums, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
