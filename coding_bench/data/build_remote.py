"""Build the Tier 2 gold sets inside Modal, so restricted text never lands here.

The whole join happens on Modal: NOTEEVENTS is pulled from PhysioNet straight
into a private volume, MDACE is cloned at a pinned commit, and the parquets are
written to the gold volume. What comes back over the wire is Tier 1 only, the
manifests and label spaces, which carry hashes and code lists but no note text.

One time setup, run in your own terminal so that no credential enters a log.
The S3 route is the one to use, and it needs AWS keys and nothing else:

    modal secret create physionet \
        AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>

The secret is called `physionet` because it is the one this app declares, not
because it has to hold a PhysioNet login. A separate `aws` secret would be
tidier and would break every run that had not created it, since Modal resolves
every secret in an app at startup.

`PHYSIONET_USER` and `PHYSIONET_PASS` are needed only by the `parallel` and
`wget` routes, which download over HTTP from PhysioNet directly. The S3 route
returns before the code ever asks for them. Add them to the same secret if you
want those fallbacks to work:

    modal secret create physionet \
        AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> \
        PHYSIONET_USER=<user> PHYSIONET_PASS=<pass>

Then:

    modal run coding_bench/data/build_remote.py                 # build and write Tier 1 locally
    modal run coding_bench/data/build_remote.py --source s3     # force a specific download route
    modal run coding_bench/data/build_remote.py --verify-only   # check the volume against the repo
    modal run coding_bench/data/build_remote.py::inspect        # what is on the volumes

Download routes, in the order `auto` picks them: `s3` (fastest, AWS keys only),
`parallel` (aria2c, 16 connections, PhysioNet login), `wget` (one connection,
PhysioNet login, slow enough to be a last resort). The download is cached on the
raw volume, so this cost is paid once.

Whichever route is used, the account behind it must be credentialed for "MIMIC-III
Clinical Database" v1.4 specifically: for S3 that means an AWS account PhysioNet
has linked to a credentialed profile, and for the other two the PhysioNet login
itself. MDACE offsets index NOTEEVENTS.csv ROW_ID, so MIMIC-IV-Note cannot
substitute for it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

NOTEEVENTS_URL = "https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
SHASUMS_URL = "https://physionet.org/files/mimiciii/1.4/SHA256SUMS.txt"

# PhysioNet mirrors MIMIC-III into S3, which is far faster than the web server,
# because that throttles a single connection to a crawl. It needs an AWS account
# that PhysioNet has linked to your credentialed profile, with
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY on the physionet secret.
#
# It is served through an S3 access point rather than a plain bucket, and your
# AWS principal is granted access to that access point when you enable cloud
# access on the project page. Access point ARNs are addressed directly as the
# s3:// target, and access is granted rather than requester pays, so there is no
# --request-payer flag and the transfer is not billed to the caller.
S3_URI = (
    "s3://arn:aws:s3:us-east-1:724665945834:accesspoint/mimiciii-v1-4-01"
    "/mimiciii/1.4/NOTEEVENTS.csv.gz"
)
S3_REGION = "us-east-1"

DOWNLOAD_SOURCES = ("s3", "parallel", "wget")

GOLD_VOLUME = "coding-benchmark-gold"
RAW_VOLUME = "coding-benchmark-raw"

RAW_MOUNT = Path("/raw")
GOLD_MOUNT = Path("/gold")
NOTEEVENTS_PATH = RAW_MOUNT / "mimiciii-1.4" / "NOTEEVENTS.csv.gz"
MDACE_PATH = RAW_MOUNT / "MDACE"

LOCAL_DATA_DIR = Path(__file__).parent

app = modal.App("coding-bench-data")

gold_volume = modal.Volume.from_name(GOLD_VOLUME, create_if_missing=True)
raw_volume = modal.Volume.from_name(RAW_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "aria2")
    .pip_install("pyarrow>=15.0.0", "awscli>=1.32.0")
    .add_local_file(LOCAL_DATA_DIR / "build.py", "/root/build.py")
)


def _load_builder():
    """Import the shared builder that was added to the image."""
    sys.path.insert(0, "/root")
    import build  # noqa: PLC0415

    return build


def _credentials() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        raise RuntimeError("The physionet secret must define PHYSIONET_USER and PHYSIONET_PASS")
    return user, password


def _run_streaming(command: list[str], what: str) -> None:
    """Run a downloader with its output going straight to the container log.

    Deliberately not capture_output. Capturing puts progress into a pipe that
    nobody reads until the process exits, which makes a slow download
    indistinguishable from a hung one for however long it takes. Downloaders
    print progress to stderr and never echo the password, so letting it stream
    is both safe and the only way to see what is happening.
    """
    import subprocess

    result = subprocess.run(command)
    if result.returncode != 0:
        # Deliberately leave whatever was written in place. Deleting it would
        # throw away hundreds of megabytes that the next attempt could resume
        # from. Nothing downstream trusts the file on size alone: the cache
        # check compares it against the server and the gzip check rejects an
        # error page, so a bad partial cannot be mistaken for the dataset.
        raise RuntimeError(
            f"{what} failed with exit code {result.returncode}. A 401 or 403 means the "
            f"account is not credentialed for MIMIC-III v1.4; AccessDenied on S3 means "
            f"the AWS account is not linked to a credentialed PhysioNet profile. The "
            f"downloader's own output is above this message in the Modal log."
        )


def remote_size() -> int | None:
    """Ask PhysioNet how big NOTEEVENTS is, without downloading it."""
    import base64
    import urllib.error
    import urllib.request

    try:
        user, password = _credentials()
    except RuntimeError:
        return None

    request = urllib.request.Request(NOTEEVENTS_URL, method="HEAD")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    request.add_header("Authorization", f"Basic {token}")
    # PhysioNet answers 403 to unrecognised clients, urllib included, the same
    # way it refuses aria2. Without this the size check silently degrades to
    # "unverified" and a truncated cache would be trusted.
    request.add_header("User-Agent", "Wget/1.21.3")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError) as exc:
        print(f"Could not read the remote size ({exc})", flush=True)
        return None


def _cached_and_complete() -> tuple[int, bool] | None:
    """Return the cached size only if the cache is actually the whole file.

    Existence is not completeness. A download interrupted partway leaves a file
    that looks cached and would then be joined against as though it were the
    dataset, quietly producing a gold set missing whatever came after the cut.
    Comparing against the server's Content-Length costs one HEAD request.
    """
    if not NOTEEVENTS_PATH.exists():
        return None

    # aria2 preallocates the target at its full length before fetching a byte,
    # so size alone would call a 1 percent download complete. It removes this
    # control file only on success, which makes it the authoritative signal.
    control = NOTEEVENTS_PATH.parent / f"{NOTEEVENTS_PATH.name}.aria2"
    if control.exists():
        print("An aria2 control file is present, so the download is unfinished", flush=True)
        return None

    local = NOTEEVENTS_PATH.stat().st_size
    expected = remote_size()
    if expected is None:
        # Report the uncertainty rather than paper over it. Claiming a check
        # passed when it never ran is how a truncated file gets trusted.
        print(f"Cached file is {local / 1e6:.0f} MB, size NOT verified", flush=True)
        return local, False
    if local != expected:
        print(
            f"Cached file is {local / 1e6:.0f} MB but the server reports "
            f"{expected / 1e6:.0f} MB, so it is incomplete. Re-fetching.",
            flush=True,
        )
        return None
    return local, True


def _check_is_gzip(path: Path) -> None:
    """A login page or an error body is not a dataset."""
    with open(path, "rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            path.unlink(missing_ok=True)
            raise RuntimeError(
                "The downloaded file is not gzip, which usually means the server "
                "returned a login or error page instead of the data."
            )


def _download_noteevents(source: str) -> str:
    """Fetch NOTEEVENTS by the fastest route available, returning which was used.

    PhysioNet's web server throttles a single HTTP connection to a crawl, so
    plain wget can take hours for this file. In preference order:

      s3        the PhysioNet mirror in a requester pays bucket, fastest, but
                needs AWS keys on an account PhysioNet has linked to yours
      parallel  aria2c with 16 connections against the same PhysioNet URL,
                no extra credentials, usually many times faster than wget
      wget      one connection, the slow path, kept only as a last resort
    """
    have_aws = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))

    if source == "auto":
        # Try fastest first and fall back, because a route can fail for reasons
        # that have nothing to do with the credentials being wrong. An explicit
        # --source never falls back, so a deliberate choice fails visibly.
        chain = (["s3"] if have_aws else []) + ["parallel", "wget"]
        last_error: Exception | None = None
        for candidate in chain:
            try:
                return _download_noteevents(candidate)
            except RuntimeError as exc:
                last_error = exc
                print(f"Route {candidate!r} failed, trying the next one:\n{exc}", flush=True)
        raise RuntimeError(f"Every download route failed. Last error: {last_error}")

    if source == "s3":
        if not have_aws:
            raise RuntimeError(
                "Source 's3' needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. Add them "
                "to the existing physionet secret (Modal resolves every secret in the app "
                "at startup, so a separate optional secret would break runs that lack it), "
                "using an AWS account that PhysioNet has linked to your credentialed profile."
            )
        print(f"Downloading {S3_URI} (requester pays)", flush=True)
        _run_streaming(
            [
                "aws", "s3", "cp", S3_URI, str(NOTEEVENTS_PATH),
                "--region", S3_REGION,
            ],
            "S3 download",
        )
        _check_is_gzip(NOTEEVENTS_PATH)
        # A control file left behind by an abandoned aria2 attempt would make
        # this finished download look unfinished on the next run.
        (NOTEEVENTS_PATH.parent / f"{NOTEEVENTS_PATH.name}.aria2").unlink(missing_ok=True)
        return "s3"

    user, password = _credentials()

    if source == "parallel":
        print(f"Downloading {NOTEEVENTS_URL} with 16 connections", flush=True)
        _run_streaming(
            [
                "aria2c", NOTEEVENTS_URL,
                "--http-user", user, "--http-passwd", password,
                # PhysioNet answers 403 to aria2's own user agent even with
                # valid credentials. They document wget and the AWS CLI as the
                # supported clients, so present as wget.
                "--user-agent", "Wget/1.21.3",
                "-x", "16", "-s", "16", "-k", "10M",
                "--max-tries=5", "--retry-wait=5",
                "--summary-interval=15", "--console-log-level=warn",
                # Resume rather than restart. An interrupted download of this
                # file is expensive to repeat, and aria2c keeps a .aria2 control
                # file next to the target so a rerun picks up where it stopped.
                "--continue=true", "--auto-file-renaming=false",
                # No preallocation. It buys nothing on a network volume and it
                # makes a partial file indistinguishable from a finished one by
                # size, which is a trap for every later integrity check.
                "--file-allocation=none",
                "-d", str(NOTEEVENTS_PATH.parent), "-o", NOTEEVENTS_PATH.name,
            ],
            "Parallel download",
        )
        _check_is_gzip(NOTEEVENTS_PATH)
        return "parallel"

    print(f"Downloading {NOTEEVENTS_URL} on one connection", flush=True)
    # -c with -P rather than -O, because wget cannot resume into -O and the
    # URL basename is already the filename we want. A dot line per 32 MiB is
    # enough to see that it is moving without flooding the log.
    _run_streaming(
        [
            "wget", "--continue", "--progress=dot:giga",
            "--user", user, "--password", password,
            "-P", str(NOTEEVENTS_PATH.parent), NOTEEVENTS_URL,
        ],
        "PhysioNet download",
    )
    _check_is_gzip(NOTEEVENTS_PATH)
    return "wget"


@app.function(
    image=image,
    volumes={RAW_MOUNT: raw_volume},
    secrets=[modal.Secret.from_name("physionet")],
    # PhysioNet serves this at roughly 300 KiB/s regardless of how many
    # connections you open, and the file is 1.0 GiB, so the HTTP routes need
    # well over an hour. The S3 mirror is minutes, which is the real fix.
    timeout=10800,
)
def fetch_sources(force: bool = False, source: str = "auto") -> dict:
    """Put NOTEEVENTS and a pinned MDACE checkout on the raw volume."""
    import subprocess

    build = _load_builder()
    report = {}

    NOTEEVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached = _cached_and_complete()
    if cached and not force:
        cached_size, verified = cached
        report["noteevents"] = (
            f"cached, {cached_size / 1e6:.0f} MB, "
            + ("size matches the server" if verified else "size NOT verified against the server")
        )
    else:
        started = time.perf_counter()
        used = _download_noteevents(source)
        size_mb = NOTEEVENTS_PATH.stat().st_size / 1e6
        elapsed = time.perf_counter() - started
        report["noteevents"] = (
            f"downloaded via {used}, {size_mb:.0f} MB in {elapsed / 60:.1f} min "
            f"({size_mb / max(elapsed, 1):.1f} MB/s)"
        )

    if MDACE_PATH.exists() and not force:
        report["mdace"] = "cached"
    else:
        subprocess.run(["rm", "-rf", str(MDACE_PATH)], check=True)
        subprocess.run(
            ["git", "clone", "--quiet", build.MDACE_REPO, str(MDACE_PATH)], check=True
        )
        report["mdace"] = "cloned"

    subprocess.run(
        ["git", "-C", str(MDACE_PATH), "checkout", "--quiet", build.MDACE_COMMIT], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(MDACE_PATH), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != build.MDACE_COMMIT:
        raise RuntimeError(f"MDACE checkout is at {head}, expected {build.MDACE_COMMIT}")
    report["mdace_commit"] = head

    raw_volume.commit()
    return report


@app.function(
    image=image,
    volumes={RAW_MOUNT: raw_volume, GOLD_MOUNT: gold_volume},
    timeout=3600,
    memory=8192,
)
def build_gold(verify_only: bool = False, committed_manifests: dict | None = None) -> dict:
    """Run the shared builder against the volumes.

    Returns Tier 1 artifacts and counts. Nothing in the return value contains
    note text, which is what makes it safe to print in a CI log.
    """
    build = _load_builder()

    work_dir = Path("/tmp/work")
    (work_dir / "manifests").mkdir(parents=True, exist_ok=True)
    if committed_manifests:
        for task, manifest in committed_manifests.items():
            (work_dir / "manifests" / f"{task}.json").write_bytes(build.canonical_bytes(manifest))

    per_task = build.collect_annotations(MDACE_PATH)
    needed = {note_id for annotations in per_task.values() for note_id in annotations}
    print(f"Loading {len(needed)} note texts", file=sys.stderr)
    texts = build.load_note_texts(NOTEEVENTS_PATH, needed)
    print(f"Matched {len(texts)}/{len(needed)}", file=sys.stderr)

    out: dict = {"tasks": {}, "manifests": {}, "labels": {}}
    for task in build.TASKS:
        annotations = per_task[task]
        rows = build.build_rows(annotations, texts)
        manifest = build.make_manifest(task, rows)

        committed_path = work_dir / "manifests" / f"{task}.json"
        if committed_path.exists():
            build.verify_manifest(manifest, committed_path)
            print(f"{task}: matches the committed manifest", file=sys.stderr)
        elif verify_only:
            raise RuntimeError(f"--verify-only but no committed manifest was supplied for {task}")

        if not verify_only:
            build.write_parquet(rows, GOLD_MOUNT / "gold" / f"{task}.parquet")
            descriptions = build.full_descriptions(annotations)
            desc_path = GOLD_MOUNT / "labels" / f"{task}_descriptions.json"
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            desc_path.write_bytes(build.canonical_bytes(descriptions))

        out["manifests"][task] = manifest
        out["labels"][task] = build.make_label_space(task, annotations)
        out["tasks"][task] = {
            "n_notes": manifest["n_notes"],
            "n_codes": manifest["n_codes"],
            "n_spans": manifest["n_spans"],
            "manifest_sha256": build.checksum(manifest),
            "mean_char_len": round(sum(r["char_len"] for r in rows) / max(len(rows), 1)),
            "max_char_len": max((r["char_len"] for r in rows), default=0),
        }

    if not verify_only:
        stamp = {task: info["manifest_sha256"] for task, info in out["tasks"].items()}
        (GOLD_MOUNT / "CHECKSUMS.json").write_bytes(build.canonical_bytes(stamp))
        gold_volume.commit()

    return out


@app.function(image=image, volumes={RAW_MOUNT: raw_volume, GOLD_MOUNT: gold_volume})
def inspect() -> dict:
    """List what is on the volumes, sizes only, so it is safe to print."""
    def walk(root: Path) -> dict:
        if not root.exists():
            return {}
        return {
            str(p.relative_to(root)): p.stat().st_size
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.parts
        }

    return {
        "raw": {k: v for k, v in walk(RAW_MOUNT).items() if "MDACE" not in k},
        "mdace_files": sum(1 for _ in MDACE_PATH.rglob("*.json")) if MDACE_PATH.exists() else 0,
        "gold": walk(GOLD_MOUNT),
    }


@app.local_entrypoint()
def main(verify_only: bool = False, force_download: bool = False, source: str = "auto"):
    """Fetch, build, and land the Tier 1 artifacts in the repo."""
    if source not in ("auto",) + DOWNLOAD_SOURCES:
        raise SystemExit(f"--source must be auto or one of {DOWNLOAD_SOURCES}")
    print(json.dumps(fetch_sources.remote(force=force_download, source=source), indent=2))

    manifest_dir = LOCAL_DATA_DIR / "manifests"
    committed = {
        path.stem: json.loads(path.read_text(encoding="utf8"))
        for path in sorted(manifest_dir.glob("*.json"))
        if path.stem != "checksums"
    }
    if committed:
        print(f"Verifying against committed manifests: {', '.join(sorted(committed))}")

    result = build_gold.remote(verify_only=verify_only, committed_manifests=committed or None)

    if not verify_only:
        sys.path.insert(0, str(LOCAL_DATA_DIR))
        import build  # noqa: PLC0415

        for task, manifest in result["manifests"].items():
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / f"{task}.json").write_bytes(build.canonical_bytes(manifest))
            label_dir = LOCAL_DATA_DIR / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / f"{task}.json").write_bytes(build.canonical_bytes(result["labels"][task]))

        checksums = {task: info["manifest_sha256"] for task, info in result["tasks"].items()}
        (manifest_dir / "checksums.json").write_bytes(build.canonical_bytes(checksums))
        print(f"Wrote Tier 1 artifacts to {manifest_dir} and {LOCAL_DATA_DIR / 'labels'}")

    print(json.dumps(result["tasks"], indent=2))
