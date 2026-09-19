#!/usr/bin/env python3
"""Reproducibly fetch or verify the external-GEO archive manifest.

The manifest is deliberately separate from downstream parsing.  A download is
atomic (``.part`` then rename), size-checked against the curated manifest, and
recorded with SHA-256.  Deferred/unsuitable rows are skipped unless explicitly
requested; this prevents a convenience command from silently downloading a
large or scientifically excluded dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/external/manifests/sources.tsv")
    parser.add_argument("--out-root", default="data/external/raw")
    parser.add_argument("--only", action="append", default=None, help="Repeatable accession filter.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-deferred", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted_rows(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.only:
        selected = table[table["accession"].astype(str).isin(set(args.only))].copy()
        missing = sorted(set(args.only) - set(selected["accession"].astype(str)))
        if missing:
            raise ValueError(f"Requested accessions absent from manifest: {missing}")
        return selected
    if args.include_deferred:
        return table.copy()
    return table[table["provenance_status"].astype(str).str.startswith("selected")].copy()


def download(url: str, destination: Path, *, retries: int, timeout: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "scGRIP reproducibility downloader"})
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            partial.replace(destination)
            return
        except Exception as exc:  # network failures are retried, but never hidden
            error = exc
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Failed downloading {url} after {retries} attempts: {error}")


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.manifest, sep="\t")
    required = {"accession", "filename", "bytes", "url", "provenance_status"}
    if not required.issubset(table.columns):
        raise ValueError(f"Manifest misses columns: {sorted(required - set(table.columns))}")
    rows = wanted_rows(table, args)
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        accession = str(row.accession)
        expected = int(row.bytes)
        destination = root / accession / str(row.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        exists_and_sized = destination.exists() and destination.stat().st_size == expected
        if not exists_and_sized:
            if args.verify_only:
                status = "missing_or_size_mismatch"
            else:
                print(f"downloading {accession} -> {destination}", flush=True)
                download(str(row.url), destination, retries=args.retries, timeout=args.timeout_seconds)
                status = "downloaded"
        else:
            status = "already_present"
        actual = destination.stat().st_size if destination.exists() else None
        valid = actual == expected
        digest = sha256(destination) if valid else None
        report.append(
            {
                "accession": accession,
                "file": str(destination),
                "expected_bytes": expected,
                "actual_bytes": actual,
                "size_valid": valid,
                "sha256": digest,
                "status": status,
                "url": str(row.url),
            }
        )
        if not valid:
            print(f"INVALID {accession}: expected {expected}, got {actual}", flush=True)
    report_path = root.parent / "manifests" / "download_verification.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys() if report else [])
        writer.writeheader()
        writer.writerows(report)
    if not all(bool(item["size_valid"]) for item in report):
        raise SystemExit("At least one archive is absent or size-invalid; see " + str(report_path))
    print(f"verified {len(report)} archive(s); wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
