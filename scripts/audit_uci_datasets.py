#!/usr/bin/env python3
"""Audit the official UCI index and pin eligible single-CSV snapshots.

Run from the checkout root with a writable cache directory. The cache retains
per-dataset decisions and optional verified fixtures, keeping raw UCI data out
of the repository. Use a fresh cache to evaluate a newer upstream snapshot.
"""
import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path
import re
import time
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "https://archive.ics.uci.edu"
INDEX = HOST + "/api/datasets/list"
FIELDS = ("id", "slug", "title", "rows", "columns", "bytes", "sha256", "missing", "license", "doi", "creators")
LICENSE = re.compile(
    r'This dataset is licensed under\s+(?:an?\s+)?<a\s+href="https?://creativecommons\.org/'
    r'(licenses/by/4\.0|publicdomain/zero/1\.0)(?:/[^"]*)?"',
    re.I,
)


class TooLarge(Exception):
    """A source exceeded its inspection budget."""


def request(url, limit=None, output=None):
    """Read bounded metadata or stream an official CSV to disk, retrying transient failures."""
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "datamunge-dataset-inventory/1.0"}), timeout=15) as response:
                if not response.url.startswith(HOST + "/"):
                    raise ValueError("redirect away from official UCI host")
                if output is not None:
                    with output.open("wb") as stream:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(block)
                    if response.length not in (None, 0):
                        raise OSError("incomplete CSV transfer")
                    return
                if int(response.headers.get("Content-Length", "0")) > limit:
                    raise TooLarge
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise TooLarge
                return data
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
        except (URLError, TimeoutError, OSError, IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(0.5 * (attempt + 1))


def normalized_text(value):
    """Collapse upstream whitespace before recording TSV text."""
    return " ".join(str(value or "").split())


def inspect(dataset, fixtures):
    """Return the selection decision and metadata for one indexed UCI dataset."""
    ident, title = dataset["id"], dataset["name"]
    decision = {"id": ident, "title": title}

    def exclude(reason):
        return {**decision, "status": reason}

    try:
        metadata = json.loads(request(f"{HOST}/api/dataset?id={ident}", 32 * 1024 * 1024)).get("data")
    except HTTPError as error:
        if error.code in (400, 404):
            return exclude("no metadata")
        raise
    if not isinstance(metadata, dict):
        return exclude("no metadata")
    url = f"{HOST}/static/public/{ident}/data.csv"
    if metadata.get("data_url") != url:
        return exclude("no official single CSV")
    page_url = metadata.get("repository_url") or ""
    if not page_url.startswith(f"{HOST}/dataset/{ident}/"):
        return exclude("no official dataset page")
    try:
        page = request(page_url, 8 * 1024 * 1024).decode("utf-8")
    except HTTPError as error:
        if error.code == 404:
            return exclude("no official dataset page")
        raise
    match = LICENSE.search(page)
    if not match:
        return exclude("no explicit supported license")
    creators = metadata.get("creators")
    doi = normalized_text(metadata.get("dataset_doi"))
    if not doi or not isinstance(creators, list) or not any(isinstance(name, str) and name.strip() for name in creators):
        return exclude("incomplete attribution")
    with tempfile.TemporaryDirectory(dir=fixtures) as temp:
        payload = Path(temp) / "data.csv"
        try:
            request(url, output=payload)
        except HTTPError as error:
            if error.code == 404:
                return exclude("CSV unavailable")
            raise
        try:
            with payload.open(encoding="utf-8-sig", newline="") as stream:
                table = csv.reader(stream, strict=True)
                columns = [column.strip() for column in next(table)]
                if any(not column for column in columns) or len(set(columns)) != len(columns):
                    return exclude("unsupported CSV header")
                rows = 0
                for row in table:
                    if not row or (len(row) == 1 and not row[0]):
                        continue
                    if len(row) != len(columns):
                        return exclude("ragged CSV")
                    rows += 1
            if not rows:
                return exclude("empty CSV")
            if metadata.get("num_instances") and rows != metadata["num_instances"]:
                return exclude("row count disagrees with UCI metadata")
        except (UnicodeError, csv.Error, StopIteration):
            return exclude("unsupported CSV encoding or format")
        size = payload.stat().st_size
        with payload.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if fixtures:
            payload.replace(fixtures / f"{ident}.csv")
    license_name = "CC BY 4.0" if match[1].lower().startswith("licenses/") else "CC0 1.0"
    return {
        "status": "included",
        "id": ident,
        "slug": re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"),
        "title": normalized_text(title),
        "rows": rows,
        "columns": len(columns),
        "bytes": size,
        "sha256": sha256,
        "missing": normalized_text(metadata.get("missing_values_symbol")),
        "license": license_name,
        "doi": doi,
        "creators": normalized_text("; ".join(name for name in creators if isinstance(name, str) and name.strip())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-fixtures", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    index = args.cache / "index.json"
    if not index.exists():
        index.write_bytes(request(INDEX, 4 * 1024 * 1024))
    datasets = json.loads(index.read_text(encoding="utf-8"))["data"]
    fixtures = args.cache / "fixtures" if args.keep_fixtures else None
    if fixtures:
        fixtures.mkdir(exist_ok=True)

    def audit(dataset):
        path = args.cache / f"decision-{dataset['id']}.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["status"] != "CSV exceeds 16 MiB":
                return cached
        try:
            result = inspect(dataset, fixtures)
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead, TooLarge, ValueError) as error:
            print(f"UCI {dataset['id']}: {type(error).__name__}: {error}", flush=True)
            return {**dataset, "status": "request unresolved"}
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decisions = list(pool.map(audit, datasets))
    (args.cache / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    included = sorted((row for row in decisions if row["status"] == "included"), key=lambda row: row["id"])
    slugs = set()
    for row in included:
        if not row["slug"] or row["slug"].isdigit() or row["slug"] in slugs:
            row["slug"] = f"{row['slug']}_{row['id']}".strip("_")
        slugs.add(row["slug"])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(included)
    print(f"{len(decisions)} audited: {dict(Counter(row['status'] for row in decisions))}")


if __name__ == "__main__":
    main()
