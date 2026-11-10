#!/usr/bin/env python3
"""Track NN report replacements and retire original sources after verified execution.

The ledger preserves original paths and digests after removal. C++/Python/R
example inventories continue to describe their own runnable programs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/mepml/nn-conversion-status.json"
REPORTS = Path("docs/mepml/examples/nn")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    """Merge live sources with retained retirement records, preserving provenance."""
    records = json.loads(LEDGER.read_text())["examples"] if LEDGER.exists() else {}
    for directory, pattern in (("examples/nn", "[0-9]*/**/*.cpp"),
                               ("examples/cpp/source/nn", "[12]_*/**/*.cpp"),
                               ("examples/nn", "0_foundations/**/numpy_interop.py")):
        base = ROOT / directory
        for source in sorted(base.glob(pattern)):
            relative = source.relative_to(base)
            key = relative.with_suffix("").as_posix()
            old = records.get(key, {})
            if old.get("retired"):
                raise ValueError(f"retired source reappeared: {source}")
            records[key] = {"source": source.relative_to(ROOT).as_posix(),
                            "source_sha256": digest(source),
                            "report": (REPORTS / relative.with_suffix(".mepml")).as_posix(),
                            "retired": False}
    for record in records.values():
        source, report = ROOT / record["source"], ROOT / record["report"]
        if not source.exists() and not record["retired"]:
            raise ValueError(f"source disappeared without retirement evidence: {source}")
        if record["retired"] and not report.is_file():
            raise ValueError(f"retired source has no replacement: {report}")
    return dict(sorted(records.items()))


def retire(records, name, verification):
    """Validate the report runner's successful outcomes before unlinking one source."""
    matches = [key for key in records if key == name or Path(key).name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one source for {name!r}, found {len(matches)}")
    record = records[matches[0]]
    previously_retired = record["retired"]
    report = ROOT / record["report"]
    run = json.loads(verification.read_text())
    files = [f for f in run.get("files", []) if Path(f["file"]).resolve() == report.resolve()]
    if len(files) != 1 or not report.is_file():
        raise ValueError("verification must name the exact replacement report")
    result = files[0]
    # A report may include explanatory language-less code listings. Those need
    # no execution, but every explicitly runnable fence must have an ok outcome.
    languages = re.findall(r"^```\{\s*([\w+]+)\b", report.read_text(), re.M)
    outcomes = [b for b in result.get("blocks", []) if b.get("lang")]
    if not languages or len(languages) != len(outcomes) or sorted(languages) != sorted(b["lang"] for b in outcomes):
        raise ValueError("verification did not execute every runnable block")
    if result.get("failed") or result.get("error") or any(b["status"] != "ok" for b in outcomes):
        raise ValueError("replacement report contains failed or skipped runnable blocks")
    if report.stat().st_mtime > verification.stat().st_mtime + 1:
        raise ValueError("report changed after verification; rerun it before retiring the source")
    source = ROOT / record["source"]
    if not previously_retired and digest(source) != record["source_sha256"]:
        raise ValueError("source changed during retirement")
    record.update(retired=True, verification={"runner": "mep-mepml run", "blocks": len(outcomes),
                  "report_sha256": digest(report), "run_sha256": digest(verification)})
    if not previously_retired:
        source.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="save inventory and current report presence")
    parser.add_argument("--retire", metavar="NAME", help="remove a verified superseded original")
    parser.add_argument("--verification", type=Path, help="mep-mepml run --report JSON for --retire")
    args = parser.parse_args()
    if args.retire and not args.verification:
        parser.error("--retire requires --verification")
    records = inventory()
    if args.retire:
        retire(records, args.retire, args.verification)
    total = len(records)
    present = sum((ROOT / r["report"]).is_file() for r in records.values())
    retired = sum(r["retired"] for r in records.values())
    if args.refresh or args.retire:
        LEDGER.write_text(json.dumps({"schema": 1, "total": total, "reports_present": present,
                          "retired_sources": retired, "examples": records}, indent=2) + "\n")
    print(f"NN sources: {total}; replacement reports present: {present}; retired originals: {retired}; reports still needed: {total - present}")


if __name__ == "__main__":
    main()
