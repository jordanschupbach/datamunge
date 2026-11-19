#!/usr/bin/env python3
"""Check the TensorFlow parity inventory against the two source trees."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/nn-tensorflow-coverage.tsv"
CHECKLIST = "plans/example_plans.mepml"
STATUSES = {
    "example": {"verified", "audit report", "find or add report"},
    "checklist": {"verified", "audit variant"},
}


def paths(cell):
    return [item.strip() for item in cell.split(";") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", nargs="?", type=Path,
                        default=Path.home() / "projects/tflow_examples")
    parser.add_argument("--check-media", action="store_true",
                        help="also check generated report media in a local build checkout")
    args = parser.parse_args()
    upstream = args.upstream.expanduser().resolve()
    docs = upstream / "docs"
    if not docs.is_dir() or not (upstream / CHECKLIST).is_file():
        parser.error(f"upstream checkout is missing docs/ or {CHECKLIST}: {upstream}")

    expected = {
        "example": {
            path.relative_to(upstream).as_posix()
            for path in docs.rglob("*.mepml")
            if path != docs / "README.mepml" and docs / "common" not in path.parents
        },
        "checklist": {
            f"{CHECKLIST}#L{number}"
            for number, line in enumerate((upstream / CHECKLIST).read_text(encoding="utf-8").splitlines(), 1)
            if re.match(r"^\s*- \[[x ]\]", line)
        },
    }
    target_pattern = re.compile(r"^\s*tf_example\(\s*([^\s)]+)", re.MULTILINE)
    declared_targets = Counter(
        target for cmake in (upstream / "src").rglob("CMakeLists.txt")
        for target in target_pattern.findall(cmake.read_text(encoding="utf-8"))
    )
    if not declared_targets:
        parser.error(f"no tf_example declarations under {upstream / 'src'}")
    mapped_targets = Counter()
    seen = {kind: set() for kind in expected}
    verified = Counter()
    verified_reports = set()
    errors = []
    with INVENTORY.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            kind, locator, status = row["kind"], row["upstream locator"], row["status"]
            if kind not in STATUSES or status not in STATUSES[kind]:
                errors.append(f"{locator}: unexpected inventory kind or status")
                continue
            if locator in seen[kind]:
                errors.append(f"{locator}: duplicate {kind} row")
            seen[kind].add(locator)
            if kind == "example" and row["source type"] == "build target":
                mapped_targets[Path(locator).stem] += 1
            if status != "verified":
                continue

            verified[kind] += 1
            reports = paths(row["datamunge report candidates"])
            if not reports or (kind == "example" and len(reports) != 1):
                errors.append(f"{locator}: expected a concrete Datamunge report mapping")
            for report in reports:
                target = (ROOT / report).resolve()
                if (not target.is_relative_to(ROOT / "docs/mepml/nn")
                        or target.suffix != ".mepml" or not target.is_file()):
                    errors.append(f"{locator}: invalid verified report {report}")
                else:
                    verified_reports.add(target)

    media_pattern = re.compile(r"\\(?:image|video)\((figures/[^)]+)\)")
    for report in sorted(verified_reports):
        source = report.read_text(encoding="utf-8")
        name = report.relative_to(ROOT)
        if not re.search(r"^\s*```\{", source, flags=re.MULTILINE):
            errors.append(f"{name}: no executable code block")
        if "// result_begin" not in source:
            errors.append(f"{name}: no captured results")
        if args.check_media:
            for media in set(media_pattern.findall(source)):
                if not (report.parent / media).is_file():
                    errors.append(f"{name}: missing local media {media}")

    for target in sorted(declared_targets.keys() | mapped_targets.keys()):
        if declared_targets[target] != mapped_targets[target]:
            errors.append(f"Build target {target}: {declared_targets[target]} CMake declarations, "
                          f"{mapped_targets[target]} inventory mappings")
    print(f"build targets: {sum(mapped_targets.values())}/{sum(declared_targets.values())} inventoried")
    print(f"mapped reports: {len(verified_reports)} distinct destinations")
    for kind in expected:
        errors.extend(f"Missing {kind} row: {item}" for item in sorted(expected[kind] - seen[kind]))
        errors.extend(f"Stale {kind} row: {item}" for item in sorted(seen[kind] - expected[kind]))
        print(f"{kind}: {verified[kind]}/{len(expected[kind])} marked verified")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")


if __name__ == "__main__":
    main()
