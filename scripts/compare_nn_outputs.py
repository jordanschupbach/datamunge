#!/usr/bin/env python3
"""Compare persisted NN model outputs across C++, Python and R byte for byte."""
import argparse
import hashlib
import json
from pathlib import Path

LANGUAGES = ("cpp", "python", "r")


def compare(base, models, profile="quick", include_history=False):
    report = {"profile": profile, "models": {}, "passed": True}
    for model in models:
        directories = {language: base / language / profile / "models" / model for language in LANGUAGES}
        files = set()
        for directory in directories.values():
            files.update(path.relative_to(directory) for path in directory.rglob("*.npz"))
            if include_history:
                files.update(path.relative_to(directory) for path in directory.rglob("history.tsv"))
        entry = {"passed": any(path.suffix == ".npz" for path in files), "files": {}}
        for relative in sorted(files):
            digests = {}
            for language, directory in directories.items():
                path = directory / relative
                digests[language] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            same = None not in digests.values() and len(set(digests.values())) == 1
            entry["files"][relative.as_posix()] = {"byte_identical": same, "sha256": digests}
            entry["passed"] = entry["passed"] and same
        report["models"][model] = entry
        report["passed"] = report["passed"] and entry["passed"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="model directory names under each language's output root")
    parser.add_argument("--base", type=Path, default=Path("output/nn/language-migration"))
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if any(Path(model).is_absolute() or ".." in Path(model).parts for model in args.models):
        parser.error("models must be relative paths within the model output directory")
    report = compare(args.base, args.models, args.profile, args.include_history)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for model, entry in report["models"].items():
        print(f"{model}: {'PASS' if entry['passed'] else 'FAIL'} ({len(entry['files'])} files)")
        for name, file in entry["files"].items():
            if not file["byte_identical"]:
                print(f"  missing or different: {name}")
        if not any(Path(name).suffix == ".npz" for name in entry["files"]):
            print("  no persisted arrays found")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
