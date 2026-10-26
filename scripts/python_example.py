#!/usr/bin/env python3
"""Launch a genuine Python NN script by basename; never execute C++ experiments."""
import argparse
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    matches = [
        p
        for p in (ROOT / "examples/python/nn").rglob("*.py")
        if p.stem == args.name
        and p.read_text().startswith("# Native Python NN example")
    ]
    if len(matches) != 1:
        parser.error(
            f"{args.name!r} has no unique native Python implementation yet; compiled references are not used"
        )
    os.chdir(ROOT)
    return runpy.run_path(str(matches[0]))["main"](args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
