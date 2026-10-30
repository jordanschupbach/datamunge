#!/usr/bin/env python3
"""Launch a genuine Python NN script by basename; never execute C++ experiments."""
import argparse
import os
from pathlib import Path

from pydatamunge import nn_examples

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
    # The dispatcher scopes --quick/--full and output paths exactly as the R
    # launcher does, then passes only key=value options to the example's main.
    return nn_examples.run(args.name, args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
