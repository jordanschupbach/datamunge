#!/usr/bin/env python3
"""Stage the R package inputs without copying other languages' build trees."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stage = ROOT / "build/r/source/datamunger"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    inputs = [
        "DESCRIPTION",
        "NAMESPACE",
        "R",
        "include",
        "lib_sources.txt",
        "src/Makevars",
        "src/Makevars.win",
        "src/datamunge",
        "src/datamunge_r_wrap.cpp",
        "tests/testthat",
        "tests/testthat.R",
    ]
    inputs += [p.name for p in ROOT.glob("LICENSE*") if p.is_file()]
    for name in inputs:
        source, target = ROOT / name, stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(
                    "*.o", "*.so", "*.dll", "*.dylib", "__pycache__"
                ),
            )
        else:
            shutil.copy2(source, target)
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    subprocess.run(
        ["R", "CMD", "build", "--no-manual", "--no-build-vignettes", str(stage)],
        cwd=output,
        check=True,
    )


if __name__ == "__main__":
    main()
