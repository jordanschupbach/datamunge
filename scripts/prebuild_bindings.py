#!/usr/bin/env python3
"""Generate binding sources before packaging; failed generation leaves old files intact.

Both come from Rosetta: Python from src/pydatamunge/rosetta, R from src/datamunger/rosetta."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_if_changed(source, destination):
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        source.replace(destination)


# The generated extension is split into this many sources so they compile in parallel
# (one source took over a minute; each part mostly re-parses the shared headers).
PYTHON_PARTS = 16
R_PARTS = 16


def generate_rosetta(header, policy, module, stem, parts, include_dirs=("include",), r_interface=None):
    """Generate src/<stem>*.cpp with Rosetta: the Python extension `module`, or with
    r_interface (a path) the R package `module` and its R6 classes at that path."""
    with tempfile.TemporaryDirectory(prefix=".rosetta-", dir=ROOT / "src") as tmp:
        generated = Path(tmp) / (stem + ".cpp")
        language = ["--language", "python"]
        if r_interface is not None:
            language = ["--language", "r", "--r-package", "--r-output", str(Path(tmp) / Path(r_interface).name)]
        subprocess.run(
            [
                os.environ.get("ROSETTA", "rosetta"),
                str(header),
                *language,
                "--module", module,
                "--policy", str(policy),
                # Relative to src/<stem>.cpp, so sdists build anywhere.
                "--include-as", Path(header).relative_to("src").as_posix(),
                "--skip-function-bodies",
                "--skip-unsupported",
                "--parts", str(parts),
                "-o", str(generated),
                "--", "-std=c++23", *("-I" + d for d in include_dirs),
            ],
            cwd=ROOT,
            check=True,
        )
        generated_parts = sorted(Path(tmp).glob(stem + "*.cpp"))
        for part in generated_parts:
            install_if_changed(part, ROOT / "src" / part.name)
        # Drop parts left over from a run with more of them.
        current = {part.name for part in generated_parts}
        for stale in (ROOT / "src").glob(stem + "*.cpp"):
            if stale.name not in current:
                stale.unlink()
        # The former SWIG directors header.
        (ROOT / "src" / (stem + ".h")).unlink(missing_ok=True)
        if r_interface is not None:
            install_if_changed(Path(tmp) / Path(r_interface).name, ROOT / r_interface)


def prebuild_python():
    """Generate src/datamunge_python_wrap*.cpp, the pydatamunge._datamunge extension."""
    bindings = Path("src/pydatamunge/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "pydatamunge._datamunge",
                     "datamunge_python_wrap", PYTHON_PARTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["python", "r"])
    args = parser.parse_args()
    if args.language == "python":
        prebuild_python()
        return
    prebuild_r()


def prebuild_r():
    """Generate src/datamunge_r_wrap*.cpp and R/datamunger.R, the datamunger package's R6 classes."""
    bindings = Path("src/datamunger/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunger", "datamunge_r_wrap", R_PARTS,
                     r_interface="R/datamunger.R")

if __name__ == "__main__":
    main()
