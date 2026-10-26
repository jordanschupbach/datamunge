#!/usr/bin/env python3
"""Generate SWIG sources before packaging; failed generation leaves old files intact."""

import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_if_changed(source, destination):
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        source.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["python", "r"])
    args = parser.parse_args()
    python = args.language == "python"
    module = "datamunge" if python else "datamunger"
    binding = "pydatamunge" if python else "datamunger"
    stem = "datamunge_python_wrap" if python else "datamunge_r_wrap"
    # Keep paths relative so generated files don't embed the checkout's absolute path.
    with tempfile.TemporaryDirectory(prefix=".swig-", dir=ROOT / "src") as tmp:
        stage = Path(tmp).relative_to(ROOT)
        subprocess.run(
            [
                "swig",
                *(["-doxygen"] if python else []),
                "-c++",
                "-" + args.language,
                "-Iinclude",
                "-outdir",
                str(stage),
                "-o",
                str(stage / (stem + ".cpp")),
                "-oh",
                str(stage / (stem + ".h")),
                f"src/{binding}/swig/{binding}.i",
            ],
            cwd=ROOT,
            check=True,
        )
        proxy = stage / (module + (".py" if python else ".R"))
        if not python:
            # R embeds the invocation in a comment; remove the random staging path.
            generated_proxy = ROOT / proxy
            generated_proxy.write_text(
                generated_proxy.read_text().replace(str(stage), "src")
            )
        destination = ROOT / (f"src/{binding}" if python else "R") / proxy.name
        install_if_changed(ROOT / proxy, destination)
        for suffix in [".cpp", ".h"]:
            generated = ROOT / stage / (stem + suffix)
            # SWIG only emits a header when directors require one.
            if generated.exists():
                install_if_changed(generated, ROOT / "src" / generated.name)


if __name__ == "__main__":
    main()
