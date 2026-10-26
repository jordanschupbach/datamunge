#!/usr/bin/env python3
"""Run Python/SWIG NN entry points with the C++ runner's dependencies and deadlines."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import run_nn_examples as runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    if "--build-only" in sys.argv:
        raise SystemExit("Build the optional extension with `just build-nn-python`.")
    catalogue = {
        name: source
        for name, source in runner.examples().items()
        if runner.CPP_EXAMPLES in source.parents
    }
    scripts = {
        name: ROOT
        / "examples/python/nn"
        / source.relative_to(runner.CPP_EXAMPLES).with_suffix(".py")
        for name, source in catalogue.items()
    }
    old_output = runner.OUTPUT_ROOT
    runner.OUTPUT_BASE = ROOT / "output/nn/python"
    runner.OUTPUT_ROOT = runner.OUTPUT_BASE / "quick"
    runner.REQUIRED_OUTPUTS = {
        name: tuple(runner.OUTPUT_ROOT / p.relative_to(old_output) for p in paths)
        for name, paths in runner.REQUIRED_OUTPUTS.items()
    }
    runner.BIN_DIR = ROOT / "build/nn-python/bin"
    runner.CACHE = ROOT / "build/nn-python/run-cache.tsv"
    runner.examples = lambda: catalogue
    native_run = runner.run_example
    native_key = runner.source_key

    def run_python(binary, arguments=(), timeout=180, env=None, log_path=None):
        # The child is a Python interpreter importing SWIG, never a C++ executable.
        return native_run(
            Path(sys.executable),
            ("-I", str(scripts[binary.name]), *arguments),
            timeout,
            env,
            log_path,
        )

    def source_key(source):
        digest = hashlib.sha256(native_key(source).encode())
        # Cached passes from a different dataset or output tree must not hide
        # missing galleries/checkpoints in the current run's destination.
        for variable in ("DATAMUNGE_DATA_DIR", "DATAMUNGE_OUTPUT_DIR"):
            digest.update(
                variable.encode() + b"=" + os.environ.get(variable, "").encode() + b"\0"
            )
        for path in (
            scripts[source.stem],
            ROOT / "src/pydatamunge/nn.py",
            ROOT / "src/pydatamunge/nn_workflows.py",
            ROOT / "prebindings/nn/src/nn.i",
            ROOT / "src/pydatamunge/nn_examples.py",
            ROOT / "scripts/prebuild_nn_examples.py",
            Path(__file__),
        ):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def library_file():
        # Match the isolated child interpreter, even when an old development
        # shell still exports PYTHONPATH=src. No C++ executable is involved.
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "from pydatamunge import _nn_native; print(_nn_native.__file__)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return Path(result.stdout.strip()) if result.returncode == 0 else None

    runner.run_example = run_python
    runner.source_key = source_key
    runner.library_file = library_file
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
