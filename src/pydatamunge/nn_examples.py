"""NN example catalogue: genuine language ports and an explicit reference bridge.

Names in available(native_only=True) compose functional SWIG operations in
Python. Unported names are rejected by run(); run_reference is an explicit legacy tool.
Install with DATAMUNGE_BUILD_NN_EXAMPLES=1 or use ``just build-nn-python``.
"""

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from ._nn_catalog import EXAMPLES, NATIVE_EXAMPLES

_LOCK = threading.RLock()


def available(native_only=True):
    """Names of genuine Python implementations (or the canonical C++ inventory)."""
    return tuple(NATIVE_EXAMPLES if native_only else EXAMPLES)


@contextmanager
def _environment(arguments):
    previous = {
        key: os.environ.get(key)
        for key in ("DATAMUNGE_OUTPUT_DIR", "DATAMUNGE_EXAMPLES_FULL")
    }
    try:
        full = os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"
        if "--full" in arguments:
            full = True
        if "--quick" in arguments:
            full = False
        os.environ["DATAMUNGE_EXAMPLES_FULL"] = "1" if full else "0"
        if not os.environ.get("DATAMUNGE_OUTPUT_DIR"):
            os.environ["DATAMUNGE_OUTPUT_DIR"] = str(
                Path("output/nn/python") / ("full" if full else "quick")
            )
        yield [arg for arg in arguments if arg not in ("--full", "--quick")]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(name, arguments=None):
    """Run a native Python script; unported names never use the compiled bridge.

    ``arguments`` uses the original key=value options. If omitted, use sys.argv[1:].
    Quick mode is the default; --full opts into the original training schedule.
    Respect DATAMUNGE_DATA_DIR and DATAMUNGE_OUTPUT_DIR. Calls are serialized.
    No C++ executable is launched and no synthetic replacement model is used.
    """
    source = native_source(name)
    if source is None:
        if name not in EXAMPLES:
            raise ValueError(f"unknown NN example {name!r}")
        raise ValueError(f"{name!r} is not a native Python example yet; use the C++ reference explicitly")
    import runpy
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    with _LOCK, _environment(arguments) as native_arguments:
        return runpy.run_path(str(source))["main"](native_arguments)


def native_source(name):
    """The checked-out native Python script of `name`, or None.

    The script's own first-line marker ("# Native Python NN example") decides, so a port
    written after this package was installed runs without reinstalling; the baked
    catalogue only locates the file (and the layout is searched for new names).
    """
    root = Path(os.environ.get("DATAMUNGE_EXAMPLES_ROOT", Path.cwd()))
    tree = root / "examples/python/nn"
    if not tree.is_dir():
        if name in NATIVE_EXAMPLES:
            raise RuntimeError("Run language examples from the repository or set DATAMUNGE_EXAMPLES_ROOT")
        return None
    candidates = [tree / Path(EXAMPLES[name]).with_suffix(".py")] if name in EXAMPLES else []
    candidates += [p for p in tree.rglob(name + ".py") if "common" not in p.relative_to(tree).parts and p not in candidates]
    for source in candidates:
        if source.is_file() and source.read_text().startswith("# Native Python NN example"):
            return source
    return None


def run_reference(name, arguments=None):
    """Explicit legacy compiled-reference bridge; never used by python-example."""
    try:
        from . import _nn_examples
    except ImportError as error:
        raise RuntimeError("The optional compiled reference extension is not installed") from error
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    with _LOCK, _environment(arguments) as native_arguments:
        return _nn_examples.run_nn_example(name, native_arguments)
