"""Runtime checks, quick/full mode and output paths (examples/nn/common/{checks,runtime,paths}.hpp)."""
import math
import os
from pathlib import Path

from pydatamunge.nn_workflows import data_root, output_root


def full():
    return os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"


def limit(requested, quick_limit):
    """The requested size in full mode, else at most quick_limit."""
    return requested if full() else min(requested, quick_limit)


def describe():
    if not full():
        print("Quick demonstration: bounded training; convergence metrics require DATAMUNGE_EXAMPLES_FULL=1.")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def training_quality(condition, message):
    """Long-budget quality checks are enforced only in full mode."""
    if full():
        require(condition, message)
    else:
        print("  full-training quality check deferred: " + message)


def near(actual, expected, tolerance, message):
    if not (math.isfinite(actual) and math.isfinite(expected)) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"{message}: got {actual}, expected {expected}")


def throws(function, message):
    """Expects function() to raise; returns the message."""
    try:
        function()
    except Exception as error:  # noqa: BLE001 - mirrors check::Throws
        return str(error)
    raise RuntimeError(message + ": expected an exception")


def header(title):
    print("\n== " + title + " " + "=" * max(0, 60 - len(title)))


def dataset(name):
    return data_root() / name


def output_dir(relative):
    """output_root() / relative, created as a directory."""
    directory = output_root(full()) / relative
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def tf_models(name):
    """tflow's recorded output directory for an example (its output/models/<name>)."""
    return Path(data_root()).parent / "models" / name
