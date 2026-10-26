"""Small data-path, command-line and tensor snapshot helpers for NN examples."""

import importlib.util
import os
import sys
from pathlib import Path

from . import nn


def options(arguments, defaults):
    result = dict(defaults)
    for argument in sys.argv[1:] if arguments is None else arguments:
        if argument in ("--quick", "--full"):
            result["full"] = argument == "--full"
            continue
        key, separator, value = argument.partition("=")
        if not separator or key not in result:
            raise ValueError("unknown option: " + argument)
        result[key] = type(result[key])(value)
    return result


def data_root():
    return Path(
        os.environ.get(
            "DATAMUNGE_DATA_DIR",
            Path.home() / "projects/tflow_examples/output/datasets",
        )
    )


def output_root(full=False):
    return Path(
        os.environ.get(
            "DATAMUNGE_OUTPUT_DIR", "output/nn/python/" + ("full" if full else "quick")
        )
    )


def arrays(store):
    return [(name, store.at(name).detach().clone()) for name in store.names()]


def helper(name):
    """Import examples/python/nn/common/<name>.py: shared model pieces and data readers.

    Helpers mirror examples/nn/common/*.hpp. They never own an end-to-end
    experiment; each example script keeps its own training and evaluation loops.
    """
    key = "pydatamunge_nn_example_common_" + name
    if key in sys.modules:
        return sys.modules[key]
    start = Path(os.environ.get("DATAMUNGE_EXAMPLES_ROOT", Path.cwd())).resolve()
    for root in (start, *start.parents):
        path = root / "examples/python/nn/common" / (name + ".py")
        if path.exists():
            break
    else:
        raise ImportError("no NN example helper named " + repr(name))
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module
