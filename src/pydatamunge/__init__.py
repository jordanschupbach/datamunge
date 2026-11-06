"""Python interface to the datamunge C++ library."""

from . import datamunge
from .linalg import Mat, SpMat, solve

# Preserve the established submodule API while making tensor entry points convenient.
NnTensor = datamunge.NnTensor
NnModel = datamunge.NnModel


def greet():
    return "Hello from the package!"


__all__ = ["Mat", "SpMat", "solve", "NnModel", "NnTensor", "datamunge", "greet"]
