"""Exercise ownership and error translation through the installed SWIG extension."""

import gc
from pathlib import Path

import pytest

from pydatamunge import datamunge as dm


def test_tensor_owns_data_after_inputs_are_collected():
    x = dm.NnTensor([1, 2, 3, 4], [2, 2], "float64")
    result = x.transpose().add(x)
    del x
    gc.collect()
    assert list(result.values()) == [2, 5, 5, 8]
    with pytest.raises(RuntimeError):
        result.reshape([3])


def test_import_is_an_installed_package():
    # Detect the old source-tree/Nix shadowing bug in the local wheel workflow.
    import pydatamunge

    assert pydatamunge.NnTensor is dm.NnTensor
    assert pydatamunge.NnModel is dm.NnModel
    assert (
        Path(pydatamunge.__file__).resolve().parent != Path("src/pydatamunge").resolve()
    )
