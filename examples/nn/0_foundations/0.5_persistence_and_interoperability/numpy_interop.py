#!/usr/bin/env python3
# CMD: nix develop .#python --command python examples/nn/0_foundations/0.5_persistence_and_interoperability/numpy_interop.py prepare
"""Prepare independent NumPy fixtures, or verify C++-written NPY/NPZ outputs.

Companion of numpy_weights.cpp (a port of tflow's scripts/numpy_interop.py,
without its TensorFlow import; only NumPy is needed):

python3 examples/nn/0_foundations/0.5_persistence_and_interoperability/numpy_interop.py prepare
just run-nn numpy_weights
python3 examples/nn/0_foundations/0.5_persistence_and_interoperability/numpy_interop.py verify

The default root is $DATAMUNGE_OUTPUT_DIR/fixtures/numpy (output/nn/fixtures/numpy).
"""
import argparse
import os
from pathlib import Path

import numpy as np


def main():
    default_root = Path(os.environ.get("DATAMUNGE_OUTPUT_DIR") or "output/nn") / "fixtures" / "numpy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3)
    integers = np.array([[1, 256, -7], [2**20, -(2**20), 42]], dtype=">i4")
    weights = np.array([[.75], [-1.5]], dtype=np.float32)
    bias = np.array([.25], dtype=np.float32)
    if args.action == "prepare":
        np.save(root / "fortran.npy", np.asfortranarray(matrix))
        np.save(root / "big_endian.npy", integers)
        for version in ((2, 0), (3, 0)):
            with (root / f"version{version[0]}.npy").open("wb") as out:
                np.lib.format.write_array(out, matrix, version=version, allow_pickle=False)
        np.save(root / "bytes.npy", np.array([0, 128, 255], dtype=np.uint8))
        np.savez(root / "python_weights.npz", weights=weights, bias=bias)
        np.savez_compressed(root / "python_compressed.npz", weights=weights, bias=bias)
        np.save(root / "object.npy", np.array([{"unsupported": 1}], dtype=object))
        print("NumPy fixtures: Fortran, big endian, NPY v2/v3, uint8, stored/DEFLATE NPZ, rejected object dtype")
    else:
        with np.load(root / "cpp_weights.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["weights"], weights)
            np.testing.assert_array_equal(arrays["bias"], bias)
        np.testing.assert_array_equal(np.load(root / "cpp_weights.npy", allow_pickle=False), weights)
        with np.load(root / "cpp_types.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["double"], matrix)
            np.testing.assert_array_equal(arrays["int32"], np.array([-2, 0, 17], dtype=np.int32))
            np.testing.assert_array_equal(arrays["int64"], np.array(2**40 + 7, dtype=np.int64))
            np.testing.assert_array_equal(arrays["bool"], np.array([False, True, True]))
            np.testing.assert_array_equal(arrays["float"], np.array([1.5, -0.25], dtype=np.float32))
            assert arrays["empty"].shape == (2, 0, 3)
            assert arrays["int64"].shape == ()
            assert arrays["double"].dtype == np.dtype("float64")
            assert arrays["float"].dtype == np.dtype("float32")
            assert arrays["bool"].dtype == np.dtype("bool")
        print("NumPy read every C++-written NPY/NPZ array: exact values, dtypes, scalar and empty shapes passed.")


if __name__ == "__main__":
    main()
